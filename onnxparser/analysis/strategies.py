# -*- coding: utf-8 -*-
"""Memory allocation strategies - all strategies support memory constraints"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_analyzer import TensorInfo, MemoryConstraint


class MemoryExceededError(Exception):
    """Raised when memory limit is exceeded"""
    def __init__(self, required: int, available: int, limit: int):
        self.required = required
        self.available = available
        self.limit = limit
        super().__init__(
            f"Memory limit exceeded: need {required/1e6:.2f} MB, "
            f"available {available/1e6:.2f} MB, limit {limit/1e6:.2f} MB"
        )


@dataclass
class AllocationResult:
    """Result of memory allocation"""
    tensor_name: str
    offset: int
    size: int
    reused_from: Optional[str] = None
    is_inplace: bool = False
    exceeded_limit: bool = False  # True if allocation exceeded memory limit


@dataclass
class MemoryBlock:
    """A block in the memory pool"""
    offset: int
    size: int
    is_free: bool = True
    tensor_name: Optional[str] = None

    def can_fit(self, required: int) -> bool:
        return self.is_free and self.size >= required


class AllocationStrategy(ABC):
    """Base class for memory allocation strategies"""

    def __init__(self):
        self._memory_limit: Optional[int] = None
        self._exceeded_count: int = 0
        self._total_exceeded_bytes: int = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset strategy state"""
        self._memory_limit = None
        self._exceeded_count = 0
        self._total_exceeded_bytes = 0

    @abstractmethod
    def allocate(
        self,
        tensor: 'TensorInfo',
        live_tensors: Dict[str, 'TensorInfo'],
        constraint: 'MemoryConstraint'
    ) -> AllocationResult:
        """Allocate memory for a tensor"""
        pass

    @abstractmethod
    def deallocate(self, tensor_name: str) -> None:
        """Free memory for a tensor"""
        pass

    @abstractmethod
    def current_memory(self) -> int:
        """Get current memory usage"""
        pass

    def set_memory_limit(self, limit: Optional[int]) -> None:
        """Set memory limit for constraint checking"""
        self._memory_limit = limit

    def check_memory_limit(self, required: int) -> Tuple[bool, int]:
        """
        Check if allocation would exceed memory limit.
        Returns (would_exceed, available_bytes)
        """
        if self._memory_limit is None:
            return False, float('inf')

        current = self.current_memory()
        available = self._memory_limit - current
        would_exceed = required > available
        return would_exceed, max(0, available)

    def record_exceeded(self, bytes_exceeded: int) -> None:
        """Record that memory limit was exceeded"""
        self._exceeded_count += 1
        self._total_exceeded_bytes += bytes_exceeded

    @property
    def exceeded_stats(self) -> Dict:
        """Get statistics about memory limit violations"""
        return {
            "exceeded_count": self._exceeded_count,
            "total_exceeded_bytes": self._total_exceeded_bytes,
        }

    @staticmethod
    def _align(size: int, alignment: int) -> int:
        """Align size to specified boundary"""
        if alignment <= 1:
            return size
        return ((size + alignment - 1) // alignment) * alignment


class StrategyRegistry:
    """Registry for allocation strategies"""

    _strategies: Dict[str, Type[AllocationStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy"""
        def decorator(strategy_cls: Type[AllocationStrategy]):
            cls._strategies[name] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> AllocationStrategy:
        """Get a strategy instance by name"""
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
        return cls._strategies[name]()

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all available strategies"""
        return list(cls._strategies.keys())


@StrategyRegistry.register("no_reuse")
class NoReuseStrategy(AllocationStrategy):
    """
    No memory reuse - calculates maximum memory requirement.
    With memory constraint: tracks when limit would be exceeded.
    """

    def __init__(self):
        super().__init__()
        self._next_offset = 0
        self._allocations: Dict[str, int] = {}
        self._active_memory = 0

    @property
    def name(self) -> str:
        return "no_reuse"

    def reset(self) -> None:
        super().reset()
        self._next_offset = 0
        self._allocations = {}
        self._active_memory = 0

    def allocate(self, tensor, live_tensors, constraint) -> AllocationResult:
        self.set_memory_limit(constraint.max_memory_bytes)
        size = self._align(tensor.size_bytes, constraint.alignment)

        # Check memory limit
        exceeded, available = self.check_memory_limit(size)
        if exceeded:
            self.record_exceeded(size - available)

        offset = self._next_offset
        self._next_offset += size
        self._allocations[tensor.name] = size
        self._active_memory += size

        return AllocationResult(
            tensor_name=tensor.name,
            offset=offset,
            size=size,
            reused_from=None,
            exceeded_limit=exceeded,
        )

    def deallocate(self, tensor_name: str) -> None:
        if tensor_name in self._allocations:
            self._active_memory -= self._allocations[tensor_name]
            # Note: in no_reuse, memory is never actually freed for reuse

    def current_memory(self) -> int:
        return self._active_memory


@StrategyRegistry.register("greedy")
class GreedyReuseStrategy(AllocationStrategy):
    """
    Greedy memory reuse - Best-fit strategy.
    With memory constraint: prioritizes reuse when approaching limit.
    """

    def __init__(self):
        super().__init__()
        self._memory_pool: List[MemoryBlock] = []
        self._peak_memory = 0

    @property
    def name(self) -> str:
        return "greedy"

    def reset(self) -> None:
        super().reset()
        self._memory_pool = []
        self._peak_memory = 0

    def allocate(self, tensor, live_tensors, constraint) -> AllocationResult:
        self.set_memory_limit(constraint.max_memory_bytes)
        required = self._align(tensor.size_bytes, constraint.alignment)

        # Check memory limit before allocation
        exceeded, available = self.check_memory_limit(required)

        # Try to find a reusable block (best-fit)
        best_block = self._find_best_fit(required)

        if best_block is not None:
            # Reuse existing block
            reused_from = best_block.tensor_name
            best_block.is_free = False
            best_block.tensor_name = tensor.name

            return AllocationResult(
                tensor_name=tensor.name,
                offset=best_block.offset,
                size=required,
                reused_from=reused_from,
                exceeded_limit=False,  # Reuse doesn't exceed
            )

        # No reusable block found - must allocate new
        if exceeded:
            self.record_exceeded(required - available)

        offset = sum(b.size for b in self._memory_pool)
        self._memory_pool.append(MemoryBlock(
            offset=offset,
            size=required,
            is_free=False,
            tensor_name=tensor.name,
        ))

        current = self.current_memory()
        self._peak_memory = max(self._peak_memory, current)

        return AllocationResult(
            tensor_name=tensor.name,
            offset=offset,
            size=required,
            exceeded_limit=exceeded,
        )

    def _find_best_fit(self, required: int) -> Optional[MemoryBlock]:
        """Find the smallest free block that fits the requirement"""
        best_block = None
        for block in self._memory_pool:
            if block.can_fit(required):
                if best_block is None or block.size < best_block.size:
                    best_block = block
        return best_block

    def deallocate(self, tensor_name: str) -> None:
        for block in self._memory_pool:
            if block.tensor_name == tensor_name and not block.is_free:
                block.is_free = True
                break

    def current_memory(self) -> int:
        return sum(b.size for b in self._memory_pool if not b.is_free)


@StrategyRegistry.register("inplace")
class InplaceStrategy(AllocationStrategy):
    """
    Prefer in-place operations when possible.
    With memory constraint: more aggressively seeks in-place opportunities.
    """

    INPLACE_OPS = {
        'relu', 'relu_', 'sigmoid', 'sigmoid_', 'tanh', 'tanh_',
        'dropout', 'dropout_', 'gelu', 'silu', 'leaky_relu',
        'add_', 'mul_', 'sub_', 'div_', 'neg', 'abs', 'sqrt',
    }

    def __init__(self):
        super().__init__()
        self._greedy = GreedyReuseStrategy()
        self._inplace_allocations: Dict[str, str] = {}  # tensor -> inplace_from

    @property
    def name(self) -> str:
        return "inplace"

    def reset(self) -> None:
        super().reset()
        self._greedy.reset()
        self._inplace_allocations = {}

    def allocate(self, tensor, live_tensors, constraint) -> AllocationResult:
        self.set_memory_limit(constraint.max_memory_bytes)
        self._greedy.set_memory_limit(constraint.max_memory_bytes)

        # Check memory pressure
        exceeded, available = self.check_memory_limit(tensor.size_bytes)

        # Try in-place first (especially important when under memory pressure)
        inplace_candidate = self._find_inplace_candidate(tensor, live_tensors, constraint)

        if inplace_candidate is not None:
            self._inplace_allocations[tensor.name] = inplace_candidate.name
            return AllocationResult(
                tensor_name=tensor.name,
                offset=inplace_candidate.memory_offset,
                size=tensor.size_bytes,
                reused_from=inplace_candidate.name,
                is_inplace=True,
                exceeded_limit=False,
            )

        # Fall back to greedy strategy
        result = self._greedy.allocate(tensor, live_tensors, constraint)

        # Inherit exceeded stats
        if result.exceeded_limit:
            self.record_exceeded(tensor.size_bytes - available)

        return result

    def _find_inplace_candidate(
        self,
        tensor,
        live_tensors: Dict[str, 'TensorInfo'],
        constraint: 'MemoryConstraint'
    ) -> Optional['TensorInfo']:
        """Find a tensor whose memory can be reused in-place"""

        # Check if operation supports in-place
        can_inplace = self._can_inplace(tensor)

        # Under memory pressure, be more aggressive about in-place
        exceeded, _ = self.check_memory_limit(tensor.size_bytes)
        if exceeded and constraint.prefer_inplace:
            can_inplace = True

        if not can_inplace:
            return None

        # Find compatible tensor
        for input_tensor in live_tensors.values():
            if self._is_compatible(tensor, input_tensor):
                return input_tensor

        return None

    def _can_inplace(self, tensor) -> bool:
        """Check if tensor operation can be done in-place"""
        op_name = tensor.name.lower()
        for inplace_op in self.INPLACE_OPS:
            if inplace_op in op_name:
                return True
        return False

    def _is_compatible(self, tensor, input_tensor) -> bool:
        """Check if tensor can reuse input_tensor's memory"""
        # Same size or smaller
        if tensor.size_bytes > input_tensor.size_bytes:
            return False
        # Input is about to die (last use is current step)
        if input_tensor.death_step != tensor.birth_step:
            return False
        # Must have valid memory offset
        if input_tensor.memory_offset < 0:
            return False
        return True

    def deallocate(self, tensor_name: str) -> None:
        if tensor_name in self._inplace_allocations:
            del self._inplace_allocations[tensor_name]
        else:
            self._greedy.deallocate(tensor_name)

    def current_memory(self) -> int:
        return self._greedy.current_memory()


@StrategyRegistry.register("best_fit")
class BestFitStrategy(AllocationStrategy):
    """
    Best-fit allocation with memory defragmentation hints.
    With memory constraint: coalesces free blocks when approaching limit.
    """

    def __init__(self):
        super().__init__()
        self._memory_pool: List[MemoryBlock] = []
        self._total_allocated = 0

    @property
    def name(self) -> str:
        return "best_fit"

    def reset(self) -> None:
        super().reset()
        self._memory_pool = []
        self._total_allocated = 0

    def allocate(self, tensor, live_tensors, constraint) -> AllocationResult:
        self.set_memory_limit(constraint.max_memory_bytes)
        required = self._align(tensor.size_bytes, constraint.alignment)

        exceeded, available = self.check_memory_limit(required)

        # Try to find best-fit block
        best_block = self._find_best_fit(required)

        # If approaching memory limit, try to coalesce free blocks
        if best_block is None and exceeded:
            self._coalesce_free_blocks()
            best_block = self._find_best_fit(required)

        if best_block is not None:
            reused_from = best_block.tensor_name
            best_block.is_free = False
            best_block.tensor_name = tensor.name

            return AllocationResult(
                tensor_name=tensor.name,
                offset=best_block.offset,
                size=required,
                reused_from=reused_from,
                exceeded_limit=False,
            )

        # Allocate new block
        if exceeded:
            self.record_exceeded(required - available)

        offset = self._total_allocated
        self._memory_pool.append(MemoryBlock(
            offset=offset,
            size=required,
            is_free=False,
            tensor_name=tensor.name,
        ))
        self._total_allocated += required

        return AllocationResult(
            tensor_name=tensor.name,
            offset=offset,
            size=required,
            exceeded_limit=exceeded,
        )

    def _find_best_fit(self, required: int) -> Optional[MemoryBlock]:
        """Find smallest free block that fits"""
        best = None
        for block in self._memory_pool:
            if block.can_fit(required):
                if best is None or block.size < best.size:
                    best = block
        return best

    def _coalesce_free_blocks(self) -> None:
        """Merge adjacent free blocks (logical coalescing)"""
        # Sort by offset
        self._memory_pool.sort(key=lambda b: b.offset)

        # Merge adjacent free blocks
        i = 0
        while i < len(self._memory_pool) - 1:
            current = self._memory_pool[i]
            next_block = self._memory_pool[i + 1]

            if current.is_free and next_block.is_free:
                # Merge
                current.size += next_block.size
                self._memory_pool.pop(i + 1)
            else:
                i += 1

    def deallocate(self, tensor_name: str) -> None:
        for block in self._memory_pool:
            if block.tensor_name == tensor_name and not block.is_free:
                block.is_free = True
                break

    def current_memory(self) -> int:
        return sum(b.size for b in self._memory_pool if not b.is_free)


@StrategyRegistry.register("first_fit")
class FirstFitStrategy(AllocationStrategy):
    """
    First-fit allocation - faster but may fragment more.
    With memory constraint: falls back to best-fit when constrained.
    """

    def __init__(self):
        super().__init__()
        self._memory_pool: List[MemoryBlock] = []
        self._total_allocated = 0

    @property
    def name(self) -> str:
        return "first_fit"

    def reset(self) -> None:
        super().reset()
        self._memory_pool = []
        self._total_allocated = 0

    def allocate(self, tensor, live_tensors, constraint) -> AllocationResult:
        self.set_memory_limit(constraint.max_memory_bytes)
        required = self._align(tensor.size_bytes, constraint.alignment)

        exceeded, available = self.check_memory_limit(required)

        # First-fit: find first block that fits
        found_block = None
        for block in self._memory_pool:
            if block.can_fit(required):
                found_block = block
                break

        # If constrained and no first-fit, try best-fit
        if found_block is None and exceeded:
            found_block = self._find_best_fit(required)

        if found_block is not None:
            reused_from = found_block.tensor_name
            found_block.is_free = False
            found_block.tensor_name = tensor.name

            return AllocationResult(
                tensor_name=tensor.name,
                offset=found_block.offset,
                size=required,
                reused_from=reused_from,
                exceeded_limit=False,
            )

        # Allocate new
        if exceeded:
            self.record_exceeded(required - available)

        offset = self._total_allocated
        self._memory_pool.append(MemoryBlock(
            offset=offset,
            size=required,
            is_free=False,
            tensor_name=tensor.name,
        ))
        self._total_allocated += required

        return AllocationResult(
            tensor_name=tensor.name,
            offset=offset,
            size=required,
            exceeded_limit=exceeded,
        )

    def _find_best_fit(self, required: int) -> Optional[MemoryBlock]:
        """Fallback to best-fit when constrained"""
        best = None
        for block in self._memory_pool:
            if block.can_fit(required):
                if best is None or block.size < best.size:
                    best = block
        return best

    def deallocate(self, tensor_name: str) -> None:
        for block in self._memory_pool:
            if block.tensor_name == tensor_name and not block.is_free:
                block.is_free = True
                break

    def current_memory(self) -> int:
        return sum(b.size for b in self._memory_pool if not b.is_free)


@StrategyRegistry.register("static")
class StaticAllocationStrategy(AllocationStrategy):
    """
    Static memory allocation - pre-computes fixed offsets at compile time.
    No runtime memory pool management, each tensor gets a fixed address.

    Uses interval coloring algorithm to minimize total memory while
    ensuring non-overlapping lifetimes share the same memory region.
    """

    def __init__(self):
        super().__init__()
        # Pre-computed static offsets: tensor_name -> (offset, size)
        self._static_offsets: Dict[str, Tuple[int, int]] = {}
        self._total_memory = 0
        self._active_memory = 0
        self._active_tensors: set = set()

    @property
    def name(self) -> str:
        return "static"

    def reset(self) -> None:
        super().reset()
        self._static_offsets = {}
        self._total_memory = 0
        self._active_memory = 0
        self._active_tensors = set()

    def precompute_offsets(
        self,
        tensors: List['TensorInfo'],
        alignment: int = 64,
    ) -> Dict[str, int]:
        """
        Pre-compute static memory offsets for all tensors.
        Uses interval graph coloring to find optimal placement.

        Returns: Dict mapping tensor_name -> fixed_offset
        """
        if not tensors:
            return {}

        # Sort by birth_step for deterministic ordering
        sorted_tensors = sorted(tensors, key=lambda t: (t.birth_step, -t.size_bytes))

        # Track allocated regions: list of (offset, end_offset, death_step)
        allocated_regions: List[Tuple[int, int, int]] = []

        for tensor in sorted_tensors:
            size = self._align(tensor.size_bytes, alignment)
            birth = tensor.birth_step
            death = tensor.death_step

            # Find a gap where this tensor can fit
            # Remove expired regions first
            allocated_regions = [
                (off, end, d) for off, end, d in allocated_regions
                if d >= birth  # Keep regions that are still alive
            ]

            # Sort by offset to find gaps
            allocated_regions.sort(key=lambda r: r[0])

            # Find first gap that fits
            best_offset = None
            current_pos = 0

            for region_offset, region_end, _ in allocated_regions:
                if current_pos + size <= region_offset:
                    # Found a gap
                    best_offset = current_pos
                    break
                current_pos = max(current_pos, region_end)

            if best_offset is None:
                # No gap found, allocate at the end
                best_offset = current_pos

            # Record this allocation
            self._static_offsets[tensor.name] = (best_offset, size)
            allocated_regions.append((best_offset, best_offset + size, death))
            self._total_memory = max(self._total_memory, best_offset + size)

        return {name: offset for name, (offset, _) in self._static_offsets.items()}

    def allocate(self, tensor, live_tensors, constraint) -> AllocationResult:
        self.set_memory_limit(constraint.max_memory_bytes)
        size = self._align(tensor.size_bytes, constraint.alignment)

        # Check if we have pre-computed offset
        if tensor.name in self._static_offsets:
            offset, _ = self._static_offsets[tensor.name]
        else:
            # Fallback: allocate at next available position
            offset = self._total_memory
            self._static_offsets[tensor.name] = (offset, size)
            self._total_memory += size

        # Check memory limit
        exceeded, available = self.check_memory_limit(size)
        if exceeded:
            self.record_exceeded(size - available)

        self._active_tensors.add(tensor.name)
        self._active_memory += size

        return AllocationResult(
            tensor_name=tensor.name,
            offset=offset,
            size=size,
            reused_from=None,  # Static allocation, no runtime reuse concept
            exceeded_limit=exceeded,
        )

    def deallocate(self, tensor_name: str) -> None:
        if tensor_name in self._active_tensors:
            self._active_tensors.remove(tensor_name)
            if tensor_name in self._static_offsets:
                _, size = self._static_offsets[tensor_name]
                self._active_memory -= size

    def current_memory(self) -> int:
        return self._active_memory

    def get_memory_layout(self) -> Dict[str, Dict]:
        """
        Get the complete static memory layout.
        Returns dict with tensor_name -> {offset, size} for code generation.
        """
        return {
            name: {"offset": offset, "size": size}
            for name, (offset, size) in self._static_offsets.items()
        }

    def total_static_memory(self) -> int:
        """Get total static memory required"""
        return self._total_memory

    def generate_memory_map(self) -> str:
        """
        Generate C-style memory map for static allocation.
        Useful for embedded systems or hardware code generation.
        """
        lines = [
            "// Static Memory Layout",
            f"// Total Memory Required: {self._total_memory} bytes",
            "",
            "#define MEMORY_POOL_SIZE {}".format(self._total_memory),
            "",
            "// Tensor Offsets",
        ]

        for name, (offset, size) in sorted(
            self._static_offsets.items(), key=lambda x: x[1][0]
        ):
            safe_name = name.upper().replace(".", "_").replace("-", "_")
            lines.append(f"#define OFFSET_{safe_name} {offset}  // size: {size}")

        return "\n".join(lines)
