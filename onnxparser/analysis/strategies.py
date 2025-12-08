# -*- coding: utf-8 -*-
"""Memory allocation strategies - all use static pre-computation with different reuse algorithms"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_analyzer import TensorInfo


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
    exceeded_limit: bool = False


@dataclass
class MemoryBlock:
    """A memory block with lifetime info"""
    offset: int
    size: int
    tensor_name: str
    birth_step: int
    death_step: int


class AllocationStrategy(ABC):
    """
    Base class for static memory allocation strategies.
    All strategies pre-compute fixed offsets at compile time.
    """

    def __init__(self):
        self._memory_limit: Optional[int] = None
        self._exceeded_count: int = 0
        self._total_exceeded_bytes: int = 0
        # Static allocation state
        self._static_offsets: Dict[str, Tuple[int, int]] = {}  # name -> (offset, size)
        self._reused_from: Dict[str, str] = {}  # name -> reused_from_name
        self._total_memory = 0
        self._active_memory = 0
        self._active_tensors: set = set()

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass

    def reset(self) -> None:
        """Reset strategy state"""
        self._memory_limit = None
        self._exceeded_count = 0
        self._total_exceeded_bytes = 0
        self._static_offsets = {}
        self._reused_from = {}
        self._total_memory = 0
        self._active_memory = 0
        self._active_tensors = set()

    def set_memory_limit(self, limit: Optional[int]) -> None:
        """Set memory limit for constraint checking"""
        self._memory_limit = limit

    def check_memory_limit(self, required: int) -> Tuple[bool, int]:
        """Check if allocation would exceed memory limit"""
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

    def precompute_offsets(
        self,
        tensors: List['TensorInfo'],
        alignment: int = 64,
    ) -> Dict[str, int]:
        """
        Pre-compute static memory offsets for all tensors.
        Different strategies override _find_offset() to implement different algorithms.
        """
        if not tensors:
            return {}

        self.reset()

        # Sort by birth_step, then by size (larger first for better packing)
        sorted_tensors = sorted(tensors, key=lambda t: (t.birth_step, -t.size_bytes))

        # Track allocated blocks with their lifetimes
        allocated_blocks: List[MemoryBlock] = []

        for tensor in sorted_tensors:
            size = self._align(tensor.size_bytes, alignment)
            birth = tensor.birth_step
            death = tensor.death_step

            # Find offset using strategy-specific algorithm
            offset, reused_from = self._find_offset(
                size, birth, death, allocated_blocks
            )

            # Record allocation
            self._static_offsets[tensor.name] = (offset, size)
            if reused_from:
                self._reused_from[tensor.name] = reused_from

            allocated_blocks.append(MemoryBlock(
                offset=offset,
                size=size,
                tensor_name=tensor.name,
                birth_step=birth,
                death_step=death,
            ))

            self._total_memory = max(self._total_memory, offset + size)

        return {name: offset for name, (offset, _) in self._static_offsets.items()}

    @abstractmethod
    def _find_offset(
        self,
        size: int,
        birth: int,
        death: int,
        allocated_blocks: List[MemoryBlock],
    ) -> Tuple[int, Optional[str]]:
        """
        Find memory offset for a tensor. Strategy-specific algorithm.
        Returns: (offset, reused_from_tensor_name or None)
        """
        pass

    def allocate(self, tensor, live_tensors, constraint) -> AllocationResult:
        """Allocate using pre-computed offset"""
        self.set_memory_limit(constraint.max_memory_bytes)
        size = self._align(tensor.size_bytes, constraint.alignment)

        if tensor.name in self._static_offsets:
            offset, _ = self._static_offsets[tensor.name]
            reused_from = self._reused_from.get(tensor.name)
        else:
            # Fallback if not pre-computed
            offset = self._total_memory
            self._static_offsets[tensor.name] = (offset, size)
            self._total_memory += size
            reused_from = None

        exceeded, available = self.check_memory_limit(size)
        if exceeded:
            self.record_exceeded(size - available)

        self._active_tensors.add(tensor.name)
        self._active_memory += size

        return AllocationResult(
            tensor_name=tensor.name,
            offset=offset,
            size=size,
            reused_from=reused_from,
            exceeded_limit=exceeded,
        )

    def deallocate(self, tensor_name: str) -> None:
        """Deallocate tensor"""
        if tensor_name in self._active_tensors:
            self._active_tensors.remove(tensor_name)
            if tensor_name in self._static_offsets:
                _, size = self._static_offsets[tensor_name]
                self._active_memory -= size

    def current_memory(self) -> int:
        """Get current active memory"""
        return self._active_memory

    def get_memory_layout(self) -> Dict[str, Dict]:
        """Get complete static memory layout for code generation"""
        return {
            name: {"offset": offset, "size": size}
            for name, (offset, size) in self._static_offsets.items()
        }

    def total_static_memory(self) -> int:
        """Get total static memory required"""
        return self._total_memory

    def generate_memory_map(self) -> str:
        """Generate C-style memory map for code generation"""
        lines = [
            "// Static Memory Layout",
            f"// Total Memory Required: {self._total_memory} bytes",
            f"// Strategy: {self.name}",
            "",
            f"#define MEMORY_POOL_SIZE {self._total_memory}",
            "",
            "// Tensor Offsets",
        ]

        for name, (offset, size) in sorted(
            self._static_offsets.items(), key=lambda x: x[1][0]
        ):
            safe_name = name.upper().replace(".", "_").replace("-", "_")
            reuse_comment = ""
            if name in self._reused_from:
                reuse_comment = f" (reuses {self._reused_from[name]})"
            lines.append(f"#define OFFSET_{safe_name} {offset}  // size: {size}{reuse_comment}")

        return "\n".join(lines)


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
    No memory reuse - each tensor gets unique address.
    Useful for calculating maximum memory requirement.
    """

    @property
    def name(self) -> str:
        return "no_reuse"

    def _find_offset(
        self,
        size: int,
        birth: int,
        death: int,
        allocated_blocks: List[MemoryBlock],
    ) -> Tuple[int, Optional[str]]:
        """Never reuse - always allocate at end"""
        if not allocated_blocks:
            return 0, None
        # Allocate after all existing blocks
        max_end = max(b.offset + b.size for b in allocated_blocks)
        return max_end, None


@StrategyRegistry.register("greedy")
class GreedyReuseStrategy(AllocationStrategy):
    """
    Greedy reuse with best-fit algorithm.
    Finds smallest available gap that fits the tensor.
    """

    @property
    def name(self) -> str:
        return "greedy"

    def _find_offset(
        self,
        size: int,
        birth: int,
        death: int,
        allocated_blocks: List[MemoryBlock],
    ) -> Tuple[int, Optional[str]]:
        """Best-fit: find smallest gap that fits"""
        if not allocated_blocks:
            return 0, None

        # Find blocks whose lifetime doesn't overlap
        available_blocks = [
            b for b in allocated_blocks
            if b.death_step < birth  # Block is dead before we're born
        ]

        # Best-fit: find smallest block that fits
        best_block = None
        for block in available_blocks:
            if block.size >= size:
                if best_block is None or block.size < best_block.size:
                    best_block = block

        if best_block:
            return best_block.offset, best_block.tensor_name

        # No reusable block - find gap or allocate at end
        return self._find_gap_or_end(size, birth, allocated_blocks)

    def _find_gap_or_end(
        self,
        size: int,
        birth: int,
        allocated_blocks: List[MemoryBlock],
    ) -> Tuple[int, Optional[str]]:
        """Find a gap in memory or allocate at end"""
        # Get live blocks at birth time
        live_blocks = [
            b for b in allocated_blocks
            if b.birth_step <= birth <= b.death_step
        ]

        if not live_blocks:
            return 0, None

        # Sort by offset and find gaps
        live_blocks.sort(key=lambda b: b.offset)

        # Check gap at beginning
        if live_blocks[0].offset >= size:
            return 0, None

        # Check gaps between blocks
        for i in range(len(live_blocks) - 1):
            gap_start = live_blocks[i].offset + live_blocks[i].size
            gap_end = live_blocks[i + 1].offset
            if gap_end - gap_start >= size:
                return gap_start, None

        # Allocate at end
        max_end = max(b.offset + b.size for b in live_blocks)
        return max_end, None


@StrategyRegistry.register("best_fit")
class BestFitStrategy(AllocationStrategy):
    """
    Best-fit allocation - finds smallest fitting gap.
    Same as greedy but with different naming for clarity.
    """

    @property
    def name(self) -> str:
        return "best_fit"

    def _find_offset(
        self,
        size: int,
        birth: int,
        death: int,
        allocated_blocks: List[MemoryBlock],
    ) -> Tuple[int, Optional[str]]:
        """Best-fit with gap searching"""
        if not allocated_blocks:
            return 0, None

        # Find reusable blocks (dead before we're born)
        reusable = [
            b for b in allocated_blocks
            if b.death_step < birth and b.size >= size
        ]

        # Best-fit: smallest fitting block
        if reusable:
            best = min(reusable, key=lambda b: b.size)
            return best.offset, best.tensor_name

        # Find gaps among live blocks
        live_blocks = [
            b for b in allocated_blocks
            if not (b.death_step < birth)  # Still alive or overlaps
        ]

        if not live_blocks:
            return 0, None

        live_blocks.sort(key=lambda b: b.offset)

        # Find best-fit gap
        best_gap = None
        best_gap_size = float('inf')

        # Gap at start
        if live_blocks[0].offset >= size:
            if live_blocks[0].offset < best_gap_size:
                best_gap = 0
                best_gap_size = live_blocks[0].offset

        # Gaps between blocks
        for i in range(len(live_blocks) - 1):
            gap_start = live_blocks[i].offset + live_blocks[i].size
            gap_size = live_blocks[i + 1].offset - gap_start
            if gap_size >= size and gap_size < best_gap_size:
                best_gap = gap_start
                best_gap_size = gap_size

        if best_gap is not None:
            return best_gap, None

        # Allocate at end
        max_end = max(b.offset + b.size for b in live_blocks)
        return max_end, None


@StrategyRegistry.register("first_fit")
class FirstFitStrategy(AllocationStrategy):
    """
    First-fit allocation - finds first fitting gap.
    Faster than best-fit but may fragment more.
    """

    @property
    def name(self) -> str:
        return "first_fit"

    def _find_offset(
        self,
        size: int,
        birth: int,
        death: int,
        allocated_blocks: List[MemoryBlock],
    ) -> Tuple[int, Optional[str]]:
        """First-fit: find first available slot"""
        if not allocated_blocks:
            return 0, None

        # Find reusable blocks sorted by offset (first-fit)
        reusable = sorted(
            [b for b in allocated_blocks if b.death_step < birth and b.size >= size],
            key=lambda b: b.offset
        )

        if reusable:
            return reusable[0].offset, reusable[0].tensor_name

        # Find first gap among live blocks
        live_blocks = [
            b for b in allocated_blocks
            if not (b.death_step < birth)
        ]

        if not live_blocks:
            return 0, None

        live_blocks.sort(key=lambda b: b.offset)

        # Check gap at start
        if live_blocks[0].offset >= size:
            return 0, None

        # Find first fitting gap
        for i in range(len(live_blocks) - 1):
            gap_start = live_blocks[i].offset + live_blocks[i].size
            gap_size = live_blocks[i + 1].offset - gap_start
            if gap_size >= size:
                return gap_start, None

        # Allocate at end
        max_end = max(b.offset + b.size for b in live_blocks)
        return max_end, None


@StrategyRegistry.register("inplace")
class InplaceStrategy(AllocationStrategy):
    """
    In-place allocation - prioritizes reusing input tensor memory.
    Falls back to best-fit for non-inplace operations.
    """

    INPLACE_OPS = {
        'relu', 'relu_', 'sigmoid', 'sigmoid_', 'tanh', 'tanh_',
        'dropout', 'dropout_', 'gelu', 'silu', 'leaky_relu',
        'add_', 'mul_', 'sub_', 'div_', 'neg', 'abs', 'sqrt',
    }

    def __init__(self):
        super().__init__()
        self._inplace_reuse: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "inplace"

    def reset(self) -> None:
        super().reset()
        self._inplace_reuse = {}

    def _find_offset(
        self,
        size: int,
        birth: int,
        death: int,
        allocated_blocks: List[MemoryBlock],
    ) -> Tuple[int, Optional[str]]:
        """Prioritize in-place reuse, fall back to best-fit"""
        if not allocated_blocks:
            return 0, None

        # Find blocks dying exactly at our birth (potential in-place)
        inplace_candidates = [
            b for b in allocated_blocks
            if b.death_step == birth - 1 and b.size >= size
        ]

        if inplace_candidates:
            # Prefer exact size match for in-place
            exact_match = [b for b in inplace_candidates if b.size == size]
            if exact_match:
                return exact_match[0].offset, exact_match[0].tensor_name
            # Otherwise smallest fitting
            best = min(inplace_candidates, key=lambda b: b.size)
            return best.offset, best.tensor_name

        # Fall back to best-fit
        reusable = [
            b for b in allocated_blocks
            if b.death_step < birth and b.size >= size
        ]

        if reusable:
            best = min(reusable, key=lambda b: b.size)
            return best.offset, best.tensor_name

        # Find gap or allocate at end
        live_blocks = [
            b for b in allocated_blocks
            if not (b.death_step < birth)
        ]

        if not live_blocks:
            return 0, None

        live_blocks.sort(key=lambda b: b.offset)
        max_end = max(b.offset + b.size for b in live_blocks)
        return max_end, None


@StrategyRegistry.register("optimal")
class OptimalStrategy(AllocationStrategy):
    """
    Optimal allocation using weighted scoring function.

    Based on mathematical optimization theory combining:
    1. Size Fit Score - minimize internal fragmentation
    2. Temporal Locality Score - prefer reusing recently freed blocks (cache locality)
    3. Reuse Distribution Score - balance reuse across blocks (reduce hotspots)
    4. Memory Locality Score - prefer lower addresses (spatial locality)

    Score formula:
        score = w1 * size_fit + w2 * temporal + w3 * reuse_balance + w4 * spatial

    Inspired by:
    - Linear Scan Register Allocation (Poletto & Sarkar, 1999)
    - Weighted Interval Scheduling
    - Cache-oblivious algorithms
    """

    def __init__(
        self,
        w_size_fit: float = 0.4,      # Weight for size fitting
        w_temporal: float = 0.3,       # Weight for temporal locality
        w_reuse_balance: float = 0.2,  # Weight for reuse distribution
        w_spatial: float = 0.1,        # Weight for spatial locality
    ):
        super().__init__()
        self.w_size_fit = w_size_fit
        self.w_temporal = w_temporal
        self.w_reuse_balance = w_reuse_balance
        self.w_spatial = w_spatial
        # Track reuse counts per offset
        self._reuse_counts: Dict[int, int] = {}
        self._max_offset = 0

    @property
    def name(self) -> str:
        return "optimal"

    def reset(self) -> None:
        super().reset()
        self._reuse_counts = {}
        self._max_offset = 0

    def _find_offset(
        self,
        size: int,
        birth: int,
        death: int,
        allocated_blocks: List[MemoryBlock],
    ) -> Tuple[int, Optional[str]]:
        """Find optimal offset using weighted scoring function"""
        if not allocated_blocks:
            return 0, None

        # Find all candidate blocks (dead before birth)
        candidates = [
            b for b in allocated_blocks
            if b.death_step < birth and b.size >= size
        ]

        if candidates:
            # Score each candidate
            best_block = None
            best_score = float('-inf')

            # Get max values for normalization
            max_size = max(b.size for b in candidates)
            max_gap = max(birth - b.death_step for b in candidates)
            max_reuse = max(self._reuse_counts.get(b.offset, 0) for b in candidates) + 1

            for block in candidates:
                score = self._compute_score(
                    block, size, birth, max_size, max_gap, max_reuse
                )
                if score > best_score:
                    best_score = score
                    best_block = block

            if best_block:
                # Update reuse count
                self._reuse_counts[best_block.offset] = \
                    self._reuse_counts.get(best_block.offset, 0) + 1
                return best_block.offset, best_block.tensor_name

        # No reusable block - find gap or allocate at end
        return self._find_gap_or_end(size, birth, allocated_blocks)

    def _compute_score(
        self,
        block: MemoryBlock,
        required_size: int,
        birth: int,
        max_size: int,
        max_gap: int,
        max_reuse: int,
    ) -> float:
        """
        Compute weighted score for a candidate block.

        Mathematical formulation:
        - size_fit ∈ [0,1]: 1 - (waste / max_possible_waste)
        - temporal ∈ [0,1]: 1 / (1 + normalized_gap)
        - reuse_balance ∈ [0,1]: 1 - (reuse_count / max_reuse)
        - spatial ∈ [0,1]: 1 - (offset / max_offset)
        """
        # 1. Size Fit Score (minimize internal fragmentation)
        # Perfect fit = 1.0, larger blocks get lower scores
        waste = block.size - required_size
        size_fit = 1.0 - (waste / max_size) if max_size > 0 else 1.0

        # 2. Temporal Locality Score (prefer recently freed blocks)
        # Smaller gap = higher score (better cache locality)
        gap = birth - block.death_step - 1
        temporal = 1.0 / (1.0 + gap / max_gap) if max_gap > 0 else 1.0

        # 3. Reuse Balance Score (distribute reuse evenly)
        # Less reused blocks get higher scores
        reuse_count = self._reuse_counts.get(block.offset, 0)
        reuse_balance = 1.0 - (reuse_count / max_reuse) if max_reuse > 0 else 1.0

        # 4. Spatial Locality Score (prefer lower addresses)
        # Lower offset = higher score
        spatial = 1.0 - (block.offset / self._max_offset) if self._max_offset > 0 else 1.0

        # Weighted sum
        score = (
            self.w_size_fit * size_fit +
            self.w_temporal * temporal +
            self.w_reuse_balance * reuse_balance +
            self.w_spatial * spatial
        )

        return score

    def _find_gap_or_end(
        self,
        size: int,
        birth: int,
        allocated_blocks: List[MemoryBlock],
    ) -> Tuple[int, Optional[str]]:
        """Find best gap or allocate at end"""
        live_blocks = [
            b for b in allocated_blocks
            if b.birth_step <= birth <= b.death_step
        ]

        if not live_blocks:
            return 0, None

        live_blocks.sort(key=lambda b: b.offset)

        # Find all gaps and score them
        gaps = []

        # Gap at beginning
        if live_blocks[0].offset >= size:
            gaps.append((0, live_blocks[0].offset))

        # Gaps between blocks
        for i in range(len(live_blocks) - 1):
            gap_start = live_blocks[i].offset + live_blocks[i].size
            gap_end = live_blocks[i + 1].offset
            gap_size = gap_end - gap_start
            if gap_size >= size:
                gaps.append((gap_start, gap_size))

        if gaps:
            # Score gaps: prefer smaller gaps (less fragmentation) and lower addresses
            def gap_score(gap):
                offset, gap_size = gap
                fit_score = 1.0 - (gap_size - size) / gap_size if gap_size > 0 else 1.0
                pos_score = 1.0 - offset / self._max_offset if self._max_offset > 0 else 1.0
                return 0.7 * fit_score + 0.3 * pos_score

            best_gap = max(gaps, key=gap_score)
            return best_gap[0], None

        # Allocate at end
        max_end = max(b.offset + b.size for b in live_blocks)
        self._max_offset = max(self._max_offset, max_end + size)
        return max_end, None

    def get_reuse_statistics(self) -> Dict[str, any]:
        """Get detailed reuse statistics for analysis"""
        if not self._reuse_counts:
            return {"total_reuses": 0, "unique_offsets": 0}

        counts = list(self._reuse_counts.values())
        return {
            "total_reuses": sum(counts),
            "unique_offsets": len(counts),
            "max_reuse_count": max(counts) if counts else 0,
            "avg_reuse_count": sum(counts) / len(counts) if counts else 0,
            "reuse_distribution": dict(self._reuse_counts),
        }


# Keep "static" as alias for greedy (backward compatibility)
@StrategyRegistry.register("static")
class StaticAllocationStrategy(GreedyReuseStrategy):
    """
    Static allocation - alias for greedy strategy.
    Kept for backward compatibility.
    """

    @property
    def name(self) -> str:
        return "static"
