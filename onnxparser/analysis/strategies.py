# -*- coding: utf-8 -*-
"""Memory allocation strategies with delay-based reuse control"""

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_analyzer import TensorInfo, MemoryConstraint


@dataclass
class ReuseConfig:
    """Configuration for memory reuse delays

    delay = N means the tensor can be reused N steps after its death.
    delay = 0 means immediate reuse (as soon as tensor dies)
    delay = 99999 (or any large number) means never reuse
    """
    activation_delay: int = 0      # Activation can be reused immediately after death
    input_delay: int = 2           # Input can be reused 2 steps after death
    output_delay: int = 2          # Output can be reused 2 steps after death
    weight_delay: int = 99999      # Weight never reused (read-only, kept in memory)

    @classmethod
    def no_reuse(cls) -> 'ReuseConfig':
        """No reuse at all - every tensor gets independent allocation"""
        return cls(
            activation_delay=99999,
            input_delay=99999,
            output_delay=99999,
            weight_delay=99999,
        )

    @classmethod
    def aggressive(cls) -> 'ReuseConfig':
        """Aggressive reuse - minimize memory usage"""
        return cls(
            activation_delay=0,
            input_delay=0,
            output_delay=0,
            weight_delay=99999,
        )


@dataclass
class AllocationResult:
    """Result of memory allocation"""
    tensor_name: str
    offset: int
    size: int
    reused_from: Optional[str] = None
    exceeded_limit: bool = False


@dataclass
class MemoryBlock:
    """A block in the memory pool"""
    offset: int
    size: int
    tensor_name: str
    death_step: int           # When the tensor dies
    reuse_delay: int          # Steps after death before reuse allowed
    is_free: bool = False     # Whether the block is available for reuse

    def can_reuse_at(self, current_step: int, required_size: int) -> bool:
        """Check if this block can be reused at the given step"""
        if not self.is_free:
            return False
        if self.size < required_size:
            return False
        # Check if delay period has passed
        reuse_available_step = self.death_step + self.reuse_delay
        return current_step >= reuse_available_step


class DelayBasedStrategy:
    """
    Memory allocation strategy with delay-based reuse control.

    Each tensor type has a configurable delay before its memory can be reused:
    - activation_delay: Steps after death before activation memory can be reused
    - input_delay: Steps after death before input memory can be reused
    - output_delay: Steps after death before output memory can be reused
    - weight_delay: Set to large number (99999) to prevent reuse
    """

    def __init__(self, config: Optional[ReuseConfig] = None):
        self.config = config or ReuseConfig()
        self._memory_pool: List[MemoryBlock] = []
        self._total_allocated = 0
        self._current_step = 0
        self._memory_limit: Optional[int] = None
        self._exceeded_count = 0
        self._total_exceeded_bytes = 0

    @property
    def name(self) -> str:
        return "delay_based"

    def reset(self) -> None:
        """Reset strategy state"""
        self._memory_pool = []
        self._total_allocated = 0
        self._current_step = 0
        self._memory_limit = None
        self._exceeded_count = 0
        self._total_exceeded_bytes = 0

    def set_step(self, step: int) -> None:
        """Set current execution step"""
        self._current_step = step

    def _get_reuse_delay(self, tensor: 'TensorInfo') -> int:
        """Get reuse delay for a tensor based on its type"""
        if tensor.is_weight:
            return self.config.weight_delay
        elif tensor.is_input:
            return self.config.input_delay
        elif tensor.is_output:
            return self.config.output_delay
        else:
            return self.config.activation_delay

    def _align(self, size: int, alignment: int) -> int:
        """Align size to specified boundary"""
        if alignment <= 1:
            return size
        return ((size + alignment - 1) // alignment) * alignment

    def allocate(
        self,
        tensor: 'TensorInfo',
        live_tensors: Dict[str, 'TensorInfo'],
        constraint: 'MemoryConstraint'
    ) -> AllocationResult:
        """Allocate memory for a tensor"""
        self._memory_limit = constraint.max_memory_bytes
        required = self._align(tensor.size_bytes, constraint.alignment)
        reuse_delay = self._get_reuse_delay(tensor)

        # Check memory limit
        exceeded = False
        if self._memory_limit is not None:
            current = self.current_memory()
            if current + required > self._memory_limit:
                exceeded = True
                self._exceeded_count += 1
                self._total_exceeded_bytes += required

        # Try to find a reusable block (best-fit)
        best_block = None
        for block in self._memory_pool:
            if block.can_reuse_at(self._current_step, required):
                if best_block is None or block.size < best_block.size:
                    best_block = block

        if best_block is not None:
            # Reuse existing block
            reused_from = best_block.tensor_name
            best_block.tensor_name = tensor.name
            best_block.death_step = tensor.death_step
            best_block.reuse_delay = reuse_delay
            best_block.is_free = False

            return AllocationResult(
                tensor_name=tensor.name,
                offset=best_block.offset,
                size=required,
                reused_from=reused_from,
                exceeded_limit=False,
            )

        # No reusable block found - allocate new
        offset = self._total_allocated
        self._memory_pool.append(MemoryBlock(
            offset=offset,
            size=required,
            tensor_name=tensor.name,
            death_step=tensor.death_step,
            reuse_delay=reuse_delay,
            is_free=False,
        ))
        self._total_allocated += required

        return AllocationResult(
            tensor_name=tensor.name,
            offset=offset,
            size=required,
            reused_from=None,
            exceeded_limit=exceeded,
        )

    def deallocate(self, tensor_name: str) -> None:
        """Mark a tensor's memory as potentially reusable"""
        for block in self._memory_pool:
            if block.tensor_name == tensor_name and not block.is_free:
                block.is_free = True
                break

    def current_memory(self) -> int:
        """Get current active memory usage"""
        return sum(b.size for b in self._memory_pool if not b.is_free)

    @property
    def exceeded_stats(self) -> Dict:
        """Get statistics about memory limit violations"""
        return {
            "exceeded_count": self._exceeded_count,
            "total_exceeded_bytes": self._total_exceeded_bytes,
        }


class StrategyRegistry:
    """Registry for allocation strategies"""

    _configs: Dict[str, ReuseConfig] = {
        "no_reuse": ReuseConfig.no_reuse(),
        "greedy": ReuseConfig.aggressive(),
        "default": ReuseConfig(),  # activation=0, input=2, output=2, weight=99999
    }

    @classmethod
    def get(cls, name: str) -> DelayBasedStrategy:
        """Get a strategy instance by name"""
        if name in cls._configs:
            return DelayBasedStrategy(cls._configs[name])
        # Try to parse custom config like "delay_0_2_2_99999"
        if name.startswith("delay_"):
            parts = name.split("_")[1:]
            if len(parts) == 4:
                config = ReuseConfig(
                    activation_delay=int(parts[0]),
                    input_delay=int(parts[1]),
                    output_delay=int(parts[2]),
                    weight_delay=int(parts[3]),
                )
                return DelayBasedStrategy(config)
        raise ValueError(f"Unknown strategy '{name}'. Available: {cls.list_strategies()}")

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all available strategies"""
        return list(cls._configs.keys())

    @classmethod
    def register(cls, name: str, config: ReuseConfig) -> None:
        """Register a custom strategy configuration"""
        cls._configs[name] = config
