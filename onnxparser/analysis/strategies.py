# -*- coding: utf-8 -*-
"""Memory allocation strategies"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type
import torch


@dataclass
class AllocationResult:
    """Result of memory allocation"""
    tensor_name: str
    offset: int
    size: int
    reused_from: Optional[str] = None
    is_inplace: bool = False


class AllocationStrategy(ABC):
    """Base class for memory allocation strategies"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset strategy state"""
        pass

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
    """No memory reuse - calculates maximum memory requirement"""

    def __init__(self):
        self._next_offset = 0
        self._allocations: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "no_reuse"

    def reset(self) -> None:
        self._next_offset = 0
        self._allocations = {}

    def allocate(self, tensor, live_tensors, constraint) -> AllocationResult:
        size = self._align(tensor.size_bytes, constraint.alignment)
        offset = self._next_offset
        self._next_offset += size
        self._allocations[tensor.name] = size

        return AllocationResult(
            tensor_name=tensor.name,
            offset=offset,
            size=size,
            reused_from=None,
        )

    def deallocate(self, tensor_name: str) -> None:
        # No-op: we don't reuse memory
        pass

    def current_memory(self) -> int:
        return self._next_offset

    def _align(self, size: int, alignment: int) -> int:
        if alignment <= 1:
            return size
        return ((size + alignment - 1) // alignment) * alignment


@StrategyRegistry.register("greedy")
class GreedyReuseStrategy(AllocationStrategy):
    """Greedy memory reuse - Best-fit strategy"""

    def __init__(self):
        self._memory_pool: List[Dict] = []  # [{offset, size, free, tensor}]
        self._peak_memory = 0
        self._current_usage = 0

    @property
    def name(self) -> str:
        return "greedy"

    def reset(self) -> None:
        self._memory_pool = []
        self._peak_memory = 0
        self._current_usage = 0

    def allocate(self, tensor, live_tensors, constraint) -> AllocationResult:
        required = self._align(tensor.size_bytes, constraint.alignment)

        # Find best-fit free block
        best_block = None
        best_idx = -1
        for idx, block in enumerate(self._memory_pool):
            if block["free"] and block["size"] >= required:
                if best_block is None or block["size"] < best_block["size"]:
                    best_block = block
                    best_idx = idx

        if best_block:
            # Reuse existing block
            best_block["free"] = False
            reused_from = best_block["tensor"]
            best_block["tensor"] = tensor.name
            self._current_usage += required

            return AllocationResult(
                tensor_name=tensor.name,
                offset=best_block["offset"],
                size=required,
                reused_from=reused_from,
            )
        else:
            # Allocate new block
            offset = sum(b["size"] for b in self._memory_pool)
            self._memory_pool.append({
                "offset": offset,
                "size": required,
                "free": False,
                "tensor": tensor.name,
            })
            self._current_usage += required
            self._peak_memory = max(self._peak_memory, self._current_usage)

            return AllocationResult(
                tensor_name=tensor.name,
                offset=offset,
                size=required,
            )

    def deallocate(self, tensor_name: str) -> None:
        for block in self._memory_pool:
            if block["tensor"] == tensor_name and not block["free"]:
                block["free"] = True
                self._current_usage -= block["size"]
                break

    def current_memory(self) -> int:
        # Return total allocated (not freed) memory
        return sum(b["size"] for b in self._memory_pool if not b["free"])

    def _align(self, size: int, alignment: int) -> int:
        if alignment <= 1:
            return size
        return ((size + alignment - 1) // alignment) * alignment


@StrategyRegistry.register("inplace")
class InplaceStrategy(AllocationStrategy):
    """Prefer in-place operations when possible"""

    INPLACE_OPS = {
        'relu', 'relu_', 'sigmoid', 'sigmoid_', 'tanh', 'tanh_',
        'dropout', 'dropout_', 'gelu', 'silu', 'leaky_relu',
        'add_', 'mul_', 'sub_', 'div_',
    }

    def __init__(self):
        self._greedy = GreedyReuseStrategy()
        self._allocations: Dict[str, Dict] = {}

    @property
    def name(self) -> str:
        return "inplace"

    def reset(self) -> None:
        self._greedy.reset()
        self._allocations = {}

    def allocate(self, tensor, live_tensors, constraint) -> AllocationResult:
        # Check if this can be an in-place operation
        if self._can_inplace(tensor, live_tensors):
            # Find the input tensor that can be overwritten
            for input_tensor in live_tensors.values():
                if self._is_compatible(tensor, input_tensor):
                    # Reuse input's memory (in-place)
                    self._allocations[tensor.name] = {
                        "offset": input_tensor.memory_offset,
                        "size": tensor.size_bytes,
                        "inplace_from": input_tensor.name,
                    }
                    return AllocationResult(
                        tensor_name=tensor.name,
                        offset=input_tensor.memory_offset,
                        size=tensor.size_bytes,
                        reused_from=input_tensor.name,
                        is_inplace=True,
                    )

        # Fall back to greedy strategy
        result = self._greedy.allocate(tensor, live_tensors, constraint)
        self._allocations[tensor.name] = {
            "offset": result.offset,
            "size": result.size,
        }
        return result

    def deallocate(self, tensor_name: str) -> None:
        if tensor_name in self._allocations:
            alloc = self._allocations[tensor_name]
            if "inplace_from" not in alloc:
                self._greedy.deallocate(tensor_name)
            del self._allocations[tensor_name]

    def current_memory(self) -> int:
        return self._greedy.current_memory()

    def _can_inplace(self, tensor, live_tensors) -> bool:
        """Check if tensor operation can be done in-place"""
        # Check if operation name suggests in-place capability
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
        return True

    def _align(self, size: int, alignment: int) -> int:
        if alignment <= 1:
            return size
        return ((size + alignment - 1) // alignment) * alignment


@StrategyRegistry.register("memory_bounded")
class MemoryBoundedStrategy(AllocationStrategy):
    """Memory-bounded allocation with configurable limit"""

    def __init__(self):
        self._greedy = GreedyReuseStrategy()
        self._memory_limit: Optional[int] = None

    @property
    def name(self) -> str:
        return "memory_bounded"

    def reset(self) -> None:
        self._greedy.reset()

    def allocate(self, tensor, live_tensors, constraint) -> AllocationResult:
        self._memory_limit = constraint.max_memory_bytes

        # Check if we would exceed the limit
        if self._memory_limit:
            current = self.current_memory()
            required = tensor.size_bytes
            if current + required > self._memory_limit:
                # Try to find a block to reuse more aggressively
                pass  # For now, just proceed with greedy

        return self._greedy.allocate(tensor, live_tensors, constraint)

    def deallocate(self, tensor_name: str) -> None:
        self._greedy.deallocate(tensor_name)

    def current_memory(self) -> int:
        return self._greedy.current_memory()
