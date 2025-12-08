# -*- coding: utf-8 -*-
"""Memory analysis for FX GraphModule"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import torch
import torch.fx as fx
from functools import reduce
import operator

from ..core.dtypes import dtype_size, to_torch_dtype


@dataclass
class MemoryConstraint:
    """Memory constraint configuration"""
    max_memory_bytes: Optional[int] = None
    max_memory_mb: Optional[float] = None
    prefer_inplace: bool = True
    alignment: int = 256  # bytes

    def __post_init__(self):
        if self.max_memory_mb and not self.max_memory_bytes:
            self.max_memory_bytes = int(self.max_memory_mb * 1024 * 1024)


@dataclass
class TensorInfo:
    """Information about a tensor in the graph"""
    name: str
    shape: List[int]
    dtype: torch.dtype
    size_bytes: int
    birth_step: int
    death_step: int = -1
    is_input: bool = False
    is_output: bool = False
    is_weight: bool = False
    memory_offset: int = -1
    reused_from: Optional[str] = None
    is_inplace: bool = False


@dataclass
class MemoryBlock:
    """A block in the memory pool"""
    offset: int
    size: int
    is_free: bool = True
    tensor_name: Optional[str] = None


@dataclass
class StepMemoryInfo:
    """Memory information for a single execution step"""
    step: int
    node_name: str
    op_type: str
    target: str

    # Input tensors
    inputs: List[TensorInfo] = field(default_factory=list)
    input_total_bytes: int = 0

    # Output tensor
    output: Optional[TensorInfo] = None
    output_bytes: int = 0

    # Memory statistics
    max_memory: int = 0  # Input + Output (no reuse)
    min_memory: int = 0  # With reuse
    live_tensors: List[str] = field(default_factory=list)
    live_memory: int = 0

    # Allocation info
    freed_tensors: List[str] = field(default_factory=list)
    reused_from: Optional[str] = None


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    # Per-step info
    steps: List[StepMemoryInfo] = field(default_factory=list)

    # Tensor info
    tensors: Dict[str, TensorInfo] = field(default_factory=dict)

    # Summary statistics
    peak_max_memory: int = 0
    peak_min_memory: int = 0
    peak_step_max: int = 0
    peak_step_min: int = 0

    # Static memory (weights)
    static_memory: int = 0
    weight_count: int = 0

    # Strategy used
    strategy_name: str = ""

    # Memory savings
    @property
    def savings_ratio(self) -> float:
        if self.peak_max_memory == 0:
            return 0.0
        return 1.0 - (self.peak_min_memory / self.peak_max_memory)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export"""
        return {
            "summary": {
                "peak_max_memory_mb": self.peak_max_memory / 1e6,
                "peak_min_memory_mb": self.peak_min_memory / 1e6,
                "static_memory_mb": self.static_memory / 1e6,
                "savings_percent": self.savings_ratio * 100,
                "strategy": self.strategy_name,
            },
            "steps": [
                {
                    "step": s.step,
                    "node": s.node_name,
                    "op": s.op_type,
                    "max_memory_mb": s.max_memory / 1e6,
                    "min_memory_mb": s.min_memory / 1e6,
                    "live_tensors": s.live_tensors,
                }
                for s in self.steps
            ],
            "tensors": {
                name: {
                    "shape": t.shape,
                    "dtype": str(t.dtype),
                    "size_mb": t.size_bytes / 1e6,
                    "lifetime": [t.birth_step, t.death_step],
                }
                for name, t in self.tensors.items()
            },
        }


class MemoryAnalyzer:
    """Memory analyzer for FX GraphModule"""

    def __init__(
        self,
        gm: fx.GraphModule,
        strategy: str = "greedy",
        constraint: Optional[MemoryConstraint] = None,
    ):
        self.gm = gm
        self.strategy_name = strategy
        self.constraint = constraint or MemoryConstraint()

        # Import here to avoid circular imports
        from .strategies import StrategyRegistry
        self.strategy = StrategyRegistry.get(strategy)

    def analyze(
        self,
        dynamic_shapes: Optional[Dict[str, int]] = None
    ) -> AnalysisResult:
        """Run complete memory analysis"""
        dynamic_shapes = dynamic_shapes or {}

        # Step 1: Extract tensor info and shapes
        tensors = self._extract_tensors(dynamic_shapes)

        # Step 2: Analyze lifetimes
        self._analyze_lifetimes(tensors)

        # Step 3: Simulate memory allocation
        steps = self._simulate_allocation(tensors)

        # Step 4: Build result
        result = self._build_result(tensors, steps)

        return result

    def _extract_tensors(
        self,
        dynamic_shapes: Dict[str, int]
    ) -> Dict[str, TensorInfo]:
        """Extract tensor information from graph"""
        tensors = {}

        for step, node in enumerate(self.gm.graph.nodes):
            if node.op == "output":
                continue

            # Determine shape and dtype
            shape = self._get_shape(node, dynamic_shapes)
            dtype = self._get_dtype(node)

            if shape is None:
                continue

            # Calculate size
            size_bytes = self._compute_size(shape, dtype)

            # Create tensor info
            tensor = TensorInfo(
                name=node.name,
                shape=shape,
                dtype=dtype,
                size_bytes=size_bytes,
                birth_step=step,
                is_input=(node.op == "placeholder"),
                is_weight=(node.op == "get_attr"),
            )

            tensors[node.name] = tensor

        return tensors

    def _get_shape(
        self,
        node: fx.Node,
        dynamic_shapes: Dict[str, int]
    ) -> Optional[List[int]]:
        """Get tensor shape from node"""
        # Try meta first
        if "shape" in node.meta:
            shape = node.meta["shape"]
            # Replace dynamic dims
            resolved = []
            for dim in shape:
                if isinstance(dim, str):
                    resolved.append(dynamic_shapes.get(dim, 1))
                elif dim is None:
                    resolved.append(dynamic_shapes.get("batch", 1))
                else:
                    resolved.append(dim)
            return resolved

        # For get_attr, get from actual tensor
        if node.op == "get_attr":
            try:
                attrs = node.target.split(".")
                obj = self.gm
                for attr in attrs:
                    obj = getattr(obj, attr)
                if isinstance(obj, torch.Tensor):
                    return list(obj.shape)
            except Exception:
                pass

        return None

    def _get_dtype(self, node: fx.Node) -> torch.dtype:
        """Get tensor dtype from node"""
        if "dtype" in node.meta:
            return to_torch_dtype(node.meta["dtype"])

        if node.op == "get_attr":
            try:
                attrs = node.target.split(".")
                obj = self.gm
                for attr in attrs:
                    obj = getattr(obj, attr)
                if isinstance(obj, torch.Tensor):
                    return obj.dtype
            except Exception:
                pass

        return torch.float32

    def _compute_size(self, shape: List[int], dtype: torch.dtype) -> int:
        """Compute tensor size in bytes"""
        numel = reduce(operator.mul, shape, 1)
        elem_size = dtype_size(dtype)
        size = numel * elem_size

        # Apply alignment
        alignment = self.constraint.alignment
        if alignment > 1:
            size = ((size + alignment - 1) // alignment) * alignment

        return size

    def _analyze_lifetimes(self, tensors: Dict[str, TensorInfo]) -> None:
        """Analyze tensor lifetimes (when each tensor is last used)"""
        # Build consumer map: tensor -> list of consumers
        consumers: Dict[str, List[int]] = {name: [] for name in tensors}

        for step, node in enumerate(self.gm.graph.nodes):
            if node.op == "output":
                # Mark outputs as used until the end
                for arg in node.args:
                    if isinstance(arg, fx.Node) and arg.name in tensors:
                        tensors[arg.name].is_output = True
                        tensors[arg.name].death_step = step
                    elif isinstance(arg, (list, tuple)):
                        for a in arg:
                            if isinstance(a, fx.Node) and a.name in tensors:
                                tensors[a.name].is_output = True
                                tensors[a.name].death_step = step
                continue

            # Record consumers
            for arg in node.args:
                if isinstance(arg, fx.Node) and arg.name in consumers:
                    consumers[arg.name].append(step)
                elif isinstance(arg, (list, tuple)):
                    for a in arg:
                        if isinstance(a, fx.Node) and a.name in consumers:
                            consumers[a.name].append(step)

            for v in node.kwargs.values():
                if isinstance(v, fx.Node) and v.name in consumers:
                    consumers[v.name].append(step)

        # Set death_step to last consumer
        for name, consumer_steps in consumers.items():
            if name not in tensors:
                continue
            tensor = tensors[name]
            if tensor.is_output:
                continue  # Already set
            if consumer_steps:
                tensor.death_step = max(consumer_steps)
            else:
                # No consumers - dies immediately after creation
                tensor.death_step = tensor.birth_step

    def _simulate_allocation(
        self,
        tensors: Dict[str, TensorInfo]
    ) -> List[StepMemoryInfo]:
        """Simulate memory allocation step by step"""
        steps = []
        live_tensors: Dict[str, TensorInfo] = {}

        # Initialize strategy
        self.strategy.reset()

        for step, node in enumerate(self.gm.graph.nodes):
            if node.op == "output":
                continue

            step_info = StepMemoryInfo(
                step=step,
                node_name=node.name,
                op_type=node.op,
                target=str(node.target) if node.target else "",
            )

            # Get input tensors
            input_names = self._get_input_names(node)
            for name in input_names:
                if name in tensors:
                    step_info.inputs.append(tensors[name])
                    step_info.input_total_bytes += tensors[name].size_bytes

            # Find tensors that die at this step
            dying = [
                name for name, t in live_tensors.items()
                if t.death_step == step and not t.is_weight
            ]

            # Free dying tensors
            for name in dying:
                self.strategy.deallocate(name)
                step_info.freed_tensors.append(name)
                del live_tensors[name]

            # Get output tensor
            if node.name in tensors:
                output_tensor = tensors[node.name]
                step_info.output = output_tensor
                step_info.output_bytes = output_tensor.size_bytes

                # Allocate memory for output
                if not output_tensor.is_weight:
                    result = self.strategy.allocate(
                        output_tensor,
                        live_tensors,
                        self.constraint
                    )
                    output_tensor.memory_offset = result.offset
                    output_tensor.reused_from = result.reused_from
                    output_tensor.is_inplace = result.is_inplace
                    step_info.reused_from = result.reused_from

                    live_tensors[node.name] = output_tensor

            # Calculate memory stats
            step_info.max_memory = step_info.input_total_bytes + step_info.output_bytes
            step_info.live_tensors = list(live_tensors.keys())
            step_info.live_memory = sum(t.size_bytes for t in live_tensors.values())
            step_info.min_memory = self.strategy.current_memory()

            steps.append(step_info)

        return steps

    def _get_input_names(self, node: fx.Node) -> List[str]:
        """Get names of input tensors for a node"""
        names = []
        for arg in node.args:
            if isinstance(arg, fx.Node):
                names.append(arg.name)
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    if isinstance(a, fx.Node):
                        names.append(a.name)
        for v in node.kwargs.values():
            if isinstance(v, fx.Node):
                names.append(v.name)
        return names

    def _build_result(
        self,
        tensors: Dict[str, TensorInfo],
        steps: List[StepMemoryInfo]
    ) -> AnalysisResult:
        """Build final analysis result"""
        result = AnalysisResult(
            steps=steps,
            tensors=tensors,
            strategy_name=self.strategy_name,
        )

        # Calculate peaks
        if steps:
            max_memories = [s.max_memory for s in steps]
            min_memories = [s.min_memory for s in steps]

            result.peak_max_memory = max(max_memories)
            result.peak_min_memory = max(min_memories)
            result.peak_step_max = max_memories.index(result.peak_max_memory)
            result.peak_step_min = min_memories.index(result.peak_min_memory)

        # Calculate static memory (weights)
        for tensor in tensors.values():
            if tensor.is_weight:
                result.static_memory += tensor.size_bytes
                result.weight_count += 1

        return result

    def compare_strategies(
        self,
        strategies: Optional[List[str]] = None,
        dynamic_shapes: Optional[Dict[str, int]] = None
    ) -> Dict[str, AnalysisResult]:
        """Compare multiple allocation strategies"""
        from .strategies import StrategyRegistry

        strategies = strategies or StrategyRegistry.list_strategies()
        results = {}

        for strategy_name in strategies:
            analyzer = MemoryAnalyzer(
                self.gm,
                strategy=strategy_name,
                constraint=self.constraint
            )
            results[strategy_name] = analyzer.analyze(dynamic_shapes)

        return results
