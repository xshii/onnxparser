# -*- coding: utf-8 -*-
"""Memory spill/reload scheduler for memory-constrained execution"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import torch.fx as fx

from .memory_analyzer import MemoryAnalyzer, MemoryConstraint, TensorInfo, AnalysisResult


class MemoryEventType(Enum):
    """Types of memory events"""
    ALLOCATE = "allocate"      # Allocate new tensor in fast memory
    DEALLOCATE = "deallocate"  # Free tensor from fast memory
    SPILL = "spill"            # Move tensor from fast to slow memory
    RELOAD = "reload"          # Move tensor from slow to fast memory
    COMPUTE = "compute"        # Execute computation


@dataclass
class MemoryEvent:
    """A memory management event"""
    step: int
    event_type: MemoryEventType
    tensor_name: str
    size_bytes: int
    node_name: str = ""

    # For spill/reload events
    target_memory: str = ""  # "fast" or "slow"

    # Memory state after this event
    fast_memory_used: int = 0
    slow_memory_used: int = 0

    def __repr__(self):
        return f"MemoryEvent(step={self.step}, {self.event_type.value}, {self.tensor_name}, {self.size_bytes/1024:.1f}KB)"


@dataclass
class SpillDecision:
    """Decision about which tensor to spill"""
    tensor_name: str
    size_bytes: int
    spill_step: int
    reload_step: int
    priority: float  # Higher = more likely to spill

    @property
    def spill_duration(self) -> int:
        return self.reload_step - self.spill_step


@dataclass
class ScheduleResult:
    """Result of spill/reload scheduling"""
    # All memory events in execution order
    events: List[MemoryEvent] = field(default_factory=list)

    # Spill decisions
    spill_decisions: List[SpillDecision] = field(default_factory=list)

    # Per-step memory usage
    fast_memory_timeline: List[int] = field(default_factory=list)
    slow_memory_timeline: List[int] = field(default_factory=list)

    # Summary
    total_spills: int = 0
    total_reloads: int = 0
    total_spill_bytes: int = 0
    peak_fast_memory: int = 0
    peak_slow_memory: int = 0

    # Nodes that trigger memory operations
    spill_trigger_nodes: List[str] = field(default_factory=list)
    reload_trigger_nodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_spills": self.total_spills,
                "total_reloads": self.total_reloads,
                "total_spill_mb": self.total_spill_bytes / 1e6,
                "peak_fast_memory_mb": self.peak_fast_memory / 1e6,
                "peak_slow_memory_mb": self.peak_slow_memory / 1e6,
            },
            "spill_trigger_nodes": self.spill_trigger_nodes,
            "reload_trigger_nodes": self.reload_trigger_nodes,
            "spill_decisions": [
                {
                    "tensor": d.tensor_name,
                    "size_mb": d.size_bytes / 1e6,
                    "spill_step": d.spill_step,
                    "reload_step": d.reload_step,
                    "duration": d.spill_duration,
                }
                for d in self.spill_decisions
            ],
            "events": [
                {
                    "step": e.step,
                    "type": e.event_type.value,
                    "tensor": e.tensor_name,
                    "size_mb": e.size_bytes / 1e6,
                    "node": e.node_name,
                    "fast_mb": e.fast_memory_used / 1e6,
                    "slow_mb": e.slow_memory_used / 1e6,
                }
                for e in self.events
            ],
        }


class SpillStrategy(Enum):
    """Strategies for selecting tensors to spill"""
    LRU = "lru"                    # Least recently used
    SIZE_FIRST = "size_first"      # Spill largest tensors first
    LIFETIME = "lifetime"          # Consider remaining lifetime
    COST_BENEFIT = "cost_benefit"  # Balance size vs reload cost


class SpillScheduler:
    """
    Scheduler that determines when to spill/reload tensors
    to fit computation within memory constraints.
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        memory_limit_bytes: int,
        spill_strategy: SpillStrategy = SpillStrategy.COST_BENEFIT,
        slow_memory_limit: Optional[int] = None,
    ):
        self.gm = gm
        self.memory_limit = memory_limit_bytes
        self.spill_strategy = spill_strategy
        self.slow_memory_limit = slow_memory_limit

        # Run base memory analysis first
        analyzer = MemoryAnalyzer(gm, strategy="greedy")
        self.base_analysis = analyzer.analyze()

    def schedule(self) -> ScheduleResult:
        """Generate spill/reload schedule"""
        result = ScheduleResult()

        # Track memory state
        fast_memory: Dict[str, TensorInfo] = {}  # tensors in fast memory
        slow_memory: Dict[str, TensorInfo] = {}  # tensors spilled to slow memory
        fast_memory_used = 0
        slow_memory_used = 0

        # Track tensor usage for scheduling
        tensor_last_use: Dict[str, int] = {}
        tensor_next_use: Dict[str, int] = {}

        # Build usage maps
        self._build_usage_maps(tensor_last_use, tensor_next_use)

        # Process each step
        for step_info in self.base_analysis.steps:
            step = step_info.step
            node_name = step_info.node_name

            # Check if any spilled tensor needs to be reloaded for this step
            tensors_needed = self._get_input_tensors(step_info)
            for tensor_name in tensors_needed:
                if tensor_name in slow_memory:
                    # Need to reload this tensor
                    tensor = slow_memory[tensor_name]

                    # Check if we have space, if not spill something
                    while fast_memory_used + tensor.size_bytes > self.memory_limit:
                        spilled = self._select_tensor_to_spill(
                            fast_memory, step, tensor_next_use
                        )
                        if spilled is None:
                            break  # Can't spill anything more

                        # Record spill event
                        fast_memory_used -= spilled.size_bytes
                        slow_memory_used += spilled.size_bytes
                        slow_memory[spilled.name] = spilled
                        del fast_memory[spilled.name]

                        result.events.append(MemoryEvent(
                            step=step,
                            event_type=MemoryEventType.SPILL,
                            tensor_name=spilled.name,
                            size_bytes=spilled.size_bytes,
                            node_name=node_name,
                            target_memory="slow",
                            fast_memory_used=fast_memory_used,
                            slow_memory_used=slow_memory_used,
                        ))
                        result.total_spills += 1
                        result.total_spill_bytes += spilled.size_bytes
                        if node_name not in result.spill_trigger_nodes:
                            result.spill_trigger_nodes.append(node_name)

                    # Reload the tensor
                    fast_memory_used += tensor.size_bytes
                    slow_memory_used -= tensor.size_bytes
                    fast_memory[tensor_name] = tensor
                    del slow_memory[tensor_name]

                    result.events.append(MemoryEvent(
                        step=step,
                        event_type=MemoryEventType.RELOAD,
                        tensor_name=tensor_name,
                        size_bytes=tensor.size_bytes,
                        node_name=node_name,
                        target_memory="fast",
                        fast_memory_used=fast_memory_used,
                        slow_memory_used=slow_memory_used,
                    ))
                    result.total_reloads += 1
                    if node_name not in result.reload_trigger_nodes:
                        result.reload_trigger_nodes.append(node_name)

            # Allocate output tensor
            if step_info.output and not step_info.output.is_weight:
                output_tensor = step_info.output
                needed_space = output_tensor.size_bytes

                # Spill tensors if needed to make room
                while fast_memory_used + needed_space > self.memory_limit:
                    spilled = self._select_tensor_to_spill(
                        fast_memory, step, tensor_next_use
                    )
                    if spilled is None:
                        break

                    # Record spill
                    fast_memory_used -= spilled.size_bytes
                    slow_memory_used += spilled.size_bytes
                    slow_memory[spilled.name] = spilled
                    del fast_memory[spilled.name]

                    # Record spill decision
                    reload_step = tensor_next_use.get(spilled.name, step + 1)
                    result.spill_decisions.append(SpillDecision(
                        tensor_name=spilled.name,
                        size_bytes=spilled.size_bytes,
                        spill_step=step,
                        reload_step=reload_step,
                        priority=self._compute_spill_priority(
                            spilled, step, tensor_next_use
                        ),
                    ))

                    result.events.append(MemoryEvent(
                        step=step,
                        event_type=MemoryEventType.SPILL,
                        tensor_name=spilled.name,
                        size_bytes=spilled.size_bytes,
                        node_name=node_name,
                        target_memory="slow",
                        fast_memory_used=fast_memory_used,
                        slow_memory_used=slow_memory_used,
                    ))
                    result.total_spills += 1
                    result.total_spill_bytes += spilled.size_bytes
                    if node_name not in result.spill_trigger_nodes:
                        result.spill_trigger_nodes.append(node_name)

                # Allocate output
                fast_memory_used += needed_space
                fast_memory[output_tensor.name] = output_tensor

                result.events.append(MemoryEvent(
                    step=step,
                    event_type=MemoryEventType.ALLOCATE,
                    tensor_name=output_tensor.name,
                    size_bytes=output_tensor.size_bytes,
                    node_name=node_name,
                    fast_memory_used=fast_memory_used,
                    slow_memory_used=slow_memory_used,
                ))

            # Record compute event
            result.events.append(MemoryEvent(
                step=step,
                event_type=MemoryEventType.COMPUTE,
                tensor_name=node_name,
                size_bytes=0,
                node_name=node_name,
                fast_memory_used=fast_memory_used,
                slow_memory_used=slow_memory_used,
            ))

            # Free tensors that are no longer needed
            tensors_to_free = [
                name for name, tensor in fast_memory.items()
                if tensor_last_use.get(name, -1) == step and not tensor.is_output
            ]
            for tensor_name in tensors_to_free:
                tensor = fast_memory[tensor_name]
                fast_memory_used -= tensor.size_bytes
                del fast_memory[tensor_name]

                result.events.append(MemoryEvent(
                    step=step,
                    event_type=MemoryEventType.DEALLOCATE,
                    tensor_name=tensor_name,
                    size_bytes=tensor.size_bytes,
                    node_name=node_name,
                    fast_memory_used=fast_memory_used,
                    slow_memory_used=slow_memory_used,
                ))

            # Also free from slow memory if no longer needed
            tensors_to_free_slow = [
                name for name, tensor in slow_memory.items()
                if tensor_last_use.get(name, -1) <= step
            ]
            for tensor_name in tensors_to_free_slow:
                tensor = slow_memory[tensor_name]
                slow_memory_used -= tensor.size_bytes
                del slow_memory[tensor_name]

            # Track timeline
            result.fast_memory_timeline.append(fast_memory_used)
            result.slow_memory_timeline.append(slow_memory_used)
            result.peak_fast_memory = max(result.peak_fast_memory, fast_memory_used)
            result.peak_slow_memory = max(result.peak_slow_memory, slow_memory_used)

        return result

    def _build_usage_maps(
        self,
        tensor_last_use: Dict[str, int],
        tensor_next_use: Dict[str, int],
    ) -> None:
        """Build maps of tensor usage patterns"""
        # Build from base analysis
        for tensor_name, tensor in self.base_analysis.tensors.items():
            tensor_last_use[tensor_name] = tensor.death_step

        # Build next-use map (for each step, when is each live tensor next used)
        # This is used for spill priority calculation
        for step_info in self.base_analysis.steps:
            step = step_info.step
            for tensor_name in step_info.live_tensors:
                if tensor_name not in tensor_next_use:
                    # Find next use after current step
                    for future_step in self.base_analysis.steps[step:]:
                        inputs = self._get_input_tensors(future_step)
                        if tensor_name in inputs:
                            tensor_next_use[tensor_name] = future_step.step
                            break

    def _get_input_tensors(self, step_info) -> Set[str]:
        """Get names of input tensors for a step"""
        return {t.name for t in step_info.inputs if not t.is_weight}

    def _select_tensor_to_spill(
        self,
        fast_memory: Dict[str, TensorInfo],
        current_step: int,
        tensor_next_use: Dict[str, int],
    ) -> Optional[TensorInfo]:
        """Select which tensor to spill based on strategy"""
        candidates = [
            t for t in fast_memory.values()
            if not t.is_input and not t.is_output and not t.is_weight
        ]

        if not candidates:
            return None

        if self.spill_strategy == SpillStrategy.LRU:
            # Spill the tensor that was used longest ago
            # (approximated by birth_step for simplicity)
            return min(candidates, key=lambda t: t.birth_step)

        elif self.spill_strategy == SpillStrategy.SIZE_FIRST:
            # Spill largest tensor
            return max(candidates, key=lambda t: t.size_bytes)

        elif self.spill_strategy == SpillStrategy.LIFETIME:
            # Spill tensor with longest remaining lifetime
            def remaining_life(t):
                return t.death_step - current_step
            return max(candidates, key=remaining_life)

        elif self.spill_strategy == SpillStrategy.COST_BENEFIT:
            # Balance: spill large tensors that won't be needed soon
            def score(t):
                next_use = tensor_next_use.get(t.name, current_step + 1)
                time_until_needed = next_use - current_step
                # Higher score = better candidate for spilling
                return t.size_bytes * time_until_needed
            return max(candidates, key=score)

        return candidates[0] if candidates else None

    def _compute_spill_priority(
        self,
        tensor: TensorInfo,
        current_step: int,
        tensor_next_use: Dict[str, int],
    ) -> float:
        """Compute priority score for spilling a tensor"""
        next_use = tensor_next_use.get(tensor.name, current_step + 1)
        time_until_needed = max(1, next_use - current_step)
        return tensor.size_bytes * time_until_needed


def schedule_spills(
    gm: fx.GraphModule,
    memory_limit_mb: float,
    strategy: str = "cost_benefit",
) -> ScheduleResult:
    """Convenience function to schedule spills for a graph"""
    strategy_map = {
        "lru": SpillStrategy.LRU,
        "size_first": SpillStrategy.SIZE_FIRST,
        "lifetime": SpillStrategy.LIFETIME,
        "cost_benefit": SpillStrategy.COST_BENEFIT,
    }

    scheduler = SpillScheduler(
        gm,
        memory_limit_bytes=int(memory_limit_mb * 1024 * 1024),
        spill_strategy=strategy_map.get(strategy, SpillStrategy.COST_BENEFIT),
    )
    return scheduler.schedule()
