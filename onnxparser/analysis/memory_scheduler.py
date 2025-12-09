# -*- coding: utf-8 -*-
"""Memory scheduler with transfer node generation for on-chip/off-chip memory management"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
from enum import Enum

import torch.fx as fx

if TYPE_CHECKING:
    from .memory_analyzer import TensorInfo


class TransferType(Enum):
    """Memory transfer type"""
    LOAD = "load"       # DDR -> SRAM (搬入)
    STORE = "store"     # SRAM -> DDR (搬出)
    PREFETCH = "prefetch"  # Async load (预取)


class MemoryLocation(Enum):
    """Memory location"""
    ON_CHIP = "on_chip"    # SRAM
    OFF_CHIP = "off_chip"  # DDR


@dataclass
class MemoryTransferConfig:
    """Memory transfer configuration"""
    # Memory hierarchy sizes
    on_chip_size_bytes: int              # On-chip memory size (SRAM)
    off_chip_size_bytes: Optional[int] = None  # Off-chip memory size (DDR)

    # Bandwidth for latency estimation (GB/s)
    load_bandwidth_gbps: float = 100.0   # Load bandwidth
    store_bandwidth_gbps: float = 50.0   # Store bandwidth

    # Mode switches
    enable_store: bool = True            # Generate explicit Store nodes
    enable_prefetch: bool = False        # Enable prefetching
    trace_mode: bool = False             # Trace mode: generate Store even for overwrites

    # Timing
    transfer_overhead_us: float = 0.1    # Per-transfer overhead (microseconds)

    # Alignment
    alignment: int = 256


@dataclass
class TransferNode:
    """A memory transfer node"""
    name: str
    transfer_type: TransferType
    tensor_name: str                     # Tensor being transferred
    size_bytes: int

    # Address info
    src_offset: int = 0                  # Source address offset
    dst_offset: int = 0                  # Destination address offset
    src_memory: MemoryLocation = MemoryLocation.OFF_CHIP
    dst_memory: MemoryLocation = MemoryLocation.ON_CHIP

    # Timing
    estimated_latency_us: float = 0.0
    trigger_step: float = 0.0            # When to trigger (fractional step)

    # Insertion info
    insert_before_step: int = 0          # Insert before this execution step
    insert_after_step: int = -1          # Or insert after this step

    # Optimization info
    is_overwrite: bool = False           # True if this overwrites dying tensor
    overwrites_tensor: Optional[str] = None  # Name of tensor being overwritten

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "type": self.transfer_type.value,
            "tensor": self.tensor_name,
            "size_bytes": self.size_bytes,
            "size_kb": self.size_bytes / 1024,
            "src_offset": self.src_offset,
            "dst_offset": self.dst_offset,
            "src_memory": self.src_memory.value,
            "dst_memory": self.dst_memory.value,
            "latency_us": self.estimated_latency_us,
            "trigger_step": self.trigger_step,
            "insert_before": self.insert_before_step,
            "is_overwrite": self.is_overwrite,
            "overwrites": self.overwrites_tensor,
        }


@dataclass
class TensorLocation:
    """Track tensor location in memory hierarchy"""
    name: str
    size_bytes: int
    location: MemoryLocation
    on_chip_offset: int = -1
    off_chip_offset: int = -1
    birth_step: int = 0
    death_step: int = -1
    last_access_step: int = 0


@dataclass
class MemorySchedule:
    """Complete memory schedule with transfer nodes"""
    transfers: List[TransferNode] = field(default_factory=list)
    tensor_locations: Dict[str, TensorLocation] = field(default_factory=dict)

    # Statistics
    peak_on_chip_bytes: int = 0
    total_load_bytes: int = 0
    total_store_bytes: int = 0
    total_load_latency_us: float = 0.0
    total_store_latency_us: float = 0.0
    num_evictions: int = 0
    num_overwrites: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "transfers": [t.to_dict() for t in self.transfers],
            "summary": {
                "total_transfers": len(self.transfers),
                "num_loads": sum(1 for t in self.transfers if t.transfer_type == TransferType.LOAD),
                "num_stores": sum(1 for t in self.transfers if t.transfer_type == TransferType.STORE),
                "num_prefetches": sum(1 for t in self.transfers if t.transfer_type == TransferType.PREFETCH),
                "peak_on_chip_kb": self.peak_on_chip_bytes / 1024,
                "total_load_kb": self.total_load_bytes / 1024,
                "total_store_kb": self.total_store_bytes / 1024,
                "total_load_latency_us": self.total_load_latency_us,
                "total_store_latency_us": self.total_store_latency_us,
                "total_latency_us": self.total_load_latency_us + self.total_store_latency_us,
                "num_evictions": self.num_evictions,
                "num_overwrites": self.num_overwrites,
            }
        }


class EvictionPolicy:
    """Policies for selecting tensors to evict from on-chip memory"""

    @staticmethod
    def furthest_next_use(
        on_chip_tensors: Dict[str, TensorLocation],
        current_step: int,
        next_use: Dict[str, int],
    ) -> Optional[str]:
        """
        Belady's optimal: evict tensor whose next use is furthest away.
        This is optimal for minimizing cache misses.
        """
        best_victim = None
        furthest_use = -1

        for name, loc in on_chip_tensors.items():
            next_step = next_use.get(name, float('inf'))
            if next_step > furthest_use:
                furthest_use = next_step
                best_victim = name

        return best_victim

    @staticmethod
    def largest_first(
        on_chip_tensors: Dict[str, TensorLocation],
        current_step: int,
    ) -> Optional[str]:
        """Evict largest tensor to free most space"""
        if not on_chip_tensors:
            return None
        return max(on_chip_tensors.keys(), key=lambda n: on_chip_tensors[n].size_bytes)

    @staticmethod
    def lru(
        on_chip_tensors: Dict[str, TensorLocation],
        current_step: int,
    ) -> Optional[str]:
        """Least Recently Used"""
        if not on_chip_tensors:
            return None
        return min(on_chip_tensors.keys(), key=lambda n: on_chip_tensors[n].last_access_step)


class MemoryScheduler:
    """
    Memory scheduler that generates transfer nodes for on-chip/off-chip memory management.

    Given memory constraints and tensor lifetimes, determines when to:
    - Load tensors from off-chip to on-chip memory
    - Store tensors from on-chip back to off-chip memory
    - Overwrite dying tensors without explicit store
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        config: MemoryTransferConfig,
    ):
        self.gm = gm
        self.config = config

    def schedule(
        self,
        tensors: Dict[str, 'TensorInfo'],
        eviction_policy: str = "furthest",
    ) -> MemorySchedule:
        """
        Generate memory schedule with transfer nodes.

        Args:
            tensors: Tensor info from MemoryAnalyzer
            eviction_policy: "furthest", "largest", or "lru"

        Returns:
            MemorySchedule with transfer nodes
        """
        schedule = MemorySchedule()

        # Filter out weights (assumed always available)
        activation_tensors = {
            name: t for name, t in tensors.items()
            if not t.is_weight
        }

        if not activation_tensors:
            return schedule

        # Build next-use map for Belady's algorithm
        next_use = self._build_next_use_map(activation_tensors)

        # Initialize tensor locations (all start off-chip except inputs)
        on_chip: Dict[str, TensorLocation] = {}
        off_chip: Dict[str, TensorLocation] = {}

        # Track on-chip memory usage
        on_chip_used = 0
        transfer_id = 0

        # Get eviction function
        evict_fn = {
            "furthest": EvictionPolicy.furthest_next_use,
            "largest": EvictionPolicy.largest_first,
            "lru": EvictionPolicy.lru,
        }.get(eviction_policy, EvictionPolicy.furthest_next_use)

        # Process each execution step
        num_steps = max(t.death_step for t in activation_tensors.values()) + 1

        for step in range(num_steps):
            # Find tensors needed at this step
            needed_tensors = self._get_tensors_needed_at_step(
                activation_tensors, step
            )

            # Find tensors that die at this step
            dying_tensors = [
                name for name, t in activation_tensors.items()
                if t.death_step == step and name in on_chip
            ]

            # Process each needed tensor
            for tensor_name in needed_tensors:
                if tensor_name not in activation_tensors:
                    continue

                tensor = activation_tensors[tensor_name]

                # Already on chip?
                if tensor_name in on_chip:
                    on_chip[tensor_name].last_access_step = step
                    continue

                # Need to load this tensor
                size = tensor.size_bytes

                # Check if we need to evict
                while on_chip_used + size > self.config.on_chip_size_bytes:
                    # Check for overwrite opportunity
                    overwrite_target = self._find_overwrite_candidate(
                        dying_tensors, size, on_chip
                    )

                    if overwrite_target:
                        # Can overwrite dying tensor
                        dying_tensors.remove(overwrite_target)
                        victim_loc = on_chip.pop(overwrite_target)
                        on_chip_used -= victim_loc.size_bytes
                        schedule.num_overwrites += 1

                        # Generate Store if in trace mode
                        if self.config.trace_mode and self.config.enable_store:
                            store_node = self._create_store_node(
                                transfer_id, overwrite_target, victim_loc, step
                            )
                            schedule.transfers.append(store_node)
                            schedule.total_store_bytes += victim_loc.size_bytes
                            schedule.total_store_latency_us += store_node.estimated_latency_us
                            transfer_id += 1
                    else:
                        # Must evict a tensor
                        if eviction_policy == "furthest":
                            victim_name = evict_fn(on_chip, step, next_use.get(step, {}))
                        else:
                            victim_name = evict_fn(on_chip, step)

                        if victim_name is None:
                            break  # No more to evict

                        victim_loc = on_chip.pop(victim_name)
                        on_chip_used -= victim_loc.size_bytes
                        schedule.num_evictions += 1

                        # Generate Store node
                        if self.config.enable_store:
                            store_node = self._create_store_node(
                                transfer_id, victim_name, victim_loc, step
                            )
                            schedule.transfers.append(store_node)
                            schedule.total_store_bytes += victim_loc.size_bytes
                            schedule.total_store_latency_us += store_node.estimated_latency_us
                            transfer_id += 1

                        # Move to off-chip
                        off_chip[victim_name] = victim_loc
                        victim_loc.location = MemoryLocation.OFF_CHIP

                # Generate Load node
                load_node = self._create_load_node(
                    transfer_id, tensor_name, tensor, step, on_chip_used
                )

                # Check for overwrite annotation
                if dying_tensors:
                    # Find best dying tensor to overwrite
                    for dying in dying_tensors:
                        if dying in on_chip and on_chip[dying].size_bytes >= size:
                            load_node.is_overwrite = True
                            load_node.overwrites_tensor = dying
                            load_node.dst_offset = on_chip[dying].on_chip_offset
                            break

                schedule.transfers.append(load_node)
                schedule.total_load_bytes += size
                schedule.total_load_latency_us += load_node.estimated_latency_us
                transfer_id += 1

                # Add to on-chip
                loc = TensorLocation(
                    name=tensor_name,
                    size_bytes=size,
                    location=MemoryLocation.ON_CHIP,
                    on_chip_offset=load_node.dst_offset,
                    birth_step=tensor.birth_step,
                    death_step=tensor.death_step,
                    last_access_step=step,
                )
                on_chip[tensor_name] = loc
                on_chip_used += size

                # Track peak
                schedule.peak_on_chip_bytes = max(
                    schedule.peak_on_chip_bytes, on_chip_used
                )

            # Clean up dead tensors from on-chip
            for dying in dying_tensors:
                if dying in on_chip:
                    victim_loc = on_chip.pop(dying)
                    on_chip_used -= victim_loc.size_bytes

        # Store final tensor locations
        schedule.tensor_locations = {**on_chip, **off_chip}

        return schedule

    def _build_next_use_map(
        self,
        tensors: Dict[str, 'TensorInfo']
    ) -> Dict[int, Dict[str, int]]:
        """Build map: step -> tensor -> next_use_step"""
        # First build full usage list for each tensor
        tensor_uses: Dict[str, List[int]] = {name: [] for name in tensors}

        for step, node in enumerate(self.gm.graph.nodes):
            if node.op == "output":
                continue

            # Check args
            for arg in node.args:
                if isinstance(arg, fx.Node) and arg.name in tensor_uses:
                    tensor_uses[arg.name].append(step)
                elif isinstance(arg, (list, tuple)):
                    for a in arg:
                        if isinstance(a, fx.Node) and a.name in tensor_uses:
                            tensor_uses[a.name].append(step)

        # Build next-use map for each step
        next_use: Dict[int, Dict[str, int]] = {}
        max_step = max(t.death_step for t in tensors.values()) + 1

        for step in range(max_step):
            next_use[step] = {}
            for name, uses in tensor_uses.items():
                future_uses = [u for u in uses if u > step]
                if future_uses:
                    next_use[step][name] = min(future_uses)
                else:
                    next_use[step][name] = float('inf')

        return next_use

    def _get_tensors_needed_at_step(
        self,
        tensors: Dict[str, 'TensorInfo'],
        step: int,
    ) -> List[str]:
        """Get tensors that are inputs to the node at this step"""
        needed = []

        for i, node in enumerate(self.gm.graph.nodes):
            if i != step:
                continue
            if node.op == "output":
                break

            # Collect input tensor names
            for arg in node.args:
                if isinstance(arg, fx.Node) and arg.name in tensors:
                    needed.append(arg.name)
                elif isinstance(arg, (list, tuple)):
                    for a in arg:
                        if isinstance(a, fx.Node) and a.name in tensors:
                            needed.append(a.name)

            for v in node.kwargs.values():
                if isinstance(v, fx.Node) and v.name in tensors:
                    needed.append(v.name)

            break

        return needed

    def _find_overwrite_candidate(
        self,
        dying_tensors: List[str],
        required_size: int,
        on_chip: Dict[str, TensorLocation],
    ) -> Optional[str]:
        """Find a dying tensor that can be overwritten"""
        for name in dying_tensors:
            if name in on_chip and on_chip[name].size_bytes >= required_size:
                return name
        return None

    def _create_load_node(
        self,
        transfer_id: int,
        tensor_name: str,
        tensor: 'TensorInfo',
        step: int,
        current_offset: int,
    ) -> TransferNode:
        """Create a Load transfer node"""
        size = tensor.size_bytes
        latency = self._estimate_latency(size, is_load=True)

        # Estimate trigger time: start loading before the step
        # Trigger early enough to complete before the tensor is needed
        trigger_step = max(0, step - latency / 10)  # Simple heuristic

        return TransferNode(
            name=f"load_{transfer_id}_{tensor_name}",
            transfer_type=TransferType.LOAD,
            tensor_name=tensor_name,
            size_bytes=size,
            src_offset=0,  # DDR offset (simplified)
            dst_offset=current_offset,
            src_memory=MemoryLocation.OFF_CHIP,
            dst_memory=MemoryLocation.ON_CHIP,
            estimated_latency_us=latency,
            trigger_step=trigger_step,
            insert_before_step=step,
        )

    def _create_store_node(
        self,
        transfer_id: int,
        tensor_name: str,
        loc: TensorLocation,
        step: int,
    ) -> TransferNode:
        """Create a Store transfer node"""
        size = loc.size_bytes
        latency = self._estimate_latency(size, is_load=False)

        return TransferNode(
            name=f"store_{transfer_id}_{tensor_name}",
            transfer_type=TransferType.STORE,
            tensor_name=tensor_name,
            size_bytes=size,
            src_offset=loc.on_chip_offset,
            dst_offset=0,  # DDR offset (simplified)
            src_memory=MemoryLocation.ON_CHIP,
            dst_memory=MemoryLocation.OFF_CHIP,
            estimated_latency_us=latency,
            trigger_step=step - 0.5,  # Trigger before step
            insert_before_step=step,
        )

    def _estimate_latency(self, size_bytes: int, is_load: bool) -> float:
        """Estimate transfer latency in microseconds"""
        bandwidth = (
            self.config.load_bandwidth_gbps if is_load
            else self.config.store_bandwidth_gbps
        )
        # latency = overhead + size / bandwidth
        transfer_time = size_bytes / (bandwidth * 1e9) * 1e6
        return self.config.transfer_overhead_us + transfer_time


def schedule_memory_transfers(
    gm: fx.GraphModule,
    tensors: Dict[str, 'TensorInfo'],
    on_chip_size_kb: float,
    enable_store: bool = True,
    trace_mode: bool = False,
    load_bandwidth_gbps: float = 100.0,
    store_bandwidth_gbps: float = 50.0,
) -> MemorySchedule:
    """
    Convenience function to schedule memory transfers.

    Args:
        gm: FX GraphModule
        tensors: Tensor info from MemoryAnalyzer
        on_chip_size_kb: On-chip memory size in KB
        enable_store: Generate explicit Store nodes
        trace_mode: Generate Store even for overwrites
        load_bandwidth_gbps: Load bandwidth for latency estimation
        store_bandwidth_gbps: Store bandwidth for latency estimation

    Returns:
        MemorySchedule with transfer nodes
    """
    config = MemoryTransferConfig(
        on_chip_size_bytes=int(on_chip_size_kb * 1024),
        load_bandwidth_gbps=load_bandwidth_gbps,
        store_bandwidth_gbps=store_bandwidth_gbps,
        enable_store=enable_store,
        trace_mode=trace_mode,
    )

    scheduler = MemoryScheduler(gm, config)
    return scheduler.schedule(tensors)


if __name__ == "__main__":
    # Test the memory scheduler
    import torch.nn as nn
    from .memory_analyzer import MemoryAnalyzer

    print("Testing MemoryScheduler...")
    print("-" * 50)

    # Build a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 256)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(256, 512)
            self.linear3 = nn.Linear(512, 128)

        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.relu(x)
            x = self.linear3(x)
            return x

    model = SimpleModel()
    gm = fx.symbolic_trace(model)

    # First run memory analysis
    analyzer = MemoryAnalyzer(gm)
    result = analyzer.analyze()

    print(f"Model has {len(result.tensors)} tensors")
    print(f"Peak memory: {result.peak_max_memory / 1024:.2f} KB")

    # Schedule with limited on-chip memory
    on_chip_kb = result.peak_max_memory / 1024 / 2  # Half of peak
    print(f"\nScheduling with on-chip limit: {on_chip_kb:.2f} KB")

    schedule = schedule_memory_transfers(
        gm, result.tensors,
        on_chip_size_kb=on_chip_kb,
        enable_store=True,
        trace_mode=False,
    )

    summary = schedule.to_dict()["summary"]
    print("\nSchedule Summary:")
    print(f"  Total transfers: {summary['total_transfers']}")
    print(f"  Loads: {summary['num_loads']}")
    print(f"  Stores: {summary['num_stores']}")
    print(f"  Overwrites: {summary['num_overwrites']}")
    print(f"  Peak on-chip: {summary['peak_on_chip_kb']:.2f} KB")
    print(f"  Total load: {summary['total_load_kb']:.2f} KB")
    print(f"  Total store: {summary['total_store_kb']:.2f} KB")
    print(f"  Total latency: {summary['total_latency_us']:.2f} us")

    print("\nTransfer nodes:")
    for t in schedule.transfers[:10]:
        print(f"  [{t.transfer_type.value}] {t.name}: "
              f"{t.size_bytes/1024:.2f} KB, trigger@{t.trigger_step:.1f}")

    print("\nMemoryScheduler test completed!")
