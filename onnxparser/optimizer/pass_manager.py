# -*- coding: utf-8 -*-
"""Pass Manager - Orchestrate optimization passes"""

import torch.fx as fx
from typing import List, Callable, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class PassLevel(Enum):
    """Optimization level for passes"""
    O0 = 0  # No optimization
    O1 = 1  # Basic optimizations (dead code, identity removal)
    O2 = 2  # Standard optimizations (+ folding, simplification)
    O3 = 3  # Aggressive optimizations (+ fusion)


@dataclass
class PassResult:
    """Result of running a pass"""
    name: str
    success: bool
    changed: bool = False
    error: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


class PassManager:
    """Manage and execute optimization passes on FX GraphModule"""

    def __init__(self, level: PassLevel = PassLevel.O2):
        self.level = level
        self._passes: List[tuple[str, Callable, PassLevel]] = []
        self._results: List[PassResult] = []

        # Register default passes
        self._register_default_passes()

    def _register_default_passes(self) -> None:
        """Register built-in passes based on optimization level"""
        from onnxparser.optimizer.passes import (
            eliminate_dead_code,
            constant_folding,
            fuse_linear_relu,
            fuse_bn_into_conv,
            fuse_consecutive_transpose,
            fuse_attention,
            remove_identity,
            simplify_reshape,
            remove_redundant_cast,
        )

        # O1: Basic
        self.register("eliminate_dead_code", eliminate_dead_code, PassLevel.O1)
        self.register("remove_identity", remove_identity, PassLevel.O1)

        # O2: Standard
        self.register("constant_folding", constant_folding, PassLevel.O2)
        self.register("simplify_reshape", simplify_reshape, PassLevel.O2)
        self.register("remove_redundant_cast", remove_redundant_cast, PassLevel.O2)
        self.register("fuse_consecutive_transpose", fuse_consecutive_transpose, PassLevel.O2)

        # O3: Aggressive
        self.register("fuse_linear_relu", fuse_linear_relu, PassLevel.O3)
        self.register("fuse_bn_into_conv", fuse_bn_into_conv, PassLevel.O3)
        self.register("fuse_attention", fuse_attention, PassLevel.O3)

    def register(
        self,
        name: str,
        pass_fn: Callable[[fx.GraphModule], fx.GraphModule],
        level: PassLevel = PassLevel.O2,
    ) -> None:
        """Register a custom pass"""
        self._passes.append((name, pass_fn, level))

    def run(self, gm: fx.GraphModule) -> fx.GraphModule:
        """Run all applicable passes on the graph"""
        self._results = []
        original_node_count = len(list(gm.graph.nodes))

        for name, pass_fn, pass_level in self._passes:
            if pass_level.value > self.level.value:
                continue

            before_count = len(list(gm.graph.nodes))

            try:
                gm = pass_fn(gm)
                after_count = len(list(gm.graph.nodes))

                self._results.append(PassResult(
                    name=name,
                    success=True,
                    changed=before_count != after_count,
                    stats={"nodes_before": before_count, "nodes_after": after_count},
                ))
            except Exception as e:
                self._results.append(PassResult(
                    name=name,
                    success=False,
                    error=str(e),
                ))

        final_node_count = len(list(gm.graph.nodes))

        return gm

    def run_pass(self, gm: fx.GraphModule, name: str) -> fx.GraphModule:
        """Run a specific pass by name"""
        for pass_name, pass_fn, _ in self._passes:
            if pass_name == name:
                return pass_fn(gm)
        raise ValueError(f"Pass '{name}' not found")

    def get_results(self) -> List[PassResult]:
        """Get results from last run"""
        return self._results

    def summary(self) -> str:
        """Get summary of last run"""
        lines = ["Optimization Summary:"]
        lines.append(f"  Level: {self.level.name}")
        lines.append(f"  Passes run: {len(self._results)}")

        for r in self._results:
            status = "✓" if r.success else "✗"
            changed = "(changed)" if r.changed else ""
            if r.error:
                lines.append(f"  {status} {r.name}: {r.error}")
            else:
                lines.append(f"  {status} {r.name} {changed}")

        return "\n".join(lines)


def optimize(
    gm: fx.GraphModule,
    level: PassLevel = PassLevel.O2,
) -> fx.GraphModule:
    """Convenience function to optimize a GraphModule"""
    pm = PassManager(level=level)
    return pm.run(gm)
