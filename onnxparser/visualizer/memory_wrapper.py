# -*- coding: utf-8 -*-
"""Memory analysis wrapper for visualization"""

from typing import Dict, Any, List, Optional

import torch
import torch.fx as fx


class MemoryTransferWrapper:
    """Wrapper for memory transfer scheduling"""

    @staticmethod
    def schedule_transfers(
        gm: fx.GraphModule,
        on_chip_size_kb: float,
        strategy: str = "no_reuse",
        enable_store: bool = True,
        trace_mode: bool = False,
        input_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Schedule memory transfers for on-chip/off-chip memory management.

        Args:
            gm: FX GraphModule
            on_chip_size_kb: On-chip memory size in KB
            strategy: Memory allocation strategy (default: no_reuse for accurate sizing)
            enable_store: Generate explicit Store nodes
            trace_mode: Generate Store even for overwrites
            input_data: Optional input data for shape propagation
        """
        try:
            from ..analysis.memory_analyzer import MemoryAnalyzer
            from ..analysis.memory_scheduler import schedule_memory_transfers

            # First run shape propagation
            if input_data:
                MemoryTransferWrapper._propagate_shapes(gm, input_data)

            # Run memory analysis to get tensor info
            # Use no_reuse as default to get accurate tensor sizes without reuse
            analyzer = MemoryAnalyzer(gm, strategy=strategy)
            result = analyzer.analyze()

            # Schedule transfers
            schedule = schedule_memory_transfers(
                gm,
                result.tensors,
                on_chip_size_kb=on_chip_size_kb,
                enable_store=enable_store,
                trace_mode=trace_mode,
            )

            return schedule.to_dict()

        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

    @staticmethod
    def _propagate_shapes(gm: fx.GraphModule, input_data: Dict[str, torch.Tensor]) -> None:
        """Propagate shapes through the graph"""
        try:
            from torch.fx.passes.shape_prop import ShapeProp

            example_inputs = []
            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    if node.name in input_data:
                        example_inputs.append(input_data[node.name])
                    else:
                        for key, val in input_data.items():
                            if key in node.name or node.name in key:
                                example_inputs.append(val)
                                break

            if example_inputs:
                ShapeProp(gm).propagate(*example_inputs)
        except Exception:
            pass


class MemoryAnalyzerWrapper:
    """Wrapper for memory analysis integration"""

    @staticmethod
    def analyze(
        gm: fx.GraphModule,
        strategy: str = "greedy",
        input_data: Optional[Dict[str, torch.Tensor]] = None,
        memory_limit_kb: Optional[float] = None
    ) -> Dict[str, Any]:
        """Run memory analysis with specified strategy

        Args:
            gm: FX GraphModule to analyze
            strategy: Memory allocation strategy
            input_data: Optional input data for shape propagation
            memory_limit_kb: Optional memory limit in KB
        """
        try:
            from ..analysis.memory_analyzer import MemoryAnalyzer, MemoryConstraint

            # Propagate shapes if input_data is provided
            if input_data:
                MemoryAnalyzerWrapper._propagate_shapes(gm, input_data)

            # Create memory constraint if limit is specified
            constraint = None
            if memory_limit_kb is not None:
                constraint = MemoryConstraint(
                    max_memory_bytes=int(memory_limit_kb * 1024)
                )

            analyzer = MemoryAnalyzer(gm, strategy=strategy, constraint=constraint)
            result = analyzer.analyze()

            tensors = []
            for name, tensor in result.tensors.items():
                if tensor.is_weight:
                    continue
                tensors.append({
                    "name": name,
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype),
                    "size_bytes": tensor.size_bytes,
                    "size_kb": tensor.size_bytes / 1024,
                    "birth": tensor.birth_step,
                    "death": tensor.death_step,
                    "is_input": tensor.is_input,
                    "is_output": tensor.is_output,
                    "reused_from": tensor.reused_from,
                    "memory_offset": tensor.memory_offset,
                })

            timeline = []
            for step in result.steps:
                timeline.append({
                    "step": step.step,
                    "node": step.node_name,
                    "max_memory": step.max_memory,
                    "min_memory": step.min_memory,
                    "live_tensors": step.live_tensors,
                    "exceeded_limit": step.exceeded_limit,
                })

            # Build constraint info
            constraint_info = {}
            if memory_limit_kb is not None:
                constraint_info = {
                    "limit_kb": memory_limit_kb,
                    "exceeded_count": result.exceeded_count,
                    "fits_in_limit": result.fits_in_limit,
                    "overflow_kb": result.overflow_bytes / 1024,
                }

            return {
                "strategy": strategy,
                "tensors": tensors,
                "timeline": timeline,
                "summary": {
                    "peak_max_kb": result.peak_max_memory / 1024,
                    "peak_min_kb": result.peak_min_memory / 1024,
                    "savings_pct": result.savings_ratio * 100,
                },
                "constraint": constraint_info,
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _propagate_shapes(gm: fx.GraphModule, input_data: Dict[str, torch.Tensor]) -> None:
        """Propagate shapes through the graph using input data"""
        try:
            from torch.fx.passes.shape_prop import ShapeProp

            # Build input args based on graph placeholder order
            example_inputs = []
            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    if node.name in input_data:
                        example_inputs.append(input_data[node.name])
                    else:
                        # Try to find a matching key
                        for key, val in input_data.items():
                            if key in node.name or node.name in key:
                                example_inputs.append(val)
                                break

            if example_inputs:
                ShapeProp(gm).propagate(*example_inputs)
        except Exception as e:
            # Shape propagation failed, shapes may not be available
            import warnings
            warnings.warn(f"Shape propagation failed: {e}")

    @staticmethod
    def list_strategies() -> List[str]:
        """List available memory strategies"""
        try:
            from ..analysis.strategies import StrategyRegistry
            return StrategyRegistry.list_strategies()
        except ImportError:
            return ["greedy"]


def main():
    """Test memory analyzer wrapper with a demo model"""
    import torch.nn as nn

    print("Testing MemoryAnalyzerWrapper...")
    print("-" * 50)

    # Build a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 128)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(128, 256)
            self.linear3 = nn.Linear(256, 64)

        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.relu(x)
            x = self.linear3(x)
            return x

    model = SimpleModel()
    gm = fx.symbolic_trace(model)

    print(f"Model: {model.__class__.__name__}")
    print(f"Available strategies: {MemoryAnalyzerWrapper.list_strategies()}")

    # Analyze with each strategy
    for strategy in MemoryAnalyzerWrapper.list_strategies():
        print(f"\n--- Strategy: {strategy} ---")
        result = MemoryAnalyzerWrapper.analyze(gm, strategy)

        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        print(f"  Tensors: {len(result.get('tensors', []))}")
        print(f"  Timeline steps: {len(result.get('timeline', []))}")

        summary = result.get("summary", {})
        print(f"  Peak memory (max): {summary.get('peak_max_kb', 'N/A'):.2f} KB")
        print(f"  Peak memory (min): {summary.get('peak_min_kb', 'N/A'):.2f} KB")
        print(f"  Savings: {summary.get('savings_pct', 0):.1f}%")

        # Show tensor details
        tensors = result.get("tensors", [])[:5]
        if tensors:
            print("\n  Sample tensors:")
            for t in tensors:
                reused = f" (reused from {t['reused_from']})" if t.get('reused_from') else ""
                print(f"    - {t['name']}: {t['shape']} {t['dtype']}, "
                      f"birth={t['birth']}, death={t['death']}{reused}")

    print("\nMemoryAnalyzerWrapper test completed!")


if __name__ == "__main__":
    main()
