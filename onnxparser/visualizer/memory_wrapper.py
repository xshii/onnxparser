# -*- coding: utf-8 -*-
"""Memory analysis wrapper for visualization"""

from typing import Dict, Any, List

import torch.fx as fx


class MemoryAnalyzerWrapper:
    """Wrapper for memory analysis integration"""

    @staticmethod
    def analyze(gm: fx.GraphModule, strategy: str = "greedy") -> Dict[str, Any]:
        """Run memory analysis with specified strategy"""
        try:
            from ..analysis.memory_analyzer import MemoryAnalyzer

            analyzer = MemoryAnalyzer(gm, strategy=strategy)
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
                })

            return {
                "strategy": strategy,
                "tensors": tensors,
                "timeline": timeline,
                "summary": {
                    "peak_max_kb": result.peak_max_memory / 1024,
                    "peak_min_kb": result.peak_min_memory / 1024,
                    "savings_pct": result.savings_ratio * 100,
                }
            }
        except Exception as e:
            return {"error": str(e)}

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
    import torch
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
