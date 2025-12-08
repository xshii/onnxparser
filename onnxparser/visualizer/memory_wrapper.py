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
