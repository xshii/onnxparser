# -*- coding: utf-8 -*-
"""Runtime - Execute FX GraphModule"""

import torch
import torch.fx as fx
from typing import Dict, Any, Union, List


class Runtime:
    """Simple runtime wrapper for FX GraphModule"""

    def __init__(self, gm: fx.GraphModule):
        self.gm = gm
        self.device = "cpu"

    def to(self, device: str) -> "Runtime":
        """Move model to device"""
        self.device = device
        self.gm = self.gm.to(device)
        return self

    def eval(self) -> "Runtime":
        """Set to evaluation mode"""
        self.gm.eval()
        return self

    def train(self) -> "Runtime":
        """Set to training mode"""
        self.gm.train()
        return self

    def run(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Run inference"""
        with torch.no_grad():
            if isinstance(inputs, dict):
                return self.gm(**inputs)
            elif isinstance(inputs, (list, tuple)):
                return self.gm(*inputs)
            else:
                return self.gm(inputs)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Direct call"""
        return self.gm(*args, **kwargs)

    def benchmark(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        warmup: int = 10,
        runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark inference time"""
        import time

        # Warmup
        for _ in range(warmup):
            self.run(inputs)

        # Benchmark
        if self.device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            self.run(inputs)
            if self.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        return {
            "mean_ms": sum(times) / len(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "runs": runs,
        }
