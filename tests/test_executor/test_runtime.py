# -*- coding: utf-8 -*-
"""Tests for Runtime"""

import pytest
import torch
from onnxparser.builder import GraphBuilder
from onnxparser.executor import Runtime


class TestRuntime:
    """Test Runtime functionality"""

    def test_basic_inference(self):
        """Test basic inference"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        x = builder.relu(x)
        builder.output(x)
        gm = builder.build()

        runtime = Runtime(gm)
        inp = torch.randn(1, 64)
        out = runtime.run(inp)
        assert out.shape == (1, 32)

    def test_dict_input(self):
        """Test inference with dict input"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        runtime = Runtime(gm)
        out = runtime.run({"x": torch.randn(1, 64)})
        assert out.shape == (1, 32)

    def test_eval_mode(self):
        """Test eval mode"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        runtime = Runtime(gm).eval()
        assert not runtime.gm.training

    def test_train_mode(self):
        """Test train mode"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        runtime = Runtime(gm).eval().train()
        assert runtime.gm.training

    def test_callable(self):
        """Test direct call"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        runtime = Runtime(gm)
        inp = torch.randn(1, 64)
        out = runtime(inp)
        assert out.shape == (1, 32)

    def test_benchmark(self):
        """Test benchmark function"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        runtime = Runtime(gm)
        inp = torch.randn(1, 64)
        result = runtime.benchmark(inp, warmup=2, runs=5)

        assert "mean_ms" in result
        assert "min_ms" in result
        assert "max_ms" in result
        assert result["runs"] == 5
