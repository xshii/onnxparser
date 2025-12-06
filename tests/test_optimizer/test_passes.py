# -*- coding: utf-8 -*-
"""Tests for Optimizer Passes"""

import pytest
import torch
from onnxparser.builder import GraphBuilder
from onnxparser.optimizer import (
    PassManager,
    PassLevel,
    optimize,
    eliminate_dead_code,
    constant_folding,
    remove_identity,
)


class TestPassManager:
    """Test PassManager functionality"""

    def test_pass_manager_levels(self):
        """Test different optimization levels"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        for level in [PassLevel.O0, PassLevel.O1, PassLevel.O2, PassLevel.O3]:
            pm = PassManager(level=level)
            optimized = pm.run(gm)
            assert optimized is not None

    def test_pass_manager_summary(self):
        """Test summary output"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        pm = PassManager(level=PassLevel.O3)
        pm.run(gm)
        summary = pm.summary()

        assert "Optimization Summary" in summary
        assert "Level: O3" in summary

    def test_optimize_function(self):
        """Test convenience optimize function"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        optimized = optimize(gm, level=PassLevel.O2)
        assert optimized is not None


class TestIndividualPasses:
    """Test individual optimization passes"""

    def test_eliminate_dead_code(self):
        """Test dead code elimination"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        optimized = eliminate_dead_code(gm)
        assert optimized is not None

    def test_constant_folding(self):
        """Test constant folding"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        optimized = constant_folding(gm)
        assert optimized is not None

    def test_remove_identity(self):
        """Test identity removal"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        optimized = remove_identity(gm)
        assert optimized is not None

    def test_optimized_model_correctness(self):
        """Test that optimization preserves model correctness"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 128)
        x = builder.relu(x)
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        # Get original output
        inp = torch.randn(1, 64)
        original_out = gm(inp)

        # Optimize
        optimized = optimize(gm, level=PassLevel.O3)
        optimized_out = optimized(inp)

        # Results should match
        assert torch.allclose(original_out, optimized_out, atol=1e-5)
