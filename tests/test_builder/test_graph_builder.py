# -*- coding: utf-8 -*-
"""Tests for GraphBuilder"""

import pytest
import torch
from onnxparser.builder import GraphBuilder


class TestGraphBuilder:
    """Test GraphBuilder functionality"""

    def test_simple_linear(self):
        """Test building a simple linear model"""
        builder = GraphBuilder("test_linear")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 128)
        builder.output(x)
        gm = builder.build()

        assert gm is not None
        assert len(list(gm.graph.nodes)) > 0

    def test_mlp(self):
        """Test building MLP with activations"""
        builder = GraphBuilder("test_mlp")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 128)
        x = builder.relu(x)
        x = builder.linear(x, 64)
        x = builder.relu(x)
        x = builder.linear(x, 10)
        builder.output(x)
        gm = builder.build()

        # Test inference
        inp = torch.randn(1, 64)
        out = gm(inp)
        assert out.shape == (1, 10)

    def test_operator_overloading(self):
        """Test operator overloading (+, -, *, @)"""
        builder = GraphBuilder("test_ops")
        x = builder.input("x", [1, 64])
        y = builder.input("y", [1, 64])

        # Addition
        z = x + y
        builder.output(z)
        gm = builder.build()

        inp_x = torch.randn(1, 64)
        inp_y = torch.randn(1, 64)
        out = gm(inp_x, inp_y)
        expected = inp_x + inp_y
        assert torch.allclose(out, expected)

    def test_attention(self):
        """Test attention block"""
        builder = GraphBuilder("test_attention")
        q = builder.input("q", [1, 16, 64])  # [batch, seq, dim]
        k = builder.input("k", [1, 16, 64])
        v = builder.input("v", [1, 16, 64])
        x = builder.attention(q, k, v)
        builder.output(x)
        gm = builder.build()

        inp_q = torch.randn(1, 16, 64)
        inp_k = torch.randn(1, 16, 64)
        inp_v = torch.randn(1, 16, 64)
        out = gm(inp_q, inp_k, inp_v)
        assert out.shape == (1, 16, 64)

    def test_layer_norm(self):
        """Test layer normalization"""
        builder = GraphBuilder("test_ln")
        x = builder.input("x", [1, 16, 64])
        x = builder.layer_norm(x, [64])  # normalized_shape must be a list
        builder.output(x)
        gm = builder.build()

        inp = torch.randn(1, 16, 64)
        out = gm(inp)
        assert out.shape == (1, 16, 64)

    def test_matmul(self):
        """Test matmul operation"""
        builder = GraphBuilder("test_matmul")
        x = builder.input("x", [2, 3, 4])
        y = builder.input("y", [2, 4, 5])
        z = x @ y  # operator overloading
        builder.output(z)
        gm = builder.build()

        inp_x = torch.randn(2, 3, 4)
        inp_y = torch.randn(2, 4, 5)
        out = gm(inp_x, inp_y)
        assert out.shape == (2, 3, 5)
