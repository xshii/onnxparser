# -*- coding: utf-8 -*-
"""Tests for ONNX to FX Parser"""

import pytest
import torch
import tempfile
import os

from onnxparser.builder import GraphBuilder


class TestOnnxToFx:
    """Test ONNX to FX conversion"""

    def test_parser_import(self):
        """Test that parser module can be imported"""
        from onnxparser.parser import load
        assert load is not None

    def test_load_simple_model(self):
        """Test loading a simple ONNX model"""
        from onnxparser.parser import load
        import onnx
        from onnx import helper, TensorProto

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple ONNX model
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])

            relu_node = helper.make_node("Relu", ["X"], ["Y"])

            graph = helper.make_graph([relu_node], "test", [X], [Y])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

            onnx_path = os.path.join(tmpdir, "simple.onnx")
            onnx.save(model, onnx_path)

            # Load and test
            gm = load(onnx_path)
            assert gm is not None

            inp = torch.randn(1, 64)
            out = gm(inp)
            expected = torch.relu(inp)
            assert torch.allclose(out, expected, atol=1e-5)
