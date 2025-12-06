# -*- coding: utf-8 -*-
"""Tests for ONNX export"""

import pytest
import torch
import os
import tempfile

from onnxparser.builder import GraphBuilder
from onnxparser.export import export_onnx, export_onnx_multi_version


class TestExportOnnx:
    """Test ONNX export functionality"""

    def test_export_simple_model(self):
        """Test exporting a simple model to ONNX"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        x = builder.relu(x)
        builder.output(x)
        gm = builder.build()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            export_onnx(gm, onnx_path, input_shapes={"x": [1, 64]})

            assert os.path.exists(onnx_path)
            assert os.path.getsize(onnx_path) > 0

    def test_export_multi_version(self):
        """Test exporting to multiple ONNX versions"""
        builder = GraphBuilder("test")
        x = builder.input("x", [1, 64])
        x = builder.linear(x, 32)
        builder.output(x)
        gm = builder.build()

        with tempfile.TemporaryDirectory() as tmpdir:
            results = export_onnx_multi_version(
                gm,
                output_dir=tmpdir,
                input_shapes={"x": [1, 64]},
                opset_versions=[11, 14, 17],
                model_name="test",
            )

            # At least opset 11, 14, 17 should succeed
            assert results[11] is not None
            assert results[14] is not None
            assert results[17] is not None

    def test_export_tiny_transformer(self):
        """Test exporting tiny transformer model"""
        from examples.tiny_transformer import build_tiny_transformer

        gm = build_tiny_transformer(
            input_dim=(28, 68, 119),
            d_model=64,
            n_heads=4,
            n_layers=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "transformer.onnx")
            export_onnx(
                gm,
                onnx_path,
                input_shapes={"input": [28, 68, 119]},
                opset_version=17,
            )

            assert os.path.exists(onnx_path)

            # Verify with onnx
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)

    def test_onnx_inference_matches(self):
        """Test that ONNX output matches PyTorch output"""
        builder = GraphBuilder("test")
        x = builder.input("x", [2, 64])
        x = builder.linear(x, 32)
        x = builder.relu(x)
        builder.output(x)
        gm = builder.build()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            export_onnx(gm, onnx_path, input_shapes={"x": [2, 64]})

            # Run PyTorch
            inp = torch.randn(2, 64)
            torch_out = gm(inp)

            # Run ONNX
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path)
            onnx_out = sess.run(None, {"x": inp.numpy()})[0]

            # Compare
            assert torch.allclose(
                torch_out,
                torch.from_numpy(onnx_out),
                atol=1e-5,
            )
