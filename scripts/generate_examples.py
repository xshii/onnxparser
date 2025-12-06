#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate example ONNX models for testing

This script generates example models in various ONNX opset versions.
Run this after installation to create test fixtures.

Usage:
    python scripts/generate_examples.py
    # or
    python -m scripts.generate_examples
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onnxparser.builder import GraphBuilder
from onnxparser.export import export_onnx, export_onnx_multi_version, export_excel


def generate_simple_mlp():
    """Generate a simple MLP model"""
    print("Generating simple MLP...")

    builder = GraphBuilder("simple_mlp")
    x = builder.input("x", [1, 64])
    x = builder.linear(x, 128)
    x = builder.relu(x)
    x = builder.linear(x, 64)
    x = builder.relu(x)
    x = builder.linear(x, 10)
    builder.output(x)

    return builder.build(), {"x": [1, 64]}


def generate_tiny_transformer():
    """Generate tiny transformer model"""
    print("Generating tiny transformer...")

    from examples.tiny_transformer import build_tiny_transformer

    gm = build_tiny_transformer(
        input_dim=(28, 68, 119),
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
    )

    return gm, {"input": [28, 68, 119]}


def generate_attention_only():
    """Generate attention-only model"""
    print("Generating attention-only model...")

    builder = GraphBuilder("attention_only")
    q = builder.input("query", [8, 32, 64])
    k = builder.input("key", [8, 32, 64])
    v = builder.input("value", [8, 32, 64])
    out = builder.attention(q, k, v)
    builder.output(out)

    return builder.build(), {
        "query": [8, 32, 64],
        "key": [8, 32, 64],
        "value": [8, 32, 64],
    }


def main():
    output_dir = "models/examples"
    os.makedirs(output_dir, exist_ok=True)

    # Models to generate
    models = [
        ("simple_mlp", generate_simple_mlp),
        ("tiny_transformer", generate_tiny_transformer),
        ("attention_only", generate_attention_only),
    ]

    # ONNX opset versions to export
    opset_versions = [11, 13, 14, 17]

    print(f"Output directory: {output_dir}")
    print(f"Opset versions: {opset_versions}")
    print()

    for model_name, generator_fn in models:
        print(f"\n{'='*50}")
        gm, input_shapes = generator_fn()

        # Export Excel
        excel_path = os.path.join(output_dir, f"{model_name}.xlsx")
        export_excel(gm, excel_path)
        print(f"  Excel: {excel_path}")

        # Export ONNX versions
        onnx_dir = os.path.join(output_dir, "onnx")
        results = export_onnx_multi_version(
            gm,
            output_dir=onnx_dir,
            input_shapes=input_shapes,
            opset_versions=opset_versions,
            model_name=model_name,
        )

        success_count = sum(1 for v in results.values() if v)
        print(f"  ONNX: {success_count}/{len(opset_versions)} versions exported")

    print(f"\n{'='*50}")
    print("Done!")

    # List generated files
    print("\nGenerated files:")
    for root, dirs, files in os.walk(output_dir):
        for f in sorted(files):
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            print(f"  {path} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
