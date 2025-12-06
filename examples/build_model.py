# -*- coding: utf-8 -*-
"""Example: Build model using GraphBuilder"""

import torch
import sys
sys.path.insert(0, "..")

from onnxparser.builder import GraphBuilder


def build_simple_mlp():
    """Build a simple MLP"""
    print("=" * 50)
    print("Example 1: Simple MLP")
    print("=" * 50)

    builder = GraphBuilder("simple_mlp")

    # Input
    x = builder.input("input", [1, 784], dtype="float32")

    # Hidden layers
    x = builder.linear(x, 256, name="fc1")
    x = builder.relu(x)
    x = builder.linear(x, 128, name="fc2")
    x = builder.relu(x)

    # Output layer
    x = builder.linear(x, 10, name="fc3")
    x = builder.softmax(x, dim=-1)

    builder.output(x)

    # Build GraphModule
    gm = builder.build()
    print(gm.graph)

    # Test run
    test_input = torch.randn(1, 784)
    output = gm(test_input)
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    return gm


def build_with_operators():
    """Build using operator overloading"""
    print("\n" + "=" * 50)
    print("Example 2: Operator Overloading")
    print("=" * 50)

    builder = GraphBuilder("operator_demo")

    x = builder.input("x", [1, 64], dtype="float32")
    y = builder.input("y", [1, 64], dtype="float32")

    # Operator overloading
    z = x + y
    z = z * 2
    z = z - 1
    z = -z

    builder.output(z)

    gm = builder.build()
    print(gm.graph)

    # Test
    x_data = torch.randn(1, 64)
    y_data = torch.randn(1, 64)
    output = gm(x_data, y_data)
    print(f"\nOutput shape: {output.shape}")

    return gm


def build_transformer_block():
    """Build a Transformer Block"""
    print("\n" + "=" * 50)
    print("Example 3: Transformer Block (simplified)")
    print("=" * 50)

    builder = GraphBuilder("transformer_block")

    # Input: [batch, seq_len, hidden]
    x = builder.input("hidden", [1, 128, 512], dtype="float32")

    # Self-Attention (simplified)
    q = builder.linear(x, 512, name="q_proj")
    k = builder.linear(x, 512, name="k_proj")
    v = builder.linear(x, 512, name="v_proj")

    # Attention: softmax(Q @ K^T / sqrt(d)) @ V
    attn_out = builder.attention(q, k, v, scale=1.0 / (512 ** 0.5))
    attn_out = builder.linear(attn_out, 512, name="out_proj")

    # Residual + LayerNorm
    x = x + attn_out
    x = builder.layer_norm(x, [512])

    # FFN
    ffn = builder.linear(x, 2048, name="ffn1")
    ffn = builder.gelu(ffn)
    ffn = builder.linear(ffn, 512, name="ffn2")

    # Residual + LayerNorm
    x = x + ffn
    x = builder.layer_norm(x, [512])

    builder.output(x)

    gm = builder.build()
    print(gm.graph)

    # Test
    test_input = torch.randn(1, 128, 512)
    output = gm(test_input)
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    return gm


if __name__ == "__main__":
    build_simple_mlp()
    build_with_operators()
    build_transformer_block()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)
