#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example: Memory analysis visualization"""

import sys
sys.path.insert(0, "/home/user/onnxparser")

from onnxparser.builder import GraphBuilder
from onnxparser.analysis import MemoryAnalyzer, MemoryConstraint
from onnxparser.visualizer import visualize_memory, serve_memory


def create_sample_model():
    """Create a sample transformer-like model for demonstration"""
    import torch

    builder = GraphBuilder("sample_transformer")

    # Input
    x = builder.input("input", [1, 128, 64], dtype="float32")

    # Self-attention block
    # Q, K, V projections
    q_weight = builder.constant("q_weight", torch.randn(64, 64), dtype="float32")
    k_weight = builder.constant("k_weight", torch.randn(64, 64), dtype="float32")
    v_weight = builder.constant("v_weight", torch.randn(64, 64), dtype="float32")

    q = builder.matmul(x, q_weight)
    k = builder.matmul(x, k_weight)
    v = builder.matmul(x, v_weight)

    # Attention scores
    k_t = builder.transpose(k, -2, -1)
    scores = builder.matmul(q, k_t)
    scores = builder.softmax(scores, dim=-1)

    # Attention output
    attn_out = builder.matmul(scores, v)

    # Residual connection
    x = builder.add(x, attn_out)

    # Layer norm
    ln_weight = builder.constant("ln_weight", torch.randn(64), dtype="float32")
    ln_bias = builder.constant("ln_bias", torch.randn(64), dtype="float32")
    x = builder.layer_norm(x, [64], ln_weight, ln_bias)

    # FFN
    ffn_w1 = builder.constant("ffn_w1", torch.randn(64, 256), dtype="float32")
    ffn_w2 = builder.constant("ffn_w2", torch.randn(256, 64), dtype="float32")

    ffn = builder.matmul(x, ffn_w1)
    ffn = builder.gelu(ffn)
    ffn = builder.matmul(ffn, ffn_w2)

    # Final residual
    x = builder.add(x, ffn)

    builder.output(x)

    return builder.build()


def main():
    print("Creating sample model...")
    gm = create_sample_model()
    print(f"Model created with {len(list(gm.graph.nodes))} nodes")

    # Run memory analysis
    print("\nRunning memory analysis...")
    analyzer = MemoryAnalyzer(gm, strategy="greedy")
    result = analyzer.analyze()

    print(f"\n=== Memory Analysis Results ===")
    print(f"Strategy: {result.strategy_name}")
    print(f"Peak Max Memory: {result.peak_max_memory / 1e6:.2f} MB")
    print(f"Peak Min Memory: {result.peak_min_memory / 1e6:.2f} MB")
    print(f"Memory Savings: {result.savings_ratio * 100:.1f}%")
    print(f"Static Memory (Weights): {result.static_memory / 1e6:.2f} MB")

    # Compare strategies
    print("\n=== Strategy Comparison ===")
    comparison = analyzer.compare_strategies()
    for name, res in comparison.items():
        print(f"  {name:15s}: Peak={res.peak_min_memory/1e6:8.2f} MB, "
              f"Savings={res.savings_ratio*100:5.1f}%")

    # Test with memory constraint
    print("\n=== Memory Constraint Test ===")
    constraint = MemoryConstraint(max_memory_mb=0.02)  # 20KB limit
    constrained_analyzer = MemoryAnalyzer(gm, strategy="greedy", constraint=constraint)
    constrained_result = constrained_analyzer.analyze()

    print(f"Memory Limit: {constraint.max_memory_mb:.2f} MB")
    print(f"Peak Memory: {constrained_result.peak_min_memory / 1e6:.4f} MB")
    print(f"Fits in Limit: {constrained_result.fits_in_limit}")
    print(f"Overflow: {constrained_result.overflow_bytes / 1e6:.4f} MB")
    print(f"Exceeded Count: {constrained_result.exceeded_count}")

    # Generate visualization
    print("\nGenerating memory visualization...")
    output_path = "/home/user/onnxparser/examples/memory_analysis.html"
    visualize_memory(gm, output_path)
    print(f"Visualization saved to: {output_path}")

    # Optionally serve
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        print("\nStarting visualization server...")
        serve_memory(gm, port=8080)


if __name__ == "__main__":
    main()
