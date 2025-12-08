#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example: Static memory allocation with fixed addresses"""

import sys
sys.path.insert(0, "/home/user/onnxparser")

from onnxparser.builder import GraphBuilder
from onnxparser.analysis import MemoryAnalyzer, StaticAllocationStrategy
import torch


def create_sample_model():
    """Create a sample model"""
    builder = GraphBuilder("sample_model")

    # Input: [1, 128, 64]
    x = builder.input("input", [1, 128, 64], dtype="float32")

    # Q, K, V projections
    q_weight = builder.constant("q_weight", torch.randn(64, 64), dtype="float32")
    k_weight = builder.constant("k_weight", torch.randn(64, 64), dtype="float32")
    v_weight = builder.constant("v_weight", torch.randn(64, 64), dtype="float32")

    q = builder.matmul(x, q_weight)
    k = builder.matmul(x, k_weight)
    v = builder.matmul(x, v_weight)

    # Attention
    k_t = builder.transpose(k, -2, -1)
    scores = builder.matmul(q, k_t)
    scores = builder.softmax(scores, dim=-1)
    attn_out = builder.matmul(scores, v)

    # Output
    builder.output(attn_out)
    return builder.build()


def main():
    print("Creating sample model...")
    gm = create_sample_model()
    print(f"Model created with {len(list(gm.graph.nodes))} nodes\n")

    # Compare strategies
    strategies = ["no_reuse", "greedy", "static"]

    print("=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    for strategy in strategies:
        analyzer = MemoryAnalyzer(gm, strategy=strategy)
        result = analyzer.analyze()

        print(f"\n--- {strategy.upper()} Strategy ---")
        print(f"Peak Memory: {result.peak_min_memory / 1024:.1f} KB")
        print(f"Total Allocations: {len([t for t in result.tensors.values() if not t.is_weight])}")

    # Show static allocation details
    print("\n" + "=" * 70)
    print("STATIC ALLOCATION DETAILS")
    print("=" * 70)

    analyzer = MemoryAnalyzer(gm, strategy="static")
    result = analyzer.analyze()

    print("\nFixed Memory Addresses (offset -> tensor):")
    print("-" * 50)

    # Sort by offset
    tensors_by_offset = sorted(
        [(t.name, t.memory_offset, t.size_bytes)
         for t in result.tensors.values()
         if not t.is_weight and t.memory_offset >= 0],
        key=lambda x: x[1]
    )

    for name, offset, size in tensors_by_offset:
        print(f"  0x{offset:08X} ({offset:6d}) : {name:20s} [{size/1024:.1f} KB]")

    # Generate C-style memory map
    print("\n" + "=" * 70)
    print("C-STYLE MEMORY MAP")
    print("=" * 70)

    static_strategy = StaticAllocationStrategy()

    # Pre-compute offsets using tensor info
    tensors_list = [t for t in result.tensors.values() if not t.is_weight]
    static_strategy.precompute_offsets(tensors_list)

    print(static_strategy.generate_memory_map())

    # Show memory layout visualization
    print("\n" + "=" * 70)
    print("MEMORY LAYOUT VISUALIZATION")
    print("=" * 70)

    max_offset = max(t.memory_offset + t.size_bytes
                     for t in result.tensors.values()
                     if not t.is_weight and t.memory_offset >= 0)

    scale = 60 / max_offset if max_offset > 0 else 1

    for name, offset, size in tensors_by_offset:
        start_col = int(offset * scale)
        width = max(1, int(size * scale))
        bar = " " * start_col + "█" * width
        print(f"{name:20s} |{bar}")

    print(f"{'':20s} |" + "-" * 60)
    print(f"{'':20s} |0" + " " * 28 + f"{max_offset/1024:.0f}KB")


if __name__ == "__main__":
    main()
