#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example: Memory spill/reload scheduling"""

import sys
sys.path.insert(0, "/home/user/onnxparser")

from onnxparser.builder import GraphBuilder
from onnxparser.analysis import (
    MemoryAnalyzer,
    SpillScheduler,
    SpillStrategy,
    schedule_spills,
    MemoryEventType,
)


def create_sample_model():
    """Create a sample model for demonstration"""
    import torch

    builder = GraphBuilder("sample_model")

    # Input: [1, 128, 64] = 32KB
    x = builder.input("input", [1, 128, 64], dtype="float32")

    # Q, K, V projections
    q_weight = builder.constant("q_weight", torch.randn(64, 64), dtype="float32")
    k_weight = builder.constant("k_weight", torch.randn(64, 64), dtype="float32")
    v_weight = builder.constant("v_weight", torch.randn(64, 64), dtype="float32")

    q = builder.matmul(x, q_weight)  # 32KB
    k = builder.matmul(x, k_weight)  # 32KB
    v = builder.matmul(x, v_weight)  # 32KB

    # Attention
    k_t = builder.transpose(k, -2, -1)
    scores = builder.matmul(q, k_t)  # [1, 128, 128] = 64KB
    scores = builder.softmax(scores, dim=-1)
    attn_out = builder.matmul(scores, v)

    # Residual
    x = builder.add(x, attn_out)

    # Layer norm
    ln_weight = builder.constant("ln_weight", torch.randn(64), dtype="float32")
    ln_bias = builder.constant("ln_bias", torch.randn(64), dtype="float32")
    x = builder.layer_norm(x, [64], ln_weight, ln_bias)

    # FFN
    ffn_w1 = builder.constant("ffn_w1", torch.randn(64, 256), dtype="float32")
    ffn_w2 = builder.constant("ffn_w2", torch.randn(256, 64), dtype="float32")

    ffn = builder.matmul(x, ffn_w1)  # [1, 128, 256] = 128KB
    ffn = builder.gelu(ffn)
    ffn = builder.matmul(ffn, ffn_w2)

    x = builder.add(x, ffn)
    builder.output(x)

    return builder.build()


def main():
    print("Creating sample model...")
    gm = create_sample_model()
    print(f"Model created with {len(list(gm.graph.nodes))} nodes")

    # First, run baseline memory analysis
    print("\n" + "="*60)
    print("BASELINE MEMORY ANALYSIS (No constraint)")
    print("="*60)
    analyzer = MemoryAnalyzer(gm, strategy="greedy")
    baseline = analyzer.analyze()
    print(f"Peak Memory: {baseline.peak_min_memory / 1024:.1f} KB")

    # Set a tight memory limit (e.g., 100KB - less than peak)
    memory_limit_kb = 100
    print(f"\n" + "="*60)
    print(f"SPILL SCHEDULING (Memory Limit: {memory_limit_kb} KB)")
    print("="*60)

    # Schedule with different strategies
    strategies = ["lru", "size_first", "lifetime", "cost_benefit"]

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy.upper()} ---")
        result = schedule_spills(gm, memory_limit_mb=memory_limit_kb/1024, strategy=strategy)

        print(f"Total Spills: {result.total_spills}")
        print(f"Total Reloads: {result.total_reloads}")
        print(f"Total Spill Data: {result.total_spill_bytes / 1024:.1f} KB")
        print(f"Peak Fast Memory: {result.peak_fast_memory / 1024:.1f} KB")
        print(f"Peak Slow Memory: {result.peak_slow_memory / 1024:.1f} KB")

        if result.spill_trigger_nodes:
            print(f"Spill Trigger Nodes: {result.spill_trigger_nodes}")
        if result.reload_trigger_nodes:
            print(f"Reload Trigger Nodes: {result.reload_trigger_nodes}")

    # Detailed output for cost_benefit strategy
    print(f"\n" + "="*60)
    print("DETAILED SCHEDULE (cost_benefit strategy)")
    print("="*60)

    result = schedule_spills(gm, memory_limit_mb=memory_limit_kb/1024, strategy="cost_benefit")

    # Show spill decisions
    if result.spill_decisions:
        print("\nSpill Decisions:")
        for d in result.spill_decisions:
            print(f"  {d.tensor_name}: {d.size_bytes/1024:.1f}KB, "
                  f"spill@step{d.spill_step} -> reload@step{d.reload_step} "
                  f"(duration: {d.spill_duration} steps)")

    # Show memory events timeline
    print("\nMemory Events:")
    for event in result.events:
        if event.event_type in [MemoryEventType.SPILL, MemoryEventType.RELOAD]:
            print(f"  Step {event.step:2d} [{event.node_name:15s}]: "
                  f"{event.event_type.value:8s} {event.tensor_name:15s} "
                  f"({event.size_bytes/1024:.1f}KB) "
                  f"-> fast:{event.fast_memory_used/1024:.1f}KB, "
                  f"slow:{event.slow_memory_used/1024:.1f}KB")

    # Export to JSON
    print("\n" + "="*60)
    print("EXPORT")
    print("="*60)
    import json
    output_path = "/home/user/onnxparser/examples/spill_schedule.json"
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Schedule exported to: {output_path}")

    # Summary table
    print("\n" + "="*60)
    print("TRIGGER NODE SUMMARY")
    print("="*60)
    print("\nNodes that trigger SPILL (memory pressure):")
    for node in result.spill_trigger_nodes:
        print(f"  - {node}")

    print("\nNodes that trigger RELOAD (need spilled data):")
    for node in result.reload_trigger_nodes:
        print(f"  - {node}")


if __name__ == "__main__":
    main()
