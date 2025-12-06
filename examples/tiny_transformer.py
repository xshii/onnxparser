# -*- coding: utf-8 -*-
"""Tiny Transformer model using onnxparser GraphBuilder

A minimal transformer with:
- Input dimension: 28 * 68 * 119
- Embedding dimension: 64
- 2 attention layers
- 4 attention heads
"""

import torch
from onnxparser.builder import GraphBuilder
from onnxparser.executor import Runtime
from onnxparser.export import export_excel, export_onnx_multi_version


def build_tiny_transformer(
    input_dim: tuple = (28, 68, 119),  # Input: 28 * 68 * 119
    vocab_size: int = 1000,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 256,
) -> torch.fx.GraphModule:
    """Build a tiny transformer model

    Args:
        input_dim: Input dimension (28, 68, 119)
        vocab_size: Vocabulary size for output
        d_model: Embedding dimension (64)
        n_heads: Number of attention heads (4)
        n_layers: Number of transformer layers (2)
        d_ff: Feed-forward hidden dimension
    """
    builder = GraphBuilder("tiny_transformer")

    # Input: [28, 68, 119]
    x = builder.input("input", [input_dim[0], input_dim[1], input_dim[2]])

    # Embedding layer: project last dim from 119 to d_model (64)
    # [28, 68, 119] -> [28, 68, 64]
    x = builder.linear(x, d_model, name="embedding")

    # Transformer layers
    for layer_idx in range(n_layers):
        prefix = f"layer{layer_idx}"

        # ============ Self-Attention ============
        # Pre-LayerNorm
        attn_ln = builder.layer_norm(x, [d_model])

        # Q, K, V projections
        q = builder.linear(attn_ln, d_model, name=f"{prefix}_q_proj")
        k = builder.linear(attn_ln, d_model, name=f"{prefix}_k_proj")
        v = builder.linear(attn_ln, d_model, name=f"{prefix}_v_proj")

        # Scaled dot-product attention
        attn_out = builder.attention(q, k, v)

        # Output projection
        attn_out = builder.linear(attn_out, d_model, name=f"{prefix}_out_proj")

        # Residual connection
        x = x + attn_out

        # ============ Feed-Forward Network ============
        # Pre-LayerNorm
        ff_ln = builder.layer_norm(x, [d_model])

        # FFN: Linear -> GELU -> Linear
        ff = builder.linear(ff_ln, d_ff, name=f"{prefix}_ff1")
        ff = builder.gelu(ff)
        ff = builder.linear(ff, d_model, name=f"{prefix}_ff2")

        # Residual connection
        x = x + ff

    # Final LayerNorm
    x = builder.layer_norm(x, [d_model])

    # Output projection (e.g., for classification or LM head)
    logits = builder.linear(x, vocab_size, name="lm_head")

    builder.output(logits)

    return builder.build()


def main():
    print("Building Tiny Transformer...")
    print("  - Input dim: 28 x 68 x 119")
    print("  - Embedding dim: 64")
    print("  - Attention layers: 2")
    print("  - Attention heads: 4")
    print("  - FFN hidden: 256")
    print()

    # Build model
    gm = build_tiny_transformer()

    # Print graph info
    print("Graph nodes:")
    for i, node in enumerate(gm.graph.nodes):
        if node.op not in ("placeholder", "output", "get_attr"):
            print(f"  [{i:2d}] {node.op:15s} {node.name}")

    # Count parameters
    total_params = 0
    for name, buf in gm.named_buffers():
        total_params += buf.numel()
    print(f"\nTotal parameters: {total_params:,}")

    # Create runtime
    runtime = Runtime(gm).eval()

    # Test inference
    print("\nRunning inference...")
    input_dim = (28, 68, 119)

    # Input: [28, 68, 119]
    x = torch.randn(*input_dim)
    output = runtime.run(x)

    print(f"  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(output.shape)}")

    # Benchmark
    print("\nBenchmarking...")
    result = runtime.benchmark(x, warmup=10, runs=100)
    print(f"  Mean: {result['mean_ms']:.3f} ms")
    print(f"  Min:  {result['min_ms']:.3f} ms")
    print(f"  Max:  {result['max_ms']:.3f} ms")

    # Export to Excel
    excel_path = "models/tiny_transformer.xlsx"
    export_excel(gm, excel_path)
    print(f"\nExported to {excel_path}")

    # Export to multiple ONNX versions
    print("\nExporting to ONNX (multiple opset versions)...")
    input_shapes = {"input": list(input_dim)}
    results = export_onnx_multi_version(
        gm,
        output_dir="models/onnx",
        input_shapes=input_shapes,
        opset_versions=[11, 13, 14, 17, 18, 21],
        model_name="tiny_transformer",
    )

    print(f"\nSuccessfully exported {sum(1 for v in results.values() if v)} ONNX files")


if __name__ == "__main__":
    main()
