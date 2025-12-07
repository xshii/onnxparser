# -*- coding: utf-8 -*-
"""Example: Visualize computation graph with data flow in web browser"""

import sys
sys.path.insert(0, ".")

import torch
from onnxparser.builder import GraphBuilder
from onnxparser.visualizer import visualize, serve


def build_mlp():
    """Build a simple MLP for visualization"""
    builder = GraphBuilder()

    x = builder.input("input", [1, 64])
    x = builder.linear(x, 128, name="fc1")
    x = builder.relu(x)
    x = builder.linear(x, 256, name="fc2")
    x = builder.relu(x)
    x = builder.linear(x, 128, name="fc3")
    x = builder.relu(x)
    x = builder.linear(x, 10, name="output_fc")
    x = builder.softmax(x, dim=-1)
    builder.output(x)

    return builder.build()


def build_attention():
    """Build a simple single-head attention block for visualization"""
    builder = GraphBuilder()

    # Use batch=4 so we have multiple slices to browse
    x = builder.input("input", [4, 8, 32])  # [batch, seq_len, d_model]
    d_model = 32

    q = builder.linear(x, d_model, name="q_proj")
    k = builder.linear(x, d_model, name="k_proj")
    v = builder.linear(x, d_model, name="v_proj")

    k_t = builder.transpose(k, 1, 2)
    scores = builder.matmul(q, k_t)
    scores = builder.mul(scores, d_model ** -0.5)
    attn_weights = builder.softmax(scores, dim=-1)
    attn_out = builder.matmul(attn_weights, v)

    output = builder.linear(attn_out, d_model, name="out_proj")
    output = builder.add(x, output)
    output = builder.layer_norm(output, d_model)
    builder.output(output)

    return builder.build()


def build_multi_head_attention():
    """Build a Multi-Head Attention block for visualization"""
    builder = GraphBuilder()

    # Parameters
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 4
    d_k = d_model // num_heads  # 16

    # Input: [batch, seq_len, d_model]
    x = builder.input("input", [batch_size, seq_len, d_model])

    # Q, K, V projections: [batch, seq_len, d_model]
    q = builder.linear(x, d_model, name="q_proj")
    k = builder.linear(x, d_model, name="k_proj")
    v = builder.linear(x, d_model, name="v_proj")

    # Reshape for multi-head: [batch, seq_len, num_heads, d_k]
    q = builder.reshape(q, [batch_size, seq_len, num_heads, d_k])
    k = builder.reshape(k, [batch_size, seq_len, num_heads, d_k])
    v = builder.reshape(v, [batch_size, seq_len, num_heads, d_k])

    # Transpose to [batch, num_heads, seq_len, d_k]
    q = builder.transpose(q, 1, 2)
    k = builder.transpose(k, 1, 2)
    v = builder.transpose(v, 1, 2)

    # Attention scores: [batch, num_heads, seq_len, seq_len]
    k_t = builder.transpose(k, 2, 3)  # [batch, num_heads, d_k, seq_len]
    scores = builder.matmul(q, k_t)
    scores = builder.mul(scores, d_k ** -0.5)  # scale

    # Softmax attention weights
    attn_weights = builder.softmax(scores, dim=-1)

    # Apply attention to values: [batch, num_heads, seq_len, d_k]
    attn_out = builder.matmul(attn_weights, v)

    # Transpose back: [batch, seq_len, num_heads, d_k]
    attn_out = builder.transpose(attn_out, 1, 2)

    # Reshape to [batch, seq_len, d_model]
    attn_out = builder.reshape(attn_out, [batch_size, seq_len, d_model])

    # Output projection
    output = builder.linear(attn_out, d_model, name="out_proj")

    # Residual connection + LayerNorm
    output = builder.add(x, output)
    output = builder.layer_norm(output, d_model)

    builder.output(output)
    return builder.build()


def build_transformer():
    """Build a complete Transformer with multiple layers and causal mask"""
    builder = GraphBuilder()

    # Parameters
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 4
    d_ff = 256  # FFN hidden dim
    num_layers = 3
    d_k = d_model // num_heads

    # Input: [batch, seq_len, d_model]
    x = builder.input("input", [batch_size, seq_len, d_model])

    # Create causal mask: upper triangular with -inf
    # Shape: [1, 1, seq_len, seq_len] for broadcasting
    mask_data = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    mask_data = mask_data.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    causal_mask = builder.constant("causal_mask", mask_data)

    for layer_idx in range(num_layers):
        prefix = f"layer{layer_idx}"

        # ===== Multi-Head Self-Attention =====
        residual = x

        # Q, K, V projections
        q = builder.linear(x, d_model, name=f"{prefix}_q")
        k = builder.linear(x, d_model, name=f"{prefix}_k")
        v = builder.linear(x, d_model, name=f"{prefix}_v")

        # Reshape: [batch, seq, d_model] -> [batch, seq, heads, d_k]
        q = builder.reshape(q, [batch_size, seq_len, num_heads, d_k])
        k = builder.reshape(k, [batch_size, seq_len, num_heads, d_k])
        v = builder.reshape(v, [batch_size, seq_len, num_heads, d_k])

        # Transpose: [batch, heads, seq, d_k]
        q = builder.transpose(q, 1, 2)
        k = builder.transpose(k, 1, 2)
        v = builder.transpose(v, 1, 2)

        # Attention: Q @ K^T / sqrt(d_k)
        k_t = builder.transpose(k, 2, 3)
        scores = builder.matmul(q, k_t)
        scores = builder.mul(scores, d_k ** -0.5)

        # Apply causal mask (add -inf to future positions)
        scores = builder.add(scores, causal_mask)

        # Softmax (masked positions become 0)
        attn_weights = builder.softmax(scores, dim=-1)

        # Apply attention to V
        attn_out = builder.matmul(attn_weights, v)

        # Transpose back and reshape
        attn_out = builder.transpose(attn_out, 1, 2)
        attn_out = builder.reshape(attn_out, [batch_size, seq_len, d_model])

        # Output projection
        attn_out = builder.linear(attn_out, d_model, name=f"{prefix}_attn_out")

        # Add & Norm
        x = builder.add(residual, attn_out)
        x = builder.layer_norm(x, d_model)

        # ===== Feed-Forward Network =====
        residual = x

        # FFN: Linear -> GELU -> Linear
        ffn = builder.linear(x, d_ff, name=f"{prefix}_ffn1")
        ffn = builder.gelu(ffn)
        ffn = builder.linear(ffn, d_model, name=f"{prefix}_ffn2")

        # Add & Norm
        x = builder.add(residual, ffn)
        x = builder.layer_norm(x, d_model)

    # Final output projection
    output = builder.linear(x, d_model, name="final_proj")
    builder.output(output)

    return builder.build()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize computation graph with data flow")
    parser.add_argument("--model", choices=["mlp", "attention", "mha", "transformer"], default="transformer",
                        help="Model to visualize (default: transformer)")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--save", type=str, help="Save to HTML file instead of serving")
    parser.add_argument("--no-data", action="store_true", help="Disable data flow visualization")
    args = parser.parse_args()

    if args.model == "mlp":
        print("Building MLP model...")
        gm = build_mlp()
        input_data = {"input": torch.randn(1, 64)}
    elif args.model == "attention":
        print("Building Single-Head Attention model...")
        gm = build_attention()
        input_data = {"input": torch.randn(4, 8, 32)}
    elif args.model == "mha":
        print("Building Multi-Head Attention model...")
        gm = build_multi_head_attention()
        input_data = {"input": torch.randn(2, 8, 64)}
    else:  # transformer
        print("Building Transformer (3 layers) model...")
        gm = build_transformer()
        input_data = {"input": torch.randn(2, 8, 64)}

    print(f"Graph has {len(list(gm.graph.nodes))} nodes")

    if args.no_data:
        input_data = None
        print("Data flow visualization disabled")
    else:
        print("Tracing data flow through the graph...")

    if args.save:
        print(f"\nSaving visualization to {args.save}...")
        visualize(gm, args.save, input_data=input_data)
        print("Done!")
    else:
        print(f"\nStarting visualization server on port {args.port}...")
        print(f"Open http://localhost:{args.port} in your browser")
        print("Press Ctrl+C to stop\n")
        serve(gm, port=args.port, input_data=input_data)


if __name__ == "__main__":
    main()
