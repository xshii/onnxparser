#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: Dynamic visualization server with multiple models.

Architecture:
- Backend (Flask): All algorithms (graph layout, memory analysis)
- Frontend (Static HTML/JS/CSS): Pure rendering and interaction

API Endpoints:
- GET  /api/models              - List all loaded models
- GET  /api/model?name=xxx      - Get graph data with layout
- GET  /api/memory?name=xxx     - Get memory analysis
- GET  /api/strategies          - List memory strategies
- POST /api/layout              - Recompute layout
- DELETE /api/model?name=xxx    - Remove model
"""

import torch
import torch.nn as nn
import torch.fx as fx

from onnxparser.visualizer import serve_dynamic, get_manager


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x


def create_models():
    """Create and trace multiple models"""
    models = {}
    input_data = {}

    # Model 1: Simple MLP
    mlp = SimpleMLP()
    mlp_input = torch.randn(2, 64)
    mlp_gm = fx.symbolic_trace(mlp)
    models["SimpleMLP"] = mlp_gm
    input_data["SimpleMLP"] = {"x": mlp_input}

    # Model 2: Transformer Block
    transformer = SimpleTransformerBlock()
    transformer_input = torch.randn(2, 8, 64)
    transformer_gm = fx.symbolic_trace(transformer)
    models["TransformerBlock"] = transformer_gm
    input_data["TransformerBlock"] = {"x": transformer_input}

    # Model 3: ConvNet
    convnet = ConvNet()
    conv_input = torch.randn(2, 3, 32, 32)
    conv_gm = fx.symbolic_trace(convnet)
    models["ConvNet"] = conv_gm
    input_data["ConvNet"] = {"x": conv_input}

    return models, input_data


def main():
    print("Creating example models...")
    models, input_data = create_models()

    print(f"Loaded {len(models)} models: {list(models.keys())}")
    print()
    print("Starting Flask visualization server...")
    print()
    print("Frontend: Static HTML/JS/CSS (pure rendering)")
    print("Backend:  Flask REST API (algorithms + data)")
    print()

    serve_dynamic(
        models=models,
        input_data=input_data,
        port=8080,
        open_browser=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
