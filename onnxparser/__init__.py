# -*- coding: utf-8 -*-
"""
onnxparser - ONNX model builder, parser, executor and optimizer

Usage:
    # Build model
    from onnxparser.builder import GraphBuilder
    builder = GraphBuilder("my_model")
    x = builder.input("x", [1, 64])
    x = builder.linear(x, 128)
    x = builder.relu(x)
    builder.output(x)
    gm = builder.build()

    # Load ONNX model
    from onnxparser.parser import load
    gm = load("model.onnx")

    # Export to Excel
    from onnxparser.export import export_excel
    export_excel(gm, "model.xlsx")

    # Run inference
    from onnxparser.executor import Runtime
    runtime = Runtime(gm)
    output = runtime.run(input_tensor)

    # Memory analysis
    from onnxparser.analysis import MemoryAnalyzer
    analyzer = MemoryAnalyzer(gm)
    result = analyzer.analyze()

    # Memory visualization
    from onnxparser.visualizer import visualize_memory
    visualize_memory(gm, "memory.html")
"""

__version__ = "0.1.0"

# Convenience imports
from onnxparser.parser import load
from onnxparser.export import export_excel
from onnxparser.executor import Runtime
from onnxparser.builder import GraphBuilder

__all__ = [
    "load",
    "export_excel",
    "Runtime",
    "GraphBuilder",
]
