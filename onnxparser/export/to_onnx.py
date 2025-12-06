# -*- coding: utf-8 -*-
"""Export FX GraphModule to ONNX format"""

import torch
import torch.fx as fx
from typing import Dict, List, Optional, Tuple, Union
import os


def export_onnx(
    gm: fx.GraphModule,
    path: str,
    input_shapes: Dict[str, List[int]],
    opset_version: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> None:
    """Export FX GraphModule to ONNX file

    Args:
        gm: FX GraphModule to export
        path: Output ONNX file path
        input_shapes: Dict mapping input names to shapes
        opset_version: ONNX opset version (default: 17)
        dynamic_axes: Optional dynamic axes specification
    """
    # Create dummy inputs based on input_shapes
    dummy_inputs = []
    input_names = []

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            name = node.name
            input_names.append(name)
            if name in input_shapes:
                shape = input_shapes[name]
            elif node.name in input_shapes:
                shape = input_shapes[node.name]
            else:
                # Try to get from meta
                shape = node.meta.get("shape", [1, 64])

            dummy_inputs.append(torch.randn(*shape))

    # Get output names
    output_names = []
    for node in gm.graph.nodes:
        if node.op == "output":
            args = node.args[0]
            if isinstance(args, (list, tuple)):
                for i, arg in enumerate(args):
                    output_names.append(f"output_{i}")
            else:
                output_names.append("output")

    # Export using torch.onnx.export
    if len(dummy_inputs) == 1:
        dummy_inputs = dummy_inputs[0]
    else:
        dummy_inputs = tuple(dummy_inputs)

    torch.onnx.export(
        gm,
        dummy_inputs,
        path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )


def export_onnx_multi_version(
    gm: fx.GraphModule,
    output_dir: str,
    input_shapes: Dict[str, List[int]],
    opset_versions: List[int] = None,
    model_name: str = "model",
) -> Dict[int, str]:
    """Export FX GraphModule to multiple ONNX opset versions

    Args:
        gm: FX GraphModule to export
        output_dir: Output directory for ONNX files
        input_shapes: Dict mapping input names to shapes
        opset_versions: List of opset versions to export (default: [11, 13, 14, 17, 18, 21])
        model_name: Base name for output files

    Returns:
        Dict mapping opset version to output file path
    """
    if opset_versions is None:
        opset_versions = [11, 13, 14, 17, 18, 21]

    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for opset in opset_versions:
        output_path = os.path.join(output_dir, f"{model_name}_opset{opset}.onnx")
        try:
            export_onnx(gm, output_path, input_shapes, opset_version=opset)
            results[opset] = output_path
            print(f"  Exported opset {opset}: {output_path}")
        except Exception as e:
            print(f"  Failed opset {opset}: {e}")
            results[opset] = None

    return results
