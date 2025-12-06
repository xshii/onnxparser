# -*- coding: utf-8 -*-
"""ONNX to FX Graph converter"""

import torch
import torch.fx as fx
import torch.nn.functional as F
import onnx
from onnx import numpy_helper
from typing import Dict, List, Any, Callable

from onnxparser.core.ops import ONNX_TO_TORCH, is_special_op
from onnxparser.core.dtypes import ONNX_TO_TORCH_DTYPE


class ONNXToFX:
    """Convert ONNX model to FX GraphModule"""

    def __init__(self):
        self.graph = fx.Graph()
        self.nodes: Dict[str, fx.Node] = {}
        self.initializers: Dict[str, torch.Tensor] = {}
        self.name_map: Dict[str, str] = {}  # onnx_name -> safe_name
        self.opset_version: int = 11

    def _safe_name(self, name: str) -> str:
        """Convert ONNX name to valid Python identifier"""
        safe = name.replace(".", "_").replace("/", "_").replace(":", "_")
        if safe in self.name_map.values():
            i = 1
            while f"{safe}_{i}" in self.name_map.values():
                i += 1
            safe = f"{safe}_{i}"
        self.name_map[name] = safe
        return safe

    def _get_safe_name(self, name: str) -> str:
        """Get existing safe name or create new one"""
        if name in self.name_map:
            return self.name_map[name]
        return self._safe_name(name)

    def load(self, path: str) -> fx.GraphModule:
        """Load ONNX model and convert to FX GraphModule"""
        model = onnx.load(path)
        return self.convert(model)

    def convert(self, model: onnx.ModelProto) -> fx.GraphModule:
        """Convert ONNX ModelProto to FX GraphModule"""
        # Get opset version
        if model.opset_import:
            self.opset_version = model.opset_import[0].version

        graph = model.graph

        # Load initializers (weights)
        for init in graph.initializer:
            tensor = numpy_helper.to_array(init)
            safe_name = self._safe_name(init.name)
            self.initializers[safe_name] = torch.from_numpy(tensor.copy())

        # Create input placeholders
        for inp in graph.input:
            safe_name = self._get_safe_name(inp.name)
            if safe_name not in self.initializers:
                node = self.graph.placeholder(safe_name)
                self.nodes[inp.name] = node

        # Convert each node
        for node in graph.node:
            self._convert_node(node)

        # Set output
        output_nodes = []
        for out in graph.output:
            if out.name in self.nodes:
                output_nodes.append(self.nodes[out.name])

        if len(output_nodes) == 1:
            self.graph.output(output_nodes[0])
        else:
            self.graph.output(tuple(output_nodes))

        # Build GraphModule
        return self._build_module()

    def _convert_node(self, node: onnx.NodeProto) -> None:
        """Convert a single ONNX node to FX node"""
        op_type = node.op_type
        inputs = [self._get_input(name) for name in node.input if name]
        attrs = self._get_attributes(node)

        # Handle special ops
        if is_special_op(op_type):
            output = self._convert_special_op(op_type, inputs, attrs, node)
        elif op_type in ONNX_TO_TORCH:
            torch_op = ONNX_TO_TORCH[op_type]
            if torch_op is not None:
                output = self._call_function(torch_op, inputs, attrs, op_type)
            else:
                output = self._convert_special_op(op_type, inputs, attrs, node)
        else:
            # Unknown op - create placeholder
            output = self._create_unknown_op(op_type, inputs, node)

        # Register outputs
        for i, out_name in enumerate(node.output):
            if out_name:
                self.nodes[out_name] = output

    def _get_input(self, name: str) -> fx.Node:
        """Get input node by name"""
        if name in self.nodes:
            return self.nodes[name]
        safe_name = self._get_safe_name(name)
        if safe_name in self.initializers:
            # Create get_attr for initializer
            node = self.graph.get_attr(safe_name)
            self.nodes[name] = node
            return node
        else:
            raise ValueError(f"Unknown input: {name}")

    def _get_attributes(self, node: onnx.NodeProto) -> Dict[str, Any]:
        """Extract attributes from ONNX node"""
        attrs = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.STRING:
                attrs[attr.name] = attr.s.decode()
            elif attr.type == onnx.AttributeProto.FLOATS:
                attrs[attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.TENSOR:
                attrs[attr.name] = numpy_helper.to_array(attr.t)
        return attrs

    def _call_function(
        self,
        func: Callable,
        inputs: List[fx.Node],
        attrs: Dict[str, Any],
        op_name: str,
    ) -> fx.Node:
        """Create a call_function node"""
        # Simple case: pass all inputs as args
        node = self.graph.call_function(func, tuple(inputs))
        return node

    def _convert_special_op(
        self,
        op_type: str,
        inputs: List[fx.Node],
        attrs: Dict[str, Any],
        onnx_node: onnx.NodeProto,
    ) -> fx.Node:
        """Handle special ops that need custom conversion"""

        if op_type == "Conv":
            return self._convert_conv(inputs, attrs)
        elif op_type == "Gemm":
            return self._convert_gemm(inputs, attrs)
        elif op_type == "Softmax":
            return self._convert_softmax(inputs, attrs)
        elif op_type == "MaxPool":
            return self._convert_maxpool(inputs, attrs)
        elif op_type == "AveragePool":
            return self._convert_avgpool(inputs, attrs)
        elif op_type == "Slice":
            return self._convert_slice(inputs, attrs, onnx_node)
        elif op_type == "Cast":
            return self._convert_cast(inputs, attrs)
        elif op_type == "Reshape":
            return self._convert_reshape(inputs, attrs)
        elif op_type == "Transpose":
            return self._convert_transpose(inputs, attrs)
        else:
            # Fallback
            return self._create_unknown_op(op_type, inputs, onnx_node)

    def _convert_conv(self, inputs: List[fx.Node], attrs: Dict) -> fx.Node:
        """Convert Conv op"""
        x, weight = inputs[0], inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None

        kwargs = {}
        if "strides" in attrs:
            kwargs["stride"] = tuple(attrs["strides"])
        if "pads" in attrs:
            pads = attrs["pads"]
            # ONNX format: [begin1, begin2, end1, end2]
            # PyTorch format: [pad1, pad2]
            kwargs["padding"] = (pads[0], pads[1]) if len(pads) >= 2 else pads[0]
        if "dilations" in attrs:
            kwargs["dilation"] = tuple(attrs["dilations"])
        if "group" in attrs:
            kwargs["groups"] = attrs["group"]

        args = [x, weight]
        if bias is not None:
            args.append(bias)

        return self.graph.call_function(F.conv2d, tuple(args), kwargs)

    def _convert_gemm(self, inputs: List[fx.Node], attrs: Dict) -> fx.Node:
        """Convert Gemm op (Y = alpha * A @ B + beta * C)"""
        a, b = inputs[0], inputs[1]
        c = inputs[2] if len(inputs) > 2 else None

        alpha = attrs.get("alpha", 1.0)
        beta = attrs.get("beta", 1.0)
        trans_a = attrs.get("transA", 0)
        trans_b = attrs.get("transB", 0)

        if trans_a:
            a = self.graph.call_function(torch.transpose, (a, -2, -1))
        if trans_b:
            b = self.graph.call_function(torch.transpose, (b, -2, -1))

        result = self.graph.call_function(torch.matmul, (a, b))

        if alpha != 1.0:
            result = self.graph.call_function(torch.mul, (result, alpha))

        if c is not None and beta != 0.0:
            if beta != 1.0:
                c = self.graph.call_function(torch.mul, (c, beta))
            result = self.graph.call_function(torch.add, (result, c))

        return result

    def _convert_softmax(self, inputs: List[fx.Node], attrs: Dict) -> fx.Node:
        """Convert Softmax op"""
        axis = attrs.get("axis", -1)
        return self.graph.call_function(F.softmax, (inputs[0],), {"dim": axis})

    def _convert_maxpool(self, inputs: List[fx.Node], attrs: Dict) -> fx.Node:
        """Convert MaxPool op"""
        kernel_shape = tuple(attrs.get("kernel_shape", [2, 2]))
        strides = tuple(attrs.get("strides", kernel_shape))
        pads = attrs.get("pads", [0, 0])
        padding = (pads[0], pads[1]) if len(pads) >= 2 else 0

        return self.graph.call_function(
            F.max_pool2d,
            (inputs[0],),
            {"kernel_size": kernel_shape, "stride": strides, "padding": padding}
        )

    def _convert_avgpool(self, inputs: List[fx.Node], attrs: Dict) -> fx.Node:
        """Convert AveragePool op"""
        kernel_shape = tuple(attrs.get("kernel_shape", [2, 2]))
        strides = tuple(attrs.get("strides", kernel_shape))
        pads = attrs.get("pads", [0, 0])
        padding = (pads[0], pads[1]) if len(pads) >= 2 else 0

        return self.graph.call_function(
            F.avg_pool2d,
            (inputs[0],),
            {"kernel_size": kernel_shape, "stride": strides, "padding": padding}
        )

    def _convert_slice(self, inputs: List[fx.Node], attrs: Dict, node: onnx.NodeProto) -> fx.Node:
        """Convert Slice op"""
        # For older opset, attributes are used
        # For newer opset, inputs are: data, starts, ends, [axes], [steps]
        # Simplified: just pass through for now
        return inputs[0]

    def _convert_cast(self, inputs: List[fx.Node], attrs: Dict) -> fx.Node:
        """Convert Cast op"""
        to_type = attrs.get("to", 1)
        torch_dtype = ONNX_TO_TORCH_DTYPE.get(to_type, torch.float32)

        def cast_fn(x, dtype=torch_dtype):
            return x.to(dtype)

        return self.graph.call_function(cast_fn, (inputs[0],))

    def _convert_reshape(self, inputs: List[fx.Node], attrs: Dict) -> fx.Node:
        """Convert Reshape op"""
        # In newer opset, shape is an input, not attribute
        if len(inputs) > 1:
            # Dynamic shape from input
            return self.graph.call_function(torch.reshape, (inputs[0], inputs[1]))
        else:
            # Static shape from attribute
            shape = attrs.get("shape", [-1])
            return self.graph.call_function(torch.reshape, (inputs[0], shape))

    def _convert_transpose(self, inputs: List[fx.Node], attrs: Dict) -> fx.Node:
        """Convert Transpose op"""
        perm = attrs.get("perm", None)
        if perm:
            return self.graph.call_function(torch.permute, (inputs[0], perm))
        else:
            return self.graph.call_function(torch.transpose, (inputs[0], -2, -1))

    def _create_unknown_op(
        self,
        op_type: str,
        inputs: List[fx.Node],
        node: onnx.NodeProto,
    ) -> fx.Node:
        """Create placeholder for unknown op"""
        def unknown_op(*args, op=op_type):
            raise NotImplementedError(f"Unknown op: {op}")

        return self.graph.call_function(unknown_op, tuple(inputs))

    def _build_module(self) -> fx.GraphModule:
        """Build the final GraphModule"""

        class ONNXModule(torch.nn.Module):
            pass

        module = ONNXModule()

        # Register initializers as buffers
        for name, tensor in self.initializers.items():
            module.register_buffer(name, tensor)

        return fx.GraphModule(module, self.graph)


def load(path: str) -> fx.GraphModule:
    """Load ONNX model and convert to FX GraphModule"""
    converter = ONNXToFX()
    return converter.load(path)
