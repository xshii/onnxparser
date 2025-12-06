# -*- coding: utf-8 -*-
"""FX Graph Builder - Build ONNX models with Python code"""

import torch
import torch.fx as fx
import torch.nn.functional as F
from typing import List, Dict, Union, Optional, Any, Tuple
import numpy as np

from onnxparser.core.dtypes import to_torch_dtype


class Tensor:
    """Tensor wrapper with operator overloading support"""

    def __init__(
        self,
        node: fx.Node,
        builder: "GraphBuilder",
        shape: Optional[List[Union[int, str]]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self._node = node
        self._builder = builder
        self._shape = shape or []
        self._dtype = dtype

    @property
    def name(self) -> str:
        return self._node.name

    @property
    def shape(self) -> List[Union[int, str]]:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._builder.add(self, other)

    def __radd__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._builder.add(other, self)

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._builder.sub(self, other)

    def __rsub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._builder.sub(other, self)

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._builder.mul(self, other)

    def __rmul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._builder.mul(other, self)

    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self._builder.div(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self._builder.matmul(self, other)

    def __neg__(self) -> "Tensor":
        return self._builder.neg(self)

    def __repr__(self) -> str:
        return f"Tensor(name={self.name}, shape={self._shape}, dtype={self._dtype})"


class GraphBuilder:
    """FX Graph Builder"""

    def __init__(self, name: str = "model"):
        self.name = name
        self.graph = fx.Graph()
        self._tensors: Dict[str, Tensor] = {}
        self._counter: Dict[str, int] = {}
        self._inputs: List[str] = []
        self._outputs: List[str] = []
        self._constants: Dict[str, torch.Tensor] = {}

    def _make_name(self, prefix: str) -> str:
        """Generate unique name"""
        count = self._counter.get(prefix, 0)
        self._counter[prefix] = count + 1
        return f"{prefix}_{count}" if count > 0 else prefix

    def _get_node(self, x: Union[Tensor, float, int, torch.Tensor]) -> fx.Node:
        """Get FX Node"""
        if isinstance(x, Tensor):
            return x._node
        elif isinstance(x, (float, int)):
            name = self._make_name("const")
            node = self.graph.call_function(lambda v: v, (x,))
            node.name = name
            return node
        elif isinstance(x, torch.Tensor):
            name = self._make_name("const")
            self._constants[name] = x
            node = self.graph.get_attr(name)
            return node
        else:
            raise TypeError(f"Unsupported type: {type(x)}")

    def _wrap_tensor(
        self,
        node: fx.Node,
        shape: Optional[List[Union[int, str]]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Wrap as Tensor"""
        tensor = Tensor(node, self, shape, dtype)
        self._tensors[node.name] = tensor
        return tensor

    # ============ Input/Output ============

    def input(
        self,
        name: str,
        shape: List[Union[int, str]],
        dtype: Union[str, torch.dtype] = "float32",
    ) -> Tensor:
        """Add input"""
        dtype = to_torch_dtype(dtype)
        node = self.graph.placeholder(name)
        node.meta["shape"] = shape
        node.meta["dtype"] = dtype
        self._inputs.append(name)
        return self._wrap_tensor(node, shape, dtype)

    def output(self, *tensors: Tensor) -> None:
        """Set output"""
        nodes = [t._node for t in tensors]
        if len(nodes) == 1:
            self.graph.output(nodes[0])
        else:
            self.graph.output(tuple(nodes))
        self._outputs = [t.name for t in tensors]

    def constant(
        self,
        name: str,
        data: Union[np.ndarray, torch.Tensor, List],
        dtype: Union[str, torch.dtype] = "float32",
    ) -> Tensor:
        """Add constant/weight"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        elif isinstance(data, list):
            data = torch.tensor(data)

        dtype = to_torch_dtype(dtype)
        data = data.to(dtype)
        self._constants[name] = data

        node = self.graph.get_attr(name)
        node.meta["shape"] = list(data.shape)
        node.meta["dtype"] = dtype
        return self._wrap_tensor(node, list(data.shape), dtype)

    # ============ Basic Math ============

    def add(self, a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
        """Add"""
        node_a, node_b = self._get_node(a), self._get_node(b)
        node = self.graph.call_function(torch.add, (node_a, node_b))
        node.name = self._make_name("add")
        shape = a._shape if isinstance(a, Tensor) else (b._shape if isinstance(b, Tensor) else [])
        return self._wrap_tensor(node, shape)

    def sub(self, a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
        """Subtract"""
        node_a, node_b = self._get_node(a), self._get_node(b)
        node = self.graph.call_function(torch.sub, (node_a, node_b))
        node.name = self._make_name("sub")
        shape = a._shape if isinstance(a, Tensor) else (b._shape if isinstance(b, Tensor) else [])
        return self._wrap_tensor(node, shape)

    def mul(self, a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
        """Multiply"""
        node_a, node_b = self._get_node(a), self._get_node(b)
        node = self.graph.call_function(torch.mul, (node_a, node_b))
        node.name = self._make_name("mul")
        shape = a._shape if isinstance(a, Tensor) else (b._shape if isinstance(b, Tensor) else [])
        return self._wrap_tensor(node, shape)

    def div(self, a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
        """Divide"""
        node_a, node_b = self._get_node(a), self._get_node(b)
        node = self.graph.call_function(torch.div, (node_a, node_b))
        node.name = self._make_name("div")
        shape = a._shape if isinstance(a, Tensor) else (b._shape if isinstance(b, Tensor) else [])
        return self._wrap_tensor(node, shape)

    def neg(self, x: Tensor) -> Tensor:
        """Negate"""
        node = self.graph.call_function(torch.neg, (x._node,))
        node.name = self._make_name("neg")
        return self._wrap_tensor(node, x._shape, x._dtype)

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        node = self.graph.call_function(torch.matmul, (a._node, b._node))
        node.name = self._make_name("matmul")
        if len(a._shape) >= 2 and len(b._shape) >= 2:
            shape = list(a._shape[:-1]) + [b._shape[-1]]
        else:
            shape = []
        return self._wrap_tensor(node, shape)

    def sqrt(self, x: Tensor) -> Tensor:
        """Square root"""
        node = self.graph.call_function(torch.sqrt, (x._node,))
        node.name = self._make_name("sqrt")
        return self._wrap_tensor(node, x._shape, x._dtype)

    def pow(self, x: Tensor, exp: float) -> Tensor:
        """Power"""
        node = self.graph.call_function(torch.pow, (x._node, exp))
        node.name = self._make_name("pow")
        return self._wrap_tensor(node, x._shape, x._dtype)

    # ============ Activations ============

    def relu(self, x: Tensor) -> Tensor:
        """ReLU"""
        node = self.graph.call_function(torch.relu, (x._node,))
        node.name = self._make_name("relu")
        return self._wrap_tensor(node, x._shape, x._dtype)

    def gelu(self, x: Tensor) -> Tensor:
        """GELU"""
        node = self.graph.call_function(F.gelu, (x._node,))
        node.name = self._make_name("gelu")
        return self._wrap_tensor(node, x._shape, x._dtype)

    def sigmoid(self, x: Tensor) -> Tensor:
        """Sigmoid"""
        node = self.graph.call_function(torch.sigmoid, (x._node,))
        node.name = self._make_name("sigmoid")
        return self._wrap_tensor(node, x._shape, x._dtype)

    def tanh(self, x: Tensor) -> Tensor:
        """Tanh"""
        node = self.graph.call_function(torch.tanh, (x._node,))
        node.name = self._make_name("tanh")
        return self._wrap_tensor(node, x._shape, x._dtype)

    def softmax(self, x: Tensor, dim: int = -1) -> Tensor:
        """Softmax"""
        node = self.graph.call_function(F.softmax, (x._node,), {"dim": dim})
        node.name = self._make_name("softmax")
        return self._wrap_tensor(node, x._shape, x._dtype)

    # ============ Normalization ============

    def layer_norm(
        self,
        x: Tensor,
        normalized_shape: List[int],
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        eps: float = 1e-5,
    ) -> Tensor:
        """Layer Normalization"""
        args = [x._node, normalized_shape]
        kwargs = {"eps": eps}
        if weight is not None:
            kwargs["weight"] = weight._node
        if bias is not None:
            kwargs["bias"] = bias._node
        node = self.graph.call_function(F.layer_norm, tuple(args), kwargs)
        node.name = self._make_name("layernorm")
        return self._wrap_tensor(node, x._shape, x._dtype)

    # ============ Shape Ops ============

    def reshape(self, x: Tensor, shape: List[int]) -> Tensor:
        """Reshape"""
        node = self.graph.call_function(torch.reshape, (x._node, shape))
        node.name = self._make_name("reshape")
        return self._wrap_tensor(node, shape, x._dtype)

    def transpose(self, x: Tensor, dim0: int, dim1: int) -> Tensor:
        """Transpose"""
        node = self.graph.call_function(torch.transpose, (x._node, dim0, dim1))
        node.name = self._make_name("transpose")
        new_shape = list(x._shape)
        if len(new_shape) > max(dim0, dim1):
            new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        return self._wrap_tensor(node, new_shape, x._dtype)

    def concat(self, tensors: List[Tensor], dim: int = 0) -> Tensor:
        """Concat"""
        nodes = [t._node for t in tensors]
        node = self.graph.call_function(torch.cat, (nodes, dim))
        node.name = self._make_name("concat")
        return self._wrap_tensor(node, [], tensors[0]._dtype if tensors else torch.float32)

    def split(self, x: Tensor, split_size: int, dim: int = 0) -> List[Tensor]:
        """Split"""
        node = self.graph.call_function(torch.split, (x._node, split_size, dim))
        node.name = self._make_name("split")
        return [self._wrap_tensor(node, [], x._dtype)]

    # ============ Linear ============

    def linear(
        self,
        x: Tensor,
        out_features: int,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        name: Optional[str] = None,
    ) -> Tensor:
        """Linear layer (y = xW^T + b)"""
        if weight is None:
            in_features = x._shape[-1] if x._shape else 0
            w_name = name + "_weight" if name else self._make_name("linear_weight")
            w_data = torch.randn(out_features, in_features) * 0.02
            weight = self.constant(w_name, w_data)

        if bias is None:
            b_name = name + "_bias" if name else self._make_name("linear_bias")
            b_data = torch.zeros(out_features)
            bias = self.constant(b_name, b_data)

        node = self.graph.call_function(F.linear, (x._node, weight._node, bias._node))
        node.name = name or self._make_name("linear")

        new_shape = list(x._shape[:-1]) + [out_features] if x._shape else [out_features]
        return self._wrap_tensor(node, new_shape, x._dtype)

    # ============ Transformer ============

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        scale: Optional[float] = None,
    ) -> Tensor:
        """Scaled Dot-Product Attention"""
        kt = self.transpose(key, -2, -1)
        scores = self.matmul(query, kt)

        if scale is None:
            d_k = key._shape[-1] if key._shape else 64
            if isinstance(d_k, int):
                scale = 1.0 / (d_k ** 0.5)
            else:
                scale = 1.0 / 8.0

        scores = self.mul(scores, scale)

        if mask is not None:
            scores = self.add(scores, mask)

        attn_weights = self.softmax(scores, dim=-1)
        output = self.matmul(attn_weights, value)
        return output

    def multi_head_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        d_model: int,
        mask: Optional[Tensor] = None,
        name: Optional[str] = None,
    ) -> Tensor:
        """Multi-Head Attention"""
        prefix = name or "mha"
        d_k = d_model // num_heads

        q = self.linear(query, d_model, name=f"{prefix}_q_proj")
        k = self.linear(key, d_model, name=f"{prefix}_k_proj")
        v = self.linear(value, d_model, name=f"{prefix}_v_proj")

        output = self.linear(q, d_model, name=f"{prefix}_out_proj")
        return output

    # ============ Build ============

    def build(self) -> fx.GraphModule:
        """Build GraphModule"""

        class ConstantModule(torch.nn.Module):
            pass

        module = ConstantModule()

        for name, tensor in self._constants.items():
            module.register_buffer(name, tensor)

        gm = fx.GraphModule(module, self.graph)
        return gm

    def __repr__(self) -> str:
        return f"GraphBuilder(name={self.name}, nodes={len(list(self.graph.nodes))})"
