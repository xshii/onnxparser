# -*- coding: utf-8 -*-
"""
Example: Visualize ONNX model with real input data

Usage:
    # 1. 查看 ONNX 输入信息
    python examples/visualize_onnx.py model.onnx --list-inputs

    # 2. 随机数据可视化
    python examples/visualize_onnx.py model.onnx

    # 3. 指定维度
    python examples/visualize_onnx.py model.onnx --batch 4 --seq-len 128

    # 4. 从 .npz 或 .npy 加载真实数据
    python examples/visualize_onnx.py model.onnx --input data.npz

    # 5. 从 .pt (PyTorch) 加载
    python examples/visualize_onnx.py model.onnx --input data.pt

作为 Python API 使用:
    from examples.visualize_onnx import visualize_onnx

    # 方式1: 自动生成随机输入
    visualize_onnx("model.onnx")

    # 方式2: 传入真实数据
    visualize_onnx("model.onnx", input_data={
        "input_ids": torch.tensor([[101, 2054, 2003, ...]]),
        "attention_mask": torch.tensor([[1, 1, 1, ...]]),
    })
"""

import sys
sys.path.insert(0, ".")

import argparse
import torch
import numpy as np
from pathlib import Path

from onnxparser.parser import onnx_to_fx
from onnxparser.visualizer import serve_dynamic


def get_onnx_input_info(onnx_path: str) -> dict:
    """Extract input information from ONNX model"""
    import onnx
    model = onnx.load(onnx_path)

    inputs = {}
    initializer_names = {init.name for init in model.graph.initializer}

    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue  # Skip weights

        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                # Dynamic dimension, use default value
                shape.append(1 if "batch" in dim.dim_param.lower() else 128)
            else:
                shape.append(1)

        elem_type = inp.type.tensor_type.elem_type
        # ONNX elem_type: 1=float32, 6=int32, 7=int64, 9=bool
        dtype_map = {1: torch.float32, 6: torch.int32, 7: torch.int64, 9: torch.bool}
        dtype = dtype_map.get(elem_type, torch.float32)

        inputs[inp.name] = {"shape": shape, "dtype": dtype}

    return inputs


def generate_sample_input(input_info: dict, mode: str = "random") -> dict:
    """Generate sample input data based on input info

    Args:
        input_info: Dict with shape and dtype for each input
        mode: "random" for random data, "zeros" for zeros, "ones" for ones
    """
    inputs = {}
    for name, info in input_info.items():
        shape = info["shape"]
        dtype = info["dtype"]

        if dtype in (torch.int32, torch.int64):
            # For integer inputs (like token IDs), use random integers
            if mode == "random":
                inputs[name] = torch.randint(0, 1000, shape, dtype=dtype)
            elif mode == "zeros":
                inputs[name] = torch.zeros(shape, dtype=dtype)
            else:
                inputs[name] = torch.ones(shape, dtype=dtype)
        elif dtype == torch.bool:
            # For boolean inputs (like masks)
            if mode == "random":
                inputs[name] = torch.randint(0, 2, shape, dtype=torch.bool)
            else:
                inputs[name] = torch.ones(shape, dtype=torch.bool)
        else:
            # For float inputs
            if mode == "random":
                inputs[name] = torch.randn(shape, dtype=dtype)
            elif mode == "zeros":
                inputs[name] = torch.zeros(shape, dtype=dtype)
            else:
                inputs[name] = torch.ones(shape, dtype=dtype)

        print(f"  {name}: {list(shape)} ({dtype})")

    return inputs


def load_input_from_file(file_path: str) -> dict:
    """Load input data from file (.npz, .npy, .pt)"""
    path = Path(file_path)
    inputs = {}

    if path.suffix == ".npz":
        # NumPy compressed archive (multiple arrays)
        data = np.load(file_path)
        for name in data.files:
            inputs[name] = torch.from_numpy(data[name])
            print(f"  Loaded {name}: {list(inputs[name].shape)} ({inputs[name].dtype})")

    elif path.suffix == ".npy":
        # Single NumPy array
        arr = np.load(file_path)
        inputs["input"] = torch.from_numpy(arr)
        print(f"  Loaded input: {list(inputs['input'].shape)} ({inputs['input'].dtype})")

    elif path.suffix == ".pt":
        # PyTorch save file
        data = torch.load(file_path)
        if isinstance(data, dict):
            inputs = {k: v if isinstance(v, torch.Tensor) else torch.tensor(v)
                      for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            inputs = {"input": data}
        for name, tensor in inputs.items():
            print(f"  Loaded {name}: {list(tensor.shape)} ({tensor.dtype})")

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return inputs


def visualize_onnx(
    onnx_path: str,
    input_data: dict = None,
    port: int = 8080,
    open_browser: bool = True,
    model_name: str = None,
):
    """
    可视化 ONNX 模型的简单 API

    Args:
        onnx_path: ONNX 文件路径
        input_data: 输入数据字典，如 {"input_ids": tensor, "attention_mask": tensor}
                    如果为 None，自动生成随机数据
        port: 服务器端口
        open_browser: 是否自动打开浏览器
        model_name: 模型名称（用于显示），默认从文件名推断

    Example:
        visualize_onnx("bert.onnx")

        # 带真实输入数据
        visualize_onnx("bert.onnx", input_data={
            "input_ids": torch.tensor([[101, 2054, 2003, 2023, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        })
    """
    import onnx

    print(f"Loading ONNX model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    if model_name is None:
        model_name = Path(onnx_path).stem.upper()

    # Get input info and generate data if needed
    input_info = get_onnx_input_info_from_model(onnx_model)
    if input_data is None:
        print("Generating random input data...")
        input_data = generate_sample_input(input_info, "random")

    # Convert ONNX to FX
    print("Converting ONNX to FX GraphModule...")
    converter = onnx_to_fx.ONNXToFX()
    gm = converter.convert(onnx_model)
    print(f"Graph has {len(list(gm.graph.nodes))} nodes")

    # Start server
    print(f"\nStarting visualization server on port {port}...")

    serve_dynamic(
        models={model_name: gm},
        input_data={model_name: input_data},
        port=port,
        open_browser=open_browser,
    )


def visualize_with_trace_data(
    onnx_path: str,
    trace_data: dict,
    port: int = 8080,
    open_browser: bool = True,
    model_name: str = None,
):
    """
    使用预先捕获的中间张量数据进行可视化

    Args:
        onnx_path: ONNX 文件路径
        trace_data: 中间张量数据，格式为 {node_name: tensor} 或 {node_name: numpy_array}
                    可以从 .npz/.pt 文件加载，或从推理框架导出
        port: 服务器端口
        open_browser: 是否自动打开浏览器

    Example:
        # 从 ONNX Runtime 导出中间数据
        trace = export_onnx_runtime_trace("model.onnx", inputs)
        visualize_with_trace_data("model.onnx", trace)

        # 从文件加载
        trace = load_trace_data("trace.npz")
        visualize_with_trace_data("model.onnx", trace)
    """
    import onnx

    print(f"Loading ONNX model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    if model_name is None:
        model_name = Path(onnx_path).stem.upper()

    # Convert trace_data to torch tensors
    trace_tensors = {}
    for name, data in trace_data.items():
        if isinstance(data, torch.Tensor):
            trace_tensors[name] = data
        elif isinstance(data, np.ndarray):
            trace_tensors[name] = torch.from_numpy(data)
        else:
            trace_tensors[name] = torch.tensor(data)

    print(f"Loaded {len(trace_tensors)} intermediate tensors")

    # Convert ONNX to FX (结构用于显示)
    print("Converting ONNX to FX GraphModule...")
    converter = onnx_to_fx.ONNXToFX()
    gm = converter.convert(onnx_model)

    # 直接注入 trace 数据到 GraphExtractor
    from onnxparser.visualizer.extractor import GraphExtractor

    class PreloadedGraphExtractor(GraphExtractor):
        """GraphExtractor that uses preloaded trace data"""
        def __init__(self, gm, trace_data):
            self.gm = gm
            self.trace_data = trace_data
            self.execution_trace = []
            self._build_trace_from_data()

        def _build_trace_from_data(self):
            """Build execution trace from preloaded data"""
            from onnxparser.visualizer.extractor import analyze_tensor

            for node in self.gm.graph.nodes:
                name = node.name
                target = str(node.target) if node.target else ""

                # 尝试匹配节点名
                value = self.trace_data.get(name)
                if value is None:
                    # 尝试其他名称格式
                    for key in self.trace_data:
                        if name in key or key in name:
                            value = self.trace_data[key]
                            break

                trace_info = {"name": name, "op": node.op}

                if value is not None and isinstance(value, torch.Tensor):
                    # Analyze tensor
                    insights = analyze_tensor(value, node.op, target, name)
                    if insights:
                        trace_info["insights"] = insights

                    trace_info["shape"] = list(value.shape)
                    trace_info["dtype"] = str(value.dtype)

                    # Statistics
                    finite_mask = torch.isfinite(value)
                    if finite_mask.any():
                        finite_values = value[finite_mask]
                        trace_info["min"] = float(finite_values.min().item())
                        trace_info["max"] = float(finite_values.max().item())
                        trace_info["mean"] = float(finite_values.mean().item())
                        trace_info["std"] = float(finite_values.std().item()) if finite_values.numel() > 1 else 0.0

                    # Extract histogram and heatmap
                    trace_info["histogram"] = self._extract_histogram(value)
                    if value.dim() >= 2:
                        trace_info["heatmap"] = self._extract_heatmap(value)
                    trace_info["values"] = self._extract_values_preview(value)

                self.execution_trace.append(trace_info)

    # 使用自定义 extractor
    from onnxparser.visualizer.manager import ModelManager

    class PreloadedModelManager(ModelManager):
        def __init__(self, gm, trace_data):
            super().__init__()
            self._gm = gm
            self._trace_data = trace_data

        def get_graph_data(self, name, width=1200, height=800):
            from onnxparser.visualizer.layout import GraphLayoutEngine

            extractor = PreloadedGraphExtractor(self._gm, self._trace_data)
            data = extractor.extract()

            layout_engine = GraphLayoutEngine()
            data["nodes"] = layout_engine.compute_layout(data["nodes"], data["edges"], width, height)
            return data

    # Start server with custom manager
    from onnxparser.visualizer.server import VisualizerServer
    import onnxparser.visualizer.manager as manager_module

    # 替换全局 manager
    original_manager = manager_module._manager
    manager_module._manager = PreloadedModelManager(gm, trace_tensors)
    manager_module._manager.models[model_name] = gm

    print(f"\nStarting visualization server on port {port}...")

    try:
        server = VisualizerServer(port=port)
        server.start(open_browser=open_browser)
    finally:
        manager_module._manager = original_manager


def export_onnx_runtime_trace(onnx_path: str, input_data: dict) -> dict:
    """
    使用 ONNX Runtime 运行模型并导出所有中间张量

    Args:
        onnx_path: ONNX 文件路径
        input_data: 输入数据字典

    Returns:
        dict: {node_name: numpy_array} 所有中间张量
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("请安装 onnxruntime: pip install onnxruntime")

    import onnx

    # 加载模型并添加所有中间节点作为输出
    model = onnx.load(onnx_path)
    original_outputs = [o.name for o in model.graph.output]

    # 获取所有中间节点名
    intermediate_names = []
    for node in model.graph.node:
        for output in node.output:
            if output and output not in original_outputs:
                intermediate_names.append(output)

    # 添加中间节点为输出
    for name in intermediate_names:
        model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))

    # 运行推理
    sess = ort.InferenceSession(model.SerializeToString())

    # 准备输入
    ort_inputs = {}
    for name, tensor in input_data.items():
        if isinstance(tensor, torch.Tensor):
            ort_inputs[name] = tensor.numpy()
        else:
            ort_inputs[name] = np.array(tensor)

    # 获取所有输出名
    output_names = [o.name for o in sess.get_outputs()]

    # 运行
    outputs = sess.run(output_names, ort_inputs)

    # 构建结果
    trace = {}
    for name, value in zip(output_names, outputs):
        trace[name] = value

    print(f"Captured {len(trace)} intermediate tensors")
    return trace


def save_trace_data(trace: dict, path: str):
    """保存中间张量数据到文件"""
    p = Path(path)
    if p.suffix == ".npz":
        np.savez(path, **{k: v if isinstance(v, np.ndarray) else v.numpy() for k, v in trace.items()})
    elif p.suffix == ".pt":
        torch.save({k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in trace.items()}, path)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")
    print(f"Saved trace data to {path}")


def load_trace_data(path: str) -> dict:
    """从文件加载中间张量数据"""
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(path)
        return {name: data[name] for name in data.files}
    elif p.suffix == ".pt":
        return torch.load(path)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")


def get_onnx_input_info_from_model(model) -> dict:
    """Extract input information from ONNX ModelProto"""
    inputs = {}
    initializer_names = {init.name for init in model.graph.initializer}

    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue  # Skip weights

        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(1 if "batch" in dim.dim_param.lower() else 128)
            else:
                shape.append(1)

        elem_type = inp.type.tensor_type.elem_type
        dtype_map = {1: torch.float32, 6: torch.int32, 7: torch.int64, 9: torch.bool}
        dtype = dtype_map.get(elem_type, torch.float32)

        inputs[inp.name] = {"shape": shape, "dtype": dtype}

    return inputs


def create_attention_mask(seq_len: int, batch_size: int = 1, causal: bool = True) -> torch.Tensor:
    """Create attention mask

    Args:
        seq_len: Sequence length
        batch_size: Batch size
        causal: If True, create causal (triangular) mask
    """
    if causal:
        # Causal mask: upper triangular with -inf
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, 1, seq_len, seq_len)
    else:
        # Full attention (all zeros)
        mask = torch.zeros(batch_size, 1, seq_len, seq_len)

    return mask


def main():
    parser = argparse.ArgumentParser(description="Visualize ONNX model with real input data")
    parser.add_argument("onnx_path", nargs="?", help="Path to ONNX model file")
    parser.add_argument("--input", "-i", type=str, help="Path to .npz file with input data")
    parser.add_argument("--mode", choices=["random", "zeros", "ones"], default="random",
                        help="Input generation mode if no --input provided")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--seq-len", type=int, help="Override sequence length")
    parser.add_argument("--add-causal-mask", action="store_true",
                        help="Add causal attention mask")
    parser.add_argument("--list-inputs", action="store_true",
                        help="List ONNX model inputs and exit")
    args = parser.parse_args()

    # Demo mode if no ONNX path provided
    if not args.onnx_path:
        print("No ONNX file provided. Running demo with built-in transformer model.\n")
        print("Usage examples:")
        print("  python examples/visualize_onnx.py model.onnx")
        print("  python examples/visualize_onnx.py model.onnx --input data.npz")
        print("  python examples/visualize_onnx.py model.onnx --batch 4 --seq-len 128")
        print("  python examples/visualize_onnx.py model.onnx --list-inputs")
        print()

        # Run demo with built-in model
        from examples.visualize_graph import build_transformer
        gm = build_transformer()
        input_data = {"input": torch.randn(2, 8, 64)}
        model_name = "DEMO_TRANSFORMER"

        print(f"Starting visualization server on port {args.port}...")
        serve_dynamic(
            models={model_name: gm},
            input_data={model_name: input_data},
            port=args.port,
            open_browser=True,
        )
        return

    # Load ONNX model
    onnx_path = args.onnx_path
    if not Path(onnx_path).exists():
        print(f"Error: ONNX file not found: {onnx_path}")
        return

    print(f"Loading ONNX model: {onnx_path}")

    # Get input info
    input_info = get_onnx_input_info(onnx_path)
    print(f"\nModel inputs ({len(input_info)}):")
    for name, info in input_info.items():
        print(f"  {name}: {info['shape']} ({info['dtype']})")

    if args.list_inputs:
        return

    # Override dimensions if specified
    if args.batch or args.seq_len:
        for name, info in input_info.items():
            shape = info["shape"]
            if args.batch and len(shape) > 0:
                shape[0] = args.batch
            if args.seq_len and len(shape) > 1:
                shape[1] = args.seq_len
            input_info[name]["shape"] = shape

    # Generate or load input data
    print("\nPreparing input data:")
    if args.input:
        input_data = load_input_from_file(args.input)
    else:
        input_data = generate_sample_input(input_info, args.mode)

    # Add causal mask if requested
    if args.add_causal_mask:
        # Find sequence length from first input
        seq_len = list(input_data.values())[0].shape[1] if input_data else 128
        batch_size = list(input_data.values())[0].shape[0] if input_data else 1
        input_data["attention_mask"] = create_attention_mask(seq_len, batch_size, causal=True)
        print(f"  attention_mask: {list(input_data['attention_mask'].shape)} (causal)")

    # Convert ONNX to FX
    print("\nConverting ONNX to FX GraphModule...")
    try:
        gm = onnx_to_fx.load(onnx_path)
        print(f"Graph has {len(list(gm.graph.nodes))} nodes")
    except Exception as e:
        print(f"Error converting ONNX: {e}")
        return

    # Start visualization
    model_name = Path(onnx_path).stem.upper()
    print(f"\nStarting visualization server on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser")

    serve_dynamic(
        models={model_name: gm},
        input_data={model_name: input_data},
        port=args.port,
        open_browser=True,
    )


if __name__ == "__main__":
    main()
