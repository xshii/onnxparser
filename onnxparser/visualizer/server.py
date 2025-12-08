# -*- coding: utf-8 -*-
"""Flask backend for dynamic visualization with REST API"""

import os
import signal
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.fx as fx
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from .layout import GraphLayoutEngine
from .manager import ModelManager, get_manager
from .memory_wrapper import MemoryAnalyzerWrapper


def create_app() -> Flask:
    """Create Flask application with API routes"""
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")
    CORS(app)

    manager = get_manager()

    @app.route("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.route("/api/models", methods=["GET"])
    def list_models():
        return jsonify({"models": manager.list_models()})

    @app.route("/api/model", methods=["GET", "DELETE"])
    def handle_model():
        name = request.args.get("name")
        if not name:
            return jsonify({"error": "Missing model name"}), 400

        if request.method == "GET":
            width = request.args.get("width", 1200, type=int)
            height = request.args.get("height", 800, type=int)

            data = manager.get_graph_data(name, width, height)
            if data:
                return jsonify({"name": name, "graph": data})
            return jsonify({"error": f"Model '{name}' not found"}), 404

        elif request.method == "DELETE":
            if manager.remove_model(name):
                return jsonify({"success": True})
            return jsonify({"error": f"Model '{name}' not found"}), 404

    @app.route("/api/memory", methods=["GET"])
    def get_memory():
        name = request.args.get("name")
        strategy = request.args.get("strategy", "greedy")

        if not name:
            return jsonify({"error": "Missing model name"}), 400

        data = manager.get_memory_data(name, strategy)
        if data:
            return jsonify(data)
        return jsonify({"error": f"Model '{name}' not found"}), 404

    @app.route("/api/strategies", methods=["GET"])
    def list_strategies():
        return jsonify({"strategies": MemoryAnalyzerWrapper.list_strategies()})

    @app.route("/api/layout", methods=["POST"])
    def compute_layout():
        """Recompute layout with new canvas size"""
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        width = data.get("width", 1200)
        height = data.get("height", 800)

        engine = GraphLayoutEngine()
        nodes = engine.compute_layout(nodes, edges, width, height)
        return jsonify({"nodes": nodes})

    @app.route("/api/reload", methods=["POST"])
    def reload_model():
        """Clear cache and reload model data"""
        name = request.args.get("name")
        manager.clear_cache(name)
        return jsonify({"success": True, "cleared": name or "all"})

    return app


def _is_port_in_use(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _kill_existing_server(port: int) -> bool:
    """Try to kill an existing visualizer server on the port"""
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True
            )
            if result.returncode != 0 or not result.stdout.strip():
                return False

            for pid in result.stdout.strip().split("\n"):
                pid = pid.strip()
                if not pid:
                    continue
                try:
                    if sys.platform == "darwin":
                        ps_result = subprocess.run(
                            ["ps", "-p", pid, "-o", "command="],
                            capture_output=True, text=True
                        )
                        cmdline = ps_result.stdout
                        if "python" in cmdline.lower() and "onnxparser" in cmdline:
                            os.kill(int(pid), signal.SIGTERM)
                            print(f"Killed existing server (PID: {pid})")
                            return True
                except (ProcessLookupError, PermissionError):
                    continue
        except FileNotFoundError:
            pass
    return False


class VisualizerServer:
    """Flask-based visualization server"""

    def __init__(self, port: int = 8080):
        self.port = port
        self.app = create_app()
        self.manager = get_manager()

    def add_model(self, name: str, gm: fx.GraphModule,
                  input_data: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Add a model to the server"""
        self.manager.add_model(name, gm, input_data)

    def remove_model(self, name: str) -> bool:
        """Remove a model from the server"""
        return self.manager.remove_model(name)

    def list_models(self):
        """List all loaded models"""
        return self.manager.list_models()

    def start(self, open_browser: bool = True, debug: bool = False) -> None:
        """Start the visualization server"""
        if _is_port_in_use(self.port):
            print(f"Port {self.port} in use, attempting to kill existing server...")
            if _kill_existing_server(self.port):
                import time
                time.sleep(0.5)

        url = f"http://localhost:{self.port}"
        print(f"Starting visualizer server at: {url}")
        print(f"Loaded models: {self.list_models()}")

        if open_browser:
            import threading
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()

        self.app.run(host="0.0.0.0", port=self.port, debug=debug, use_reloader=False)


def serve_dynamic(models: Optional[Dict[str, fx.GraphModule]] = None,
                  input_data: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
                  port: int = 8080,
                  open_browser: bool = True,
                  debug: bool = False) -> VisualizerServer:
    """
    Start dynamic visualization server with multiple models.

    Args:
        models: Dict of model_name -> GraphModule
        input_data: Dict of model_name -> input tensors
        port: Server port
        open_browser: Whether to open browser
        debug: Enable Flask debug mode

    Example:
        >>> serve_dynamic({"model1": gm1, "model2": gm2}, port=8080)
    """
    server = VisualizerServer(port=port)

    if models:
        for name, gm in models.items():
            inp = input_data.get(name) if input_data else None
            server.add_model(name, gm, inp)

    server.start(open_browser=open_browser, debug=debug)
    return server


def _build_demo_model():
    """Build a demo transformer model for testing"""
    from ..builder import GraphBuilder

    print("Building demo transformer model...")
    builder = GraphBuilder()

    # Simple transformer block
    x = builder.placeholder("input", shape=(2, 8, 64))

    # Attention
    q = builder.linear(x, 64, name="query")
    k = builder.linear(x, 64, name="key")
    v = builder.linear(x, 64, name="value")

    scores = builder.matmul(q, builder.transpose(k, -2, -1))
    scores = builder.div(scores, 8.0)
    attn = builder.softmax(scores, dim=-1)
    attn_out = builder.matmul(attn, v)

    # FFN
    ffn = builder.linear(attn_out, 256, name="ffn1")
    ffn = builder.relu(ffn)
    ffn = builder.linear(ffn, 64, name="ffn2")

    out = builder.add(x, ffn)
    out = builder.layer_norm(out, [64], name="ln")

    builder.output(out)
    return builder.build()


def main():
    """Main entry point for standalone server execution"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ONNX Parser Visualization Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with demo model
  python -m onnxparser.visualizer.server

  # Specify port
  python -m onnxparser.visualizer.server --port 8888

  # Don't open browser
  python -m onnxparser.visualizer.server --no-browser
"""
    )
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    args = parser.parse_args()

    # Build demo model
    gm = _build_demo_model()
    input_data = {"input": torch.randn(2, 8, 64)}

    print(f"\nStarting visualization server on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser\n")

    serve_dynamic(
        models={"DEMO": gm},
        input_data={"DEMO": input_data},
        port=args.port,
        open_browser=not args.no_browser,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
