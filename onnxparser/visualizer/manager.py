# -*- coding: utf-8 -*-
"""Model manager for visualization server"""

from typing import Dict, Optional, Any, List

import torch
import torch.fx as fx

from .extractor import GraphExtractor
from .layout import GraphLayoutEngine
from .memory_wrapper import MemoryAnalyzerWrapper


class ModelManager:
    """Manage multiple loaded models with caching"""

    def __init__(self):
        self.models: Dict[str, fx.GraphModule] = {}
        self.input_data: Dict[str, Dict[str, torch.Tensor]] = {}
        self.cache: Dict[str, Dict[str, Any]] = {}

    def add_model(self, name: str, gm: fx.GraphModule,
                  input_data: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Add a model to the manager"""
        self.models[name] = gm
        if input_data:
            self.input_data[name] = input_data
        # Clear cache for this model
        self._clear_model_cache(name)

    def remove_model(self, name: str) -> bool:
        """Remove a model from the manager"""
        if name in self.models:
            del self.models[name]
            self.input_data.pop(name, None)
            self._clear_model_cache(name)
            return True
        return False

    def list_models(self) -> List[str]:
        """List all loaded model names"""
        return list(self.models.keys())

    def get_graph_data(self, name: str, width: int = 1200, height: int = 800) -> Optional[Dict]:
        """Get graph data with layout for a model"""
        if name not in self.models:
            return None

        cache_key = f"{name}_graph_{width}_{height}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        gm = self.models[name]
        input_data = self.input_data.get(name)

        # Extract graph structure with tensor data
        extractor = GraphExtractor(gm, input_data)
        data = extractor.extract()

        # Compute layout
        layout_engine = GraphLayoutEngine()
        data["nodes"] = layout_engine.compute_layout(
            data["nodes"], data["edges"], width, height
        )

        self.cache[cache_key] = data
        return data

    def get_memory_data(self, name: str, strategy: str = "greedy",
                        memory_limit_kb: Optional[float] = None) -> Optional[Dict]:
        """Get memory analysis data for a model"""
        if name not in self.models:
            return None

        # Include memory limit in cache key
        limit_str = f"_{memory_limit_kb}" if memory_limit_kb else ""
        cache_key = f"{name}_memory_{strategy}{limit_str}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        gm = self.models[name]
        input_data = self.input_data.get(name)
        data = MemoryAnalyzerWrapper.analyze(gm, strategy, input_data, memory_limit_kb)
        self.cache[cache_key] = data
        return data

    def clear_cache(self, name: Optional[str] = None) -> None:
        """Clear cache for a specific model or all models"""
        if name:
            self._clear_model_cache(name)
        else:
            self.cache.clear()

    def _clear_model_cache(self, name: str) -> None:
        """Clear all cache entries for a specific model"""
        keys_to_remove = [k for k in self.cache.keys() if k.startswith(name)]
        for k in keys_to_remove:
            self.cache.pop(k, None)


# Global model manager instance
_manager = ModelManager()


def get_manager() -> ModelManager:
    """Get the global model manager instance"""
    return _manager


def main():
    """Test model manager with a demo model"""
    import torch.nn as nn

    print("Testing ModelManager...")
    print("-" * 50)

    # Build a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 128)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(128, 64)

        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x

    model = SimpleModel()
    gm = fx.symbolic_trace(model)

    # Test manager
    manager = ModelManager()

    # Add model
    input_data = {"x": torch.randn(2, 64)}
    manager.add_model("test_model", gm, input_data)
    print(f"Added model: test_model")
    print(f"Loaded models: {manager.list_models()}")

    # Get graph data
    graph_data = manager.get_graph_data("test_model", 800, 600)
    if graph_data:
        print(f"\nGraph data:")
        print(f"  Nodes: {len(graph_data['nodes'])}")
        print(f"  Edges: {len(graph_data['edges'])}")
        for node in graph_data["nodes"][:5]:
            print(f"    - {node['name']} ({node['type']})")

    # Get memory data
    memory_data = manager.get_memory_data("test_model")
    if memory_data and "error" not in memory_data:
        print(f"\nMemory data:")
        print(f"  Tensors: {len(memory_data.get('tensors', []))}")
        print(f"  Peak memory: {memory_data.get('summary', {}).get('peak_max_kb', 'N/A')} KB")
    elif memory_data and "error" in memory_data:
        print(f"\nMemory analysis not available: {memory_data['error']}")

    # Remove model
    manager.remove_model("test_model")
    print(f"\nAfter removal: {manager.list_models()}")

    print("\nModelManager test completed!")


if __name__ == "__main__":
    main()
