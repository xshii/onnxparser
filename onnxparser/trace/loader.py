# -*- coding: utf-8 -*-
"""Trace binary loader."""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from .format import FileHeader, TensorHeader, TraceFormat, TRACE_MAGIC, DTYPE_MAP


class TraceLoader:
    """
    Load trace data from binary file.

    Usage:
        loader = TraceLoader("trace.bin")
        tensor = loader["tensor_name"]
        tensor = loader.get("tensor_name")

        # List all tensors
        for name in loader.names:
            print(name, loader[name].shape)

        # Get as dict for visualization
        trace_dict = loader.to_dict()
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._headers: List[TensorHeader] = []
        self._name_to_idx: Dict[str, int] = {}
        self._data_offset: int = 0
        self._load_headers()

    def _load_headers(self):
        """Load file header and tensor headers."""
        with open(self.path, 'rb') as f:
            # Read file header
            file_hdr_data = f.read(FileHeader.SIZE)
            file_hdr = FileHeader.unpack(file_hdr_data)

            if file_hdr.magic != TRACE_MAGIC:
                raise ValueError(f"Invalid trace file magic: {hex(file_hdr.magic)}")

            # Read tensor headers
            for i in range(file_hdr.num_tensors):
                tensor_hdr_data = f.read(TensorHeader.SIZE)
                tensor_hdr = TensorHeader.unpack(tensor_hdr_data)
                self._headers.append(tensor_hdr)
                self._name_to_idx[tensor_hdr.name] = i

            self._data_offset = f.tell()

    @property
    def names(self) -> List[str]:
        """List of tensor names."""
        return list(self._name_to_idx.keys())

    @property
    def num_tensors(self) -> int:
        return len(self._headers)

    def get_header(self, name: str) -> Optional[TensorHeader]:
        """Get tensor header by name."""
        idx = self._name_to_idx.get(name)
        if idx is None:
            return None
        return self._headers[idx]

    def _get_tensor_offset(self, idx: int) -> int:
        """Calculate data offset for tensor at index."""
        offset = self._data_offset
        for i in range(idx):
            offset += self._headers[i].data_size
        return offset

    def __getitem__(self, name: str) -> np.ndarray:
        """Load tensor data by name."""
        idx = self._name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"Tensor not found: {name}")

        hdr = self._headers[idx]
        if not hdr.valid:
            raise ValueError(f"Tensor '{name}' was not captured (valid=0)")

        offset = self._get_tensor_offset(idx)
        dtype_str, _, _ = hdr.dtype_info

        with open(self.path, 'rb') as f:
            f.seek(offset)
            data = f.read(hdr.data_size)

        arr = np.frombuffer(data, dtype=dtype_str)
        return arr.reshape(hdr.shape)

    def get(self, name: str, default=None) -> Optional[np.ndarray]:
        """Load tensor data, return default if not found or invalid."""
        try:
            return self[name]
        except (KeyError, ValueError):
            return default

    def to_dict(self, valid_only: bool = True) -> Dict[str, np.ndarray]:
        """Convert to dict for visualization."""
        result = {}
        for hdr in self._headers:
            if valid_only and not hdr.valid:
                continue
            try:
                result[hdr.name] = self[hdr.name]
            except ValueError:
                pass
        return result

    def summary(self) -> str:
        """Print summary of trace contents."""
        lines = [f"Trace: {self.path.name}"]
        lines.append(f"Tensors: {self.num_tensors}")

        valid_count = sum(1 for h in self._headers if h.valid)
        lines.append(f"Captured: {valid_count}/{self.num_tensors}")

        total_size = sum(h.data_size for h in self._headers if h.valid)
        lines.append(f"Data size: {total_size / 1024 / 1024:.2f} MB")

        lines.append("\nTensors:")
        for hdr in self._headers:
            status = "OK" if hdr.valid else "MISSING"
            dtype_str = hdr.dtype_info[0]
            lines.append(f"  [{status:7}] {hdr.name}: {list(hdr.shape)} ({dtype_str})")

        return "\n".join(lines)


def load_trace(path: str) -> TraceLoader:
    """Load trace from binary file."""
    return TraceLoader(path)


def merge_traces(paths: List[str], output: str) -> TraceLoader:
    """
    Merge multiple partial trace files.

    If same tensor appears in multiple files, use the first valid one.
    """
    if not paths:
        raise ValueError("No trace files provided")

    # Load first as base
    base = TraceLoader(paths[0])
    base_format = TraceFormat(
        header=FileHeader(num_tensors=base.num_tensors),
        tensors=base._headers.copy()
    )

    # Track which tensors have valid data
    valid_data = {}
    for hdr in base._headers:
        if hdr.valid:
            valid_data[hdr.name] = base[hdr.name]

    # Merge others
    for path in paths[1:]:
        loader = TraceLoader(path)
        for name in loader.names:
            if name not in valid_data:
                hdr = loader.get_header(name)
                if hdr and hdr.valid:
                    valid_data[name] = loader[name]

    # Write merged file
    with open(output, 'wb') as f:
        # Update headers with valid flags
        for hdr in base_format.tensors:
            hdr.valid = 1 if hdr.name in valid_data else 0

        # Write headers
        f.write(base_format.pack_headers())

        # Write data
        for hdr in base_format.tensors:
            if hdr.name in valid_data:
                f.write(valid_data[hdr.name].tobytes())
            else:
                # Write zeros for missing data
                f.write(b'\x00' * hdr.data_size)

    return TraceLoader(output)
