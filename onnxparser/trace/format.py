# -*- coding: utf-8 -*-
"""Trace binary format definitions."""

import struct
from dataclasses import dataclass
from typing import List, Tuple
from enum import IntEnum


# Magic number: "XTRC"
TRACE_MAGIC = 0x43525458

# Version
TRACE_VERSION = 1

# Data type mapping
class DType(IntEnum):
    FLOAT32 = 0
    FLOAT16 = 1
    INT32 = 2
    INT64 = 3
    INT8 = 4
    UINT8 = 5
    BOOL = 6
    BFLOAT16 = 7


DTYPE_MAP = {
    DType.FLOAT32: ("float32", "float", 4),
    DType.FLOAT16: ("float16", "uint16_t", 2),  # fp16 as uint16 for storage
    DType.INT32: ("int32", "int32_t", 4),
    DType.INT64: ("int64", "int64_t", 8),
    DType.INT8: ("int8", "int8_t", 1),
    DType.UINT8: ("uint8", "uint8_t", 1),
    DType.BOOL: ("bool", "uint8_t", 1),
    DType.BFLOAT16: ("bfloat16", "uint16_t", 2),
}

# ONNX dtype to our dtype
ONNX_DTYPE_MAP = {
    1: DType.FLOAT32,   # FLOAT
    2: DType.UINT8,     # UINT8
    3: DType.INT8,      # INT8
    6: DType.INT32,     # INT32
    7: DType.INT64,     # INT64
    9: DType.BOOL,      # BOOL
    10: DType.FLOAT16,  # FLOAT16
    16: DType.BFLOAT16, # BFLOAT16
}


@dataclass
class FileHeader:
    """
    File header: 64 bytes

    Layout:
        magic (4B)       - 0x43525458 "XTRC"
        version (2B)     - format version
        flags (2B)       - reserved flags
        num_tensors (4B) - number of tensors
        reserved (52B)   - padding to 64 bytes
    """
    magic: int = TRACE_MAGIC
    version: int = TRACE_VERSION
    flags: int = 0
    num_tensors: int = 0

    FORMAT = "<IHHI52s"
    SIZE = 64

    def pack(self) -> bytes:
        return struct.pack(
            self.FORMAT,
            self.magic,
            self.version,
            self.flags,
            self.num_tensors,
            b'\x00' * 52
        )

    @classmethod
    def unpack(cls, data: bytes) -> "FileHeader":
        magic, version, flags, num_tensors, _ = struct.unpack(cls.FORMAT, data[:cls.SIZE])
        return cls(magic=magic, version=version, flags=flags, num_tensors=num_tensors)


@dataclass
class TensorHeader:
    """
    Tensor header: 128 bytes

    Layout:
        name (48B)          - tensor name (null-terminated)
        dtype (1B)          - data type enum (see DType / HWDType)
        hw_dtype (1B)       - hardware-specific dtype (for BFP, FP8, etc.)
        ndim (1B)           - number of dimensions
        valid (1B)          - 1 if data captured, 0 if not
        is_input (1B)       - 1 if model input, 0 otherwise
        has_exponent (1B)   - 1 if BFP format with separate exponent
        block_size (2B)     - BFP block size (0 if not BFP)
        shape (32B)         - int32[8], up to 8 dimensions
        data_size (8B)      - mantissa/data size in bytes
        exp_size (8B)       - exponent size in bytes (0 if not BFP)
        exp_offset (8B)     - exponent offset relative to data start
        reserved (16B)      - padding to 128 bytes
    """
    name: str
    dtype: int
    ndim: int
    valid: int
    is_input: int
    shape: Tuple[int, ...]
    data_size: int
    hw_dtype: int = 0       # Hardware-specific dtype
    has_exponent: int = 0   # BFP has separate exponent
    block_size: int = 0     # BFP block size
    exp_size: int = 0       # Exponent data size
    exp_offset: int = 0     # Exponent offset

    FORMAT = "<48sBBBBBBH8IQQQ16s"
    SIZE = 128

    def pack(self) -> bytes:
        name_bytes = self.name.encode('utf-8')[:47] + b'\x00'
        name_bytes = name_bytes.ljust(48, b'\x00')

        shape_padded = list(self.shape) + [0] * (8 - len(self.shape))

        return struct.pack(
            self.FORMAT,
            name_bytes,
            self.dtype,
            self.hw_dtype,
            self.ndim,
            self.valid,
            self.is_input,
            self.has_exponent,
            self.block_size,
            *shape_padded[:8],
            self.data_size,
            self.exp_size,
            self.exp_offset,
            b'\x00' * 16  # reserved
        )

    @classmethod
    def unpack(cls, data: bytes) -> "TensorHeader":
        unpacked = struct.unpack(cls.FORMAT, data[:cls.SIZE])
        name_bytes = unpacked[0]
        name = name_bytes.rstrip(b'\x00').decode('utf-8')
        dtype = unpacked[1]
        hw_dtype = unpacked[2]
        ndim = unpacked[3]
        valid = unpacked[4]
        is_input = unpacked[5]
        has_exponent = unpacked[6]
        block_size = unpacked[7]
        shape = tuple(unpacked[8:8+ndim])
        data_size = unpacked[16]
        exp_size = unpacked[17]
        exp_offset = unpacked[18]

        return cls(
            name=name,
            dtype=dtype,
            hw_dtype=hw_dtype,
            ndim=ndim,
            valid=valid,
            is_input=is_input,
            has_exponent=has_exponent,
            block_size=block_size,
            shape=shape,
            data_size=data_size,
            exp_size=exp_size,
            exp_offset=exp_offset,
        )

    @property
    def numel(self) -> int:
        """Total number of elements."""
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def dtype_info(self) -> Tuple[str, str, int]:
        """(numpy_dtype, c_type, element_size)"""
        return DTYPE_MAP.get(self.dtype, ("float32", "float", 4))


@dataclass
class TraceFormat:
    """Complete trace file format."""
    header: FileHeader
    tensors: List[TensorHeader]

    @property
    def data_offset(self) -> int:
        """Offset where tensor data starts."""
        return FileHeader.SIZE + len(self.tensors) * TensorHeader.SIZE

    def get_tensor_offset(self, index: int) -> int:
        """Get data offset for tensor at index."""
        offset = self.data_offset
        for i in range(index):
            offset += self.tensors[i].data_size
        return offset

    def pack_headers(self) -> bytes:
        """Pack file header and all tensor headers."""
        data = self.header.pack()
        for tensor in self.tensors:
            data += tensor.pack()
        return data

    @classmethod
    def from_onnx(cls, onnx_path: str) -> "TraceFormat":
        """Create TraceFormat from ONNX model."""
        import onnx
        model = onnx.load(onnx_path)
        return cls.from_onnx_model(model)

    @classmethod
    def from_onnx_model(cls, model) -> "TraceFormat":
        """Create TraceFormat from ONNX ModelProto."""
        tensors = []

        # Collect initializer names (weights)
        initializer_names = {init.name for init in model.graph.initializer}

        # Model inputs (not initializers)
        for inp in model.graph.input:
            if inp.name in initializer_names:
                continue

            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(1)  # dynamic dim default

            dtype = ONNX_DTYPE_MAP.get(inp.type.tensor_type.elem_type, DType.FLOAT32)
            elem_size = DTYPE_MAP[dtype][2]
            numel = 1
            for s in shape:
                numel *= s

            tensors.append(TensorHeader(
                name=inp.name,
                dtype=dtype,
                ndim=len(shape),
                valid=0,
                is_input=1,
                shape=tuple(shape),
                data_size=numel * elem_size
            ))

        # Node outputs
        for node in model.graph.node:
            for output_name in node.output:
                if not output_name:
                    continue

                # Try to infer shape from value_info
                shape = []
                dtype = DType.FLOAT32

                for vi in model.graph.value_info:
                    if vi.name == output_name:
                        for dim in vi.type.tensor_type.shape.dim:
                            if dim.dim_value > 0:
                                shape.append(dim.dim_value)
                            else:
                                shape.append(1)
                        dtype = ONNX_DTYPE_MAP.get(
                            vi.type.tensor_type.elem_type, DType.FLOAT32
                        )
                        break

                # Also check graph outputs
                for out in model.graph.output:
                    if out.name == output_name:
                        shape = []
                        for dim in out.type.tensor_type.shape.dim:
                            if dim.dim_value > 0:
                                shape.append(dim.dim_value)
                            else:
                                shape.append(1)
                        dtype = ONNX_DTYPE_MAP.get(
                            out.type.tensor_type.elem_type, DType.FLOAT32
                        )
                        break

                if not shape:
                    shape = [1]  # fallback

                elem_size = DTYPE_MAP[dtype][2]
                numel = 1
                for s in shape:
                    numel *= s

                tensors.append(TensorHeader(
                    name=output_name,
                    dtype=dtype,
                    ndim=len(shape),
                    valid=0,
                    is_input=0,
                    shape=tuple(shape),
                    data_size=numel * elem_size
                ))

        header = FileHeader(num_tensors=len(tensors))
        return cls(header=header, tensors=tensors)
