# -*- coding: utf-8 -*-
"""
Hardware-specific data type conversions.

Supports:
- BFP (Block Floating Point): BFP8, BFP16, etc.
- FP8 variants: E4M3, E5M2
- INT4/INT8 quantized
- Custom fixed-point formats
"""

import numpy as np
from enum import IntEnum
from typing import Tuple, Callable, Optional
from dataclasses import dataclass


class HWDType(IntEnum):
    """Hardware data types."""
    # Standard types
    FP32 = 0
    FP16 = 1
    BF16 = 2
    INT32 = 3
    INT64 = 4
    INT8 = 5
    UINT8 = 6
    INT4 = 7
    UINT4 = 8

    # FP8 variants
    FP8_E4M3 = 10
    FP8_E5M2 = 11

    # Block Floating Point
    BFP8 = 20     # 8-bit mantissa + shared exponent
    BFP12 = 21    # 12-bit mantissa + shared exponent
    BFP16 = 22    # 16-bit mantissa + shared exponent

    # Fixed point
    FXP8 = 30     # 8-bit fixed point
    FXP16 = 31    # 16-bit fixed point
    FXP32 = 32    # 32-bit fixed point


@dataclass
class BFPConfig:
    """Block Floating Point configuration."""
    mantissa_bits: int    # Bits per element
    exponent_bits: int    # Shared exponent bits (usually 8)
    block_size: int       # Number of elements sharing exponent
    signed: bool = True


# Common BFP configurations
BFP_CONFIGS = {
    HWDType.BFP8: BFPConfig(mantissa_bits=8, exponent_bits=8, block_size=16),
    HWDType.BFP12: BFPConfig(mantissa_bits=12, exponent_bits=8, block_size=16),
    HWDType.BFP16: BFPConfig(mantissa_bits=16, exponent_bits=8, block_size=16),
}


@dataclass
class FXPConfig:
    """Fixed Point configuration."""
    total_bits: int
    frac_bits: int  # Fractional bits
    signed: bool = True


# =============================================================================
# BFP Conversion
# =============================================================================

def bfp_to_fp32(
    data: np.ndarray,
    exponents: np.ndarray,
    config: BFPConfig,
) -> np.ndarray:
    """
    Convert Block Floating Point to FP32.

    Args:
        data: Mantissa data, shape (..., block_size)
        exponents: Shared exponents, shape (...,) - one per block
        config: BFP configuration

    Returns:
        FP32 array with same shape as data
    """
    # Ensure data is integer type for bit manipulation
    if data.dtype not in (np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32):
        data = data.astype(np.int32)

    # Handle signed mantissa
    if config.signed:
        # Two's complement interpretation
        max_val = 2 ** (config.mantissa_bits - 1)
        data = data.astype(np.float32)
    else:
        max_val = 2 ** config.mantissa_bits
        data = data.astype(np.float32)

    # Normalize mantissa to [-1, 1) or [0, 1)
    data = data / max_val

    # Apply shared exponent
    # Expand exponents to match data shape
    exp_shape = exponents.shape + (1,) * (data.ndim - exponents.ndim)
    exponents = exponents.reshape(exp_shape).astype(np.float32)

    # Compute 2^exponent
    scale = np.power(2.0, exponents - 127)  # Bias of 127 like FP32

    # Broadcast and multiply
    result = data * scale

    return result


def fp32_to_bfp(
    data: np.ndarray,
    config: BFPConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert FP32 to Block Floating Point.

    Args:
        data: FP32 data
        config: BFP configuration

    Returns:
        Tuple of (mantissa, exponents)
    """
    original_shape = data.shape

    # Reshape to (..., block_size)
    if data.size % config.block_size != 0:
        # Pad to multiple of block_size
        pad_size = config.block_size - (data.size % config.block_size)
        data = np.pad(data.flatten(), (0, pad_size), mode='constant')

    data = data.reshape(-1, config.block_size)
    num_blocks = data.shape[0]

    # Find max absolute value per block for shared exponent
    max_abs = np.max(np.abs(data), axis=1, keepdims=True)
    max_abs = np.maximum(max_abs, 1e-10)  # Avoid log(0)

    # Compute shared exponent
    exponents = np.floor(np.log2(max_abs)).astype(np.int32) + 127  # Bias
    exponents = np.clip(exponents, 0, 255).flatten()

    # Compute scale and quantize mantissa
    scale = np.power(2.0, exponents.reshape(-1, 1).astype(np.float32) - 127)
    normalized = data / scale

    # Quantize to mantissa bits
    if config.signed:
        max_val = 2 ** (config.mantissa_bits - 1) - 1
        mantissa = np.clip(np.round(normalized * max_val), -max_val, max_val)
        mantissa = mantissa.astype(np.int16 if config.mantissa_bits <= 16 else np.int32)
    else:
        max_val = 2 ** config.mantissa_bits - 1
        mantissa = np.clip(np.round(normalized * max_val), 0, max_val)
        mantissa = mantissa.astype(np.uint16 if config.mantissa_bits <= 16 else np.uint32)

    return mantissa, exponents.astype(np.uint8)


# =============================================================================
# FP8 Conversion
# =============================================================================

def fp8_e4m3_to_fp32(data: np.ndarray) -> np.ndarray:
    """Convert FP8 E4M3 to FP32."""
    # E4M3: 1 sign, 4 exponent, 3 mantissa
    data = data.astype(np.uint8)

    sign = (data >> 7) & 0x1
    exp = (data >> 3) & 0xF
    mant = data & 0x7

    result = np.zeros_like(data, dtype=np.float32)

    # Normal numbers
    normal_mask = (exp > 0) & (exp < 15)
    result[normal_mask] = (1.0 + mant[normal_mask] / 8.0) * np.power(2.0, exp[normal_mask] - 7)

    # Subnormal numbers
    subnormal_mask = exp == 0
    result[subnormal_mask] = (mant[subnormal_mask] / 8.0) * np.power(2.0, -6)

    # Apply sign
    result = np.where(sign, -result, result)

    return result


def fp8_e5m2_to_fp32(data: np.ndarray) -> np.ndarray:
    """Convert FP8 E5M2 to FP32."""
    # E5M2: 1 sign, 5 exponent, 2 mantissa
    data = data.astype(np.uint8)

    sign = (data >> 7) & 0x1
    exp = (data >> 2) & 0x1F
    mant = data & 0x3

    result = np.zeros_like(data, dtype=np.float32)

    # Normal numbers
    normal_mask = (exp > 0) & (exp < 31)
    result[normal_mask] = (1.0 + mant[normal_mask] / 4.0) * np.power(2.0, exp[normal_mask] - 15)

    # Subnormal numbers
    subnormal_mask = exp == 0
    result[subnormal_mask] = (mant[subnormal_mask] / 4.0) * np.power(2.0, -14)

    # Apply sign
    result = np.where(sign, -result, result)

    return result


# =============================================================================
# Fixed Point Conversion
# =============================================================================

def fxp_to_fp32(
    data: np.ndarray,
    total_bits: int,
    frac_bits: int,
    signed: bool = True,
) -> np.ndarray:
    """Convert fixed-point to FP32."""
    if signed:
        # Sign extend if needed
        if total_bits <= 8:
            data = data.astype(np.int8)
        elif total_bits <= 16:
            data = data.astype(np.int16)
        else:
            data = data.astype(np.int32)
    else:
        if total_bits <= 8:
            data = data.astype(np.uint8)
        elif total_bits <= 16:
            data = data.astype(np.uint16)
        else:
            data = data.astype(np.uint32)

    scale = 2.0 ** (-frac_bits)
    return data.astype(np.float32) * scale


def fp32_to_fxp(
    data: np.ndarray,
    total_bits: int,
    frac_bits: int,
    signed: bool = True,
) -> np.ndarray:
    """Convert FP32 to fixed-point."""
    scale = 2.0 ** frac_bits

    if signed:
        max_val = 2 ** (total_bits - 1) - 1
        min_val = -(2 ** (total_bits - 1))
    else:
        max_val = 2 ** total_bits - 1
        min_val = 0

    quantized = np.clip(np.round(data * scale), min_val, max_val)

    if total_bits <= 8:
        dtype = np.int8 if signed else np.uint8
    elif total_bits <= 16:
        dtype = np.int16 if signed else np.uint16
    else:
        dtype = np.int32 if signed else np.uint32

    return quantized.astype(dtype)


# =============================================================================
# INT4 Conversion
# =============================================================================

def int4_to_int8(data: np.ndarray, signed: bool = True) -> np.ndarray:
    """
    Unpack INT4 to INT8.
    Two INT4 values packed per byte, low nibble first.
    """
    data = data.astype(np.uint8)
    low = data & 0x0F
    high = (data >> 4) & 0x0F

    if signed:
        # Sign extend
        low = np.where(low >= 8, low.astype(np.int8) - 16, low.astype(np.int8))
        high = np.where(high >= 8, high.astype(np.int8) - 16, high.astype(np.int8))

    # Interleave
    result = np.empty(data.size * 2, dtype=np.int8 if signed else np.uint8)
    result[0::2] = low.flatten()
    result[1::2] = high.flatten()

    return result


# =============================================================================
# Generic Converter
# =============================================================================

class DTypeConverter:
    """Generic data type converter."""

    def __init__(self):
        self._converters = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default converters."""
        # FP8
        self.register(HWDType.FP8_E4M3, fp8_e4m3_to_fp32)
        self.register(HWDType.FP8_E5M2, fp8_e5m2_to_fp32)

        # Standard types (passthrough or simple cast)
        self.register(HWDType.FP32, lambda x: x.astype(np.float32))
        self.register(HWDType.FP16, lambda x: x.view(np.float16).astype(np.float32))
        self.register(HWDType.BF16, self._bf16_to_fp32)
        self.register(HWDType.INT8, lambda x: x.astype(np.int8).astype(np.float32))
        self.register(HWDType.UINT8, lambda x: x.astype(np.uint8).astype(np.float32))
        self.register(HWDType.INT4, lambda x: int4_to_int8(x, signed=True).astype(np.float32))
        self.register(HWDType.UINT4, lambda x: int4_to_int8(x, signed=False).astype(np.float32))

    def _bf16_to_fp32(self, data: np.ndarray) -> np.ndarray:
        """Convert BF16 to FP32."""
        # BF16 is just FP32 with truncated mantissa
        data = data.astype(np.uint16)
        # Shift to upper 16 bits of FP32
        fp32_bits = data.astype(np.uint32) << 16
        return fp32_bits.view(np.float32)

    def register(self, dtype: HWDType, converter: Callable[[np.ndarray], np.ndarray]):
        """Register a converter for a data type."""
        self._converters[dtype] = converter

    def register_bfp(self, dtype: HWDType, config: BFPConfig):
        """Register a BFP converter with custom config."""
        def converter(data: np.ndarray, exponents: np.ndarray = None) -> np.ndarray:
            if exponents is None:
                raise ValueError(f"BFP conversion requires exponents array")
            return bfp_to_fp32(data, exponents, config)
        self._converters[dtype] = converter
        BFP_CONFIGS[dtype] = config

    def register_fxp(self, dtype: HWDType, total_bits: int, frac_bits: int, signed: bool = True):
        """Register a fixed-point converter."""
        def converter(data: np.ndarray) -> np.ndarray:
            return fxp_to_fp32(data, total_bits, frac_bits, signed)
        self._converters[dtype] = converter

    def convert(
        self,
        data: np.ndarray,
        dtype: HWDType,
        exponents: np.ndarray = None,
    ) -> np.ndarray:
        """
        Convert data from hardware dtype to FP32.

        Args:
            data: Raw data
            dtype: Hardware data type
            exponents: Shared exponents (for BFP formats)

        Returns:
            FP32 array
        """
        if dtype not in self._converters:
            raise ValueError(f"No converter registered for dtype: {dtype}")

        converter = self._converters[dtype]

        # BFP needs exponents
        if dtype in BFP_CONFIGS:
            if exponents is None:
                raise ValueError(f"BFP dtype {dtype} requires exponents")
            return converter(data, exponents)

        return converter(data)

    def to_fp32(
        self,
        data: np.ndarray,
        dtype: HWDType,
        **kwargs
    ) -> np.ndarray:
        """Alias for convert()."""
        return self.convert(data, dtype, **kwargs)


# Global converter instance
converter = DTypeConverter()


def convert_to_fp32(data: np.ndarray, dtype: HWDType, **kwargs) -> np.ndarray:
    """Convenience function for conversion."""
    return converter.convert(data, dtype, **kwargs)
