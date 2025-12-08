# -*- coding: utf-8 -*-
"""Compare trace data with reference."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .loader import TraceLoader


@dataclass
class CompareResult:
    """Result of comparing two tensors."""
    name: str
    shape_match: bool
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float
    cosine_sim: float
    mismatch_count: int  # Number of elements exceeding threshold
    total_elements: int
    status: str  # "PASS", "FAIL", "MISSING"


def compare_tensors(
    actual: np.ndarray,
    expected: np.ndarray,
    name: str = "",
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> CompareResult:
    """
    Compare two tensors.

    Args:
        actual: Tensor from hardware trace
        expected: Reference tensor (e.g., from ONNX Runtime)
        name: Tensor name for reporting
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        CompareResult with detailed metrics
    """
    # Shape check
    if actual.shape != expected.shape:
        return CompareResult(
            name=name,
            shape_match=False,
            max_abs_diff=float('inf'),
            max_rel_diff=float('inf'),
            mean_abs_diff=float('inf'),
            cosine_sim=0.0,
            mismatch_count=actual.size,
            total_elements=actual.size,
            status="FAIL"
        )

    # Flatten for comparison
    a = actual.flatten().astype(np.float64)
    e = expected.flatten().astype(np.float64)

    # Absolute difference
    abs_diff = np.abs(a - e)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    # Relative difference (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = abs_diff / (np.abs(e) + 1e-10)
        rel_diff = np.nan_to_num(rel_diff, nan=0.0, posinf=0.0, neginf=0.0)
    max_rel_diff = float(np.max(rel_diff))

    # Cosine similarity
    norm_a = np.linalg.norm(a)
    norm_e = np.linalg.norm(e)
    if norm_a > 0 and norm_e > 0:
        cosine_sim = float(np.dot(a, e) / (norm_a * norm_e))
    else:
        cosine_sim = 1.0 if np.allclose(a, e) else 0.0

    # Count mismatches
    threshold = atol + rtol * np.abs(e)
    mismatch_mask = abs_diff > threshold
    mismatch_count = int(np.sum(mismatch_mask))

    # Determine status
    if mismatch_count == 0:
        status = "PASS"
    elif mismatch_count / len(a) < 0.01:  # <1% mismatch
        status = "WARN"
    else:
        status = "FAIL"

    return CompareResult(
        name=name,
        shape_match=True,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        mean_abs_diff=mean_abs_diff,
        cosine_sim=cosine_sim,
        mismatch_count=mismatch_count,
        total_elements=len(a),
        status=status
    )


def compare_traces(
    trace: TraceLoader,
    reference: Dict[str, np.ndarray],
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> List[CompareResult]:
    """
    Compare trace data with reference.

    Args:
        trace: TraceLoader with hardware captured data
        reference: Dict of reference tensors (e.g., from ONNX Runtime)
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        List of CompareResult for each tensor
    """
    results = []

    for name in trace.names:
        hdr = trace.get_header(name)

        # Check if reference exists
        if name not in reference:
            results.append(CompareResult(
                name=name,
                shape_match=False,
                max_abs_diff=float('inf'),
                max_rel_diff=float('inf'),
                mean_abs_diff=float('inf'),
                cosine_sim=0.0,
                mismatch_count=0,
                total_elements=0,
                status="MISSING_REF"
            ))
            continue

        # Check if trace data was captured
        if not hdr.valid:
            results.append(CompareResult(
                name=name,
                shape_match=False,
                max_abs_diff=float('inf'),
                max_rel_diff=float('inf'),
                mean_abs_diff=float('inf'),
                cosine_sim=0.0,
                mismatch_count=0,
                total_elements=0,
                status="NOT_CAPTURED"
            ))
            continue

        # Compare
        actual = trace[name]
        expected = reference[name]
        result = compare_tensors(actual, expected, name, atol, rtol)
        results.append(result)

    return results


def print_compare_report(results: List[CompareResult]) -> None:
    """Print comparison report."""
    print("\n" + "=" * 80)
    print("TRACE COMPARISON REPORT")
    print("=" * 80)

    pass_count = sum(1 for r in results if r.status == "PASS")
    warn_count = sum(1 for r in results if r.status == "WARN")
    fail_count = sum(1 for r in results if r.status == "FAIL")
    missing_count = sum(1 for r in results if r.status in ("MISSING_REF", "NOT_CAPTURED"))

    print(f"\nSummary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL, {missing_count} MISSING")
    print("-" * 80)
    print(f"{'Tensor':<40} {'Status':<12} {'MaxAbsDiff':<12} {'CosineSim':<10}")
    print("-" * 80)

    for r in results:
        if r.status in ("MISSING_REF", "NOT_CAPTURED"):
            print(f"{r.name:<40} {r.status:<12} {'N/A':<12} {'N/A':<10}")
        else:
            print(f"{r.name:<40} {r.status:<12} {r.max_abs_diff:<12.6e} {r.cosine_sim:<10.6f}")

    print("=" * 80)

    # Print failures in detail
    failures = [r for r in results if r.status == "FAIL"]
    if failures:
        print("\nFAILED TENSORS DETAIL:")
        for r in failures:
            print(f"\n  {r.name}:")
            print(f"    Max Abs Diff: {r.max_abs_diff:.6e}")
            print(f"    Max Rel Diff: {r.max_rel_diff:.6e}")
            print(f"    Mean Abs Diff: {r.mean_abs_diff:.6e}")
            print(f"    Cosine Sim: {r.cosine_sim:.6f}")
            print(f"    Mismatches: {r.mismatch_count}/{r.total_elements} ({100*r.mismatch_count/r.total_elements:.2f}%)")
