# -*- coding: utf-8 -*-
"""
Example: Hardware trace capture and comparison workflow.

Workflow:
    1. Generate C code for trace capture from ONNX model
    2. (Hardware side) Capture trace data into binary file
    3. Load trace binary and compare with ONNX Runtime reference
    4. Visualize comparison results

Usage:
    # Step 1: Generate C code
    python examples/trace_compare.py gen model.onnx --output ./generated

    # Step 2: (After hardware capture) Compare with reference
    python examples/trace_compare.py compare model.onnx trace.bin --input data.npz

    # Step 3: Visualize
    python examples/trace_compare.py visualize model.onnx trace.bin
"""

import sys
sys.path.insert(0, ".")

import argparse
import numpy as np
import torch
from pathlib import Path


def cmd_generate(args):
    """Generate C code for trace capture."""
    from onnxparser.trace import generate_trace_code, TraceFormat

    print(f"Generating trace code for: {args.onnx}")
    files = generate_trace_code(
        args.onnx,
        output_dir=args.output,
        prefix=args.prefix,
    )

    # Print buffer size info
    trace_format = TraceFormat.from_onnx(args.onnx)
    total_size = trace_format.data_offset
    for t in trace_format.tensors:
        total_size += t.data_size

    print(f"\nTrace buffer info:")
    print(f"  Tensors: {len(trace_format.tensors)}")
    print(f"  Header size: {trace_format.data_offset} bytes")
    print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")

    print(f"\nGenerated files in: {args.output}")
    for name, path in files.items():
        print(f"  {Path(path).name}")


def cmd_compare(args):
    """Compare trace with ONNX Runtime reference."""
    from onnxparser.trace import (
        load_trace, compare_traces, print_compare_report,
        HWDType, DTypeConverter
    )
    from examples.visualize_onnx import export_onnx_runtime_trace, load_input_from_file

    print(f"Loading trace: {args.trace}")
    trace = load_trace(args.trace)
    print(trace.summary())

    # Load input data
    if args.input:
        print(f"\nLoading input data: {args.input}")
        input_data = load_input_from_file(args.input)
    else:
        # Generate random input
        from examples.visualize_onnx import get_onnx_input_info, generate_sample_input
        input_info = get_onnx_input_info(args.onnx)
        input_data = generate_sample_input(input_info, "random")

    # Run ONNX Runtime to get reference
    print(f"\nRunning ONNX Runtime for reference...")
    reference = export_onnx_runtime_trace(args.onnx, input_data)

    # Compare
    print(f"\nComparing {len(trace.names)} tensors...")
    results = compare_traces(trace, reference, atol=args.atol, rtol=args.rtol)
    print_compare_report(results)

    # Summary stats
    pass_count = sum(1 for r in results if r.status == "PASS")
    total = len(results)
    print(f"\nOverall: {pass_count}/{total} passed ({100*pass_count/total:.1f}%)")


def cmd_visualize(args):
    """Visualize trace data."""
    from onnxparser.trace import load_trace
    from examples.visualize_onnx import visualize_with_trace_data

    print(f"Loading trace: {args.trace}")
    trace = load_trace(args.trace)

    # Convert to dict for visualization
    trace_dict = trace.to_dict(valid_only=True)
    print(f"Loaded {len(trace_dict)} tensors for visualization")

    # Visualize
    visualize_with_trace_data(
        args.onnx,
        trace_dict,
        port=args.port,
        open_browser=True,
    )


def cmd_create_test(args):
    """Create a test trace binary for testing."""
    from onnxparser.trace import TraceFormat
    from examples.visualize_onnx import (
        export_onnx_runtime_trace,
        get_onnx_input_info,
        generate_sample_input,
        load_input_from_file,
    )

    print(f"Creating test trace for: {args.onnx}")

    # Load or generate input
    if args.input:
        input_data = load_input_from_file(args.input)
    else:
        input_info = get_onnx_input_info(args.onnx)
        input_data = generate_sample_input(input_info, "random")

    # Get reference data from ONNX Runtime
    print("Running ONNX Runtime...")
    reference = export_onnx_runtime_trace(args.onnx, input_data)

    # Create trace format
    trace_format = TraceFormat.from_onnx(args.onnx)

    # Write binary
    output = args.output or "test_trace.bin"
    with open(output, "wb") as f:
        # Update headers with valid flags
        for hdr in trace_format.tensors:
            if hdr.name in reference:
                hdr.valid = 1
            else:
                hdr.valid = 0

        # Write headers
        f.write(trace_format.pack_headers())

        # Write data
        for hdr in trace_format.tensors:
            if hdr.name in reference:
                data = reference[hdr.name]
                if isinstance(data, np.ndarray):
                    f.write(data.tobytes())
                else:
                    f.write(np.array(data).tobytes())
            else:
                f.write(b'\x00' * hdr.data_size)

    print(f"Created test trace: {output}")
    print(f"Size: {Path(output).stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Hardware trace comparison tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command
    gen_parser = subparsers.add_parser("gen", help="Generate C code for trace capture")
    gen_parser.add_argument("onnx", help="ONNX model path")
    gen_parser.add_argument("--output", "-o", default="./generated", help="Output directory")
    gen_parser.add_argument("--prefix", default="trace", help="File prefix")

    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare trace with reference")
    cmp_parser.add_argument("onnx", help="ONNX model path")
    cmp_parser.add_argument("trace", help="Trace binary path")
    cmp_parser.add_argument("--input", "-i", help="Input data file (.npz/.pt)")
    cmp_parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    cmp_parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance")

    # Visualize command
    vis_parser = subparsers.add_parser("visualize", help="Visualize trace data")
    vis_parser.add_argument("onnx", help="ONNX model path")
    vis_parser.add_argument("trace", help="Trace binary path")
    vis_parser.add_argument("--port", type=int, default=8080, help="Server port")

    # Create test trace command
    test_parser = subparsers.add_parser("create-test", help="Create test trace from ONNX Runtime")
    test_parser.add_argument("onnx", help="ONNX model path")
    test_parser.add_argument("--input", "-i", help="Input data file")
    test_parser.add_argument("--output", "-o", help="Output trace file")

    args = parser.parse_args()

    if args.command == "gen":
        cmd_generate(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "create-test":
        cmd_create_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
