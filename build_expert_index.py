#!/usr/bin/env python3
"""Build an index of every expert tensor across safetensors shards for Qwen3.5-397B-A17B-4bit.

Parses only headers (no data read), computes absolute byte offsets for each expert slice,
and saves expert_index.json for use by the streaming inference engine.
"""

import json
import os
import struct
import time
from pathlib import Path

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit/"
    "snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3"
)

NUM_SHARDS = 46
NUM_LAYERS = 60
NUM_EXPERTS = 512

# safetensors dtype -> bytes per element
DTYPE_SIZES = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "BOOL": 1,
}

# The switch_mlp components we care about
EXPERT_COMPONENTS = [
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
]


def parse_safetensors_header(filepath: str) -> tuple[int, dict]:
    """Parse a safetensors file header. Returns (header_size, header_dict)."""
    with open(filepath, "rb") as f:
        raw = f.read(8)
        header_size = struct.unpack("<Q", raw)[0]
        header_bytes = f.read(header_size)
    header = json.loads(header_bytes)
    # Remove metadata key if present
    header.pop("__metadata__", None)
    return header_size, header


def compute_expert_stride(shape: list[int], dtype: str) -> tuple[int, list[int]]:
    """Compute bytes per expert slice.

    Expert tensors have shape [num_experts, ...rest_dims].
    Returns (stride_bytes, shape_per_expert).
    """
    if len(shape) < 2:
        raise ValueError(f"Expected at least 2D tensor for expert, got shape {shape}")

    num_experts = shape[0]
    shape_per_expert = shape[1:]
    elem_size = DTYPE_SIZES[dtype]

    # stride = product of remaining dims * elem_size
    stride = elem_size
    for d in shape_per_expert:
        stride *= d

    return stride, shape_per_expert


def build_index():
    t0 = time.perf_counter()

    model_path = MODEL_PATH
    assert os.path.isdir(model_path), f"Model not found at {model_path}"

    # Phase 1: Parse all shard headers
    print(f"Parsing {NUM_SHARDS} shard headers...")
    all_tensors = {}  # tensor_name -> {file, file_data_offset, tensor_offset, abs_offset, size_bytes, dtype, shape}
    expert_tensor_count = 0
    total_tensor_count = 0

    for shard_idx in range(1, NUM_SHARDS + 1):
        shard_name = f"model-{shard_idx:05d}-of-{NUM_SHARDS:05d}.safetensors"
        shard_path = os.path.join(model_path, shard_name)

        header_size, header = parse_safetensors_header(shard_path)
        file_data_offset = 8 + header_size  # data starts after 8-byte length + header

        for tensor_name, info in header.items():
            total_tensor_count += 1
            dtype = info["dtype"]
            shape = info["shape"]
            data_start, data_end = info["data_offsets"]
            size_bytes = data_end - data_start
            abs_offset = file_data_offset + data_start

            # Check if this is a switch_mlp expert tensor
            if ".mlp.switch_mlp." not in tensor_name:
                continue

            expert_tensor_count += 1

            # Compute expert stride
            expert_stride, shape_per_expert = compute_expert_stride(shape, dtype)
            expert_size = expert_stride  # contiguous, so size == stride

            all_tensors[tensor_name] = {
                "file": shard_name,
                "file_data_offset": file_data_offset,
                "tensor_offset": data_start,
                "abs_offset": abs_offset,
                "size_bytes": size_bytes,
                "dtype": dtype,
                "shape": shape,
                "expert_stride": expert_stride,
                "expert_size": expert_size,
            }

    t1 = time.perf_counter()
    print(f"  Parsed {total_tensor_count} total tensors in {t1-t0:.2f}s")
    print(f"  Found {expert_tensor_count} expert tensors (switch_mlp)")

    # Phase 2: Build per-layer expert_reads lookup
    print("\nBuilding per-layer expert_reads lookup...")
    expert_reads = {}

    for tensor_name, info in sorted(all_tensors.items()):
        # Extract layer number and component
        # Format: language_model.model.layers.{L}.mlp.switch_mlp.{component}
        parts = tensor_name.split(".")
        layer_idx = None
        component = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                layer_idx = int(parts[i + 1])
            if p == "switch_mlp" and i + 1 < len(parts):
                # Component is everything after switch_mlp
                component = ".".join(parts[i + 1 :])

        if layer_idx is None or component is None:
            print(f"  WARNING: Could not parse tensor name: {tensor_name}")
            continue

        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            expert_reads[layer_key] = {}

        shape_per_expert = info["shape"][1:]  # drop the num_experts dim

        expert_reads[layer_key][component] = {
            "file": info["file"],
            "abs_offset": info["abs_offset"],
            "expert_stride": info["expert_stride"],
            "expert_size": info["expert_size"],
            "dtype": info["dtype"],
            "shape_per_expert": shape_per_expert,
        }

    # Phase 3: Compute per-expert byte totals
    print("\nComputing per-expert byte breakdown...")
    per_expert_total = 0
    component_sizes = {}
    # Use layer 0 as reference
    if "0" in expert_reads:
        for comp, info in sorted(expert_reads["0"].items()):
            component_sizes[comp] = info["expert_size"]
            per_expert_total += info["expert_size"]
            print(
                f"  {comp}: {info['expert_size']:>12,} bytes  "
                f"({info['expert_size']/1024/1024:.2f} MB)  "
                f"shape={info['shape_per_expert']}  dtype={info['dtype']}"
            )
    print(f"  {'TOTAL':30s}: {per_expert_total:>12,} bytes  ({per_expert_total/1024/1024:.2f} MB)")

    # Phase 4: Build the full index
    index = {
        "model_path": model_path,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": 10,
        "per_expert_bytes": per_expert_total,
        "per_expert_bytes_mb": round(per_expert_total / 1024 / 1024, 3),
        "tensors": all_tensors,
        "expert_components": EXPERT_COMPONENTS,
        "expert_reads": expert_reads,
    }

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expert_index.json")
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)

    t2 = time.perf_counter()
    print(f"\nSaved index to {out_path}")
    print(f"  File size: {os.path.getsize(out_path) / 1024:.1f} KB")
    print(f"  Total time: {t2-t0:.2f}s")

    # Phase 5: Summary and verification
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model:              Qwen3.5-397B-A17B-4bit")
    print(f"Layers:             {NUM_LAYERS}")
    print(f"Experts per layer:  {NUM_EXPERTS}")
    print(f"Active per token:   10")
    print(f"Expert tensors:     {expert_tensor_count}")
    print(f"Layers with MoE:    {len(expert_reads)}")
    print(f"Components/layer:   {len(expert_reads.get('0', {}))}")
    print(f"Per-expert total:   {per_expert_total:,} bytes ({per_expert_total/1024/1024:.2f} MB)")
    print(f"Per-token active:   {per_expert_total * 10:,} bytes ({per_expert_total * 10 / 1024/1024:.2f} MB)")
    total_expert_bytes = per_expert_total * NUM_EXPERTS * len(expert_reads)
    print(f"Total expert data:  {total_expert_bytes:,} bytes ({total_expert_bytes / 1024**3:.2f} GB)")

    # Phase 6: Sample read operations
    print("\n" + "=" * 80)
    print("SAMPLE READ OPERATIONS")
    print("=" * 80)
    sample_pairs = [(0, 0), (0, 255), (0, 511), (30, 42), (59, 100)]
    for layer, expert in sample_pairs:
        layer_key = str(layer)
        if layer_key not in expert_reads:
            print(f"\n  Layer {layer}: no MoE (attention-only layer)")
            continue
        print(f"\n  Layer {layer}, Expert {expert}:")
        for comp in EXPERT_COMPONENTS:
            if comp not in expert_reads[layer_key]:
                continue
            r = expert_reads[layer_key][comp]
            offset = r["abs_offset"] + expert * r["expert_stride"]
            size = r["expert_size"]
            print(
                f"    {comp:25s}  file={r['file']}  "
                f"offset={offset:>15,}  size={size:>10,}  "
                f"shape={r['shape_per_expert']}"
            )

    # Phase 7: Verify a sample read actually works (read 16 bytes from one tensor)
    print("\n" + "=" * 80)
    print("VERIFICATION: Reading 16 bytes from layer 0, expert 0, gate_proj.weight")
    print("=" * 80)
    if "0" in expert_reads and "gate_proj.weight" in expert_reads["0"]:
        r = expert_reads["0"]["gate_proj.weight"]
        fpath = os.path.join(model_path, r["file"])
        offset = r["abs_offset"]
        with open(fpath, "rb") as f:
            f.seek(offset)
            data = f.read(16)
        print(f"  File: {r['file']}")
        print(f"  Offset: {offset:,}")
        print(f"  Read 16 bytes: {data.hex()}")
        print(f"  As U32 values: {struct.unpack('<4I', data)}")
        print("  OK - read succeeded")


if __name__ == "__main__":
    build_index()
