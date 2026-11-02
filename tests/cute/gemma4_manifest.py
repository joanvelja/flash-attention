"""Freeze independent public-training and private-held-out attention workloads.

This standard-library-only sampler does not inspect kernel results. Never change
an existing manifest after observing a run. The output directory must be new.
"""

import argparse
import hashlib
import json
import random
import secrets
from collections import Counter
from pathlib import Path

DIMENSIONS = ((8, 8), (64, 64), (128, 128), (192, 128), (256, 256), (512, 512), (64, 512))
LENGTH_BUCKETS = {
    "empty": (0,), "tiny": (1, 2, 3, 7, 15, 17, 31),
    "boundary": (63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257),
    "medium": (511, 513, 1023, 1024, 1025, 2047, 2049, 4093),
    "long": (8191, 16381, 32749), "64k": (65535, 65536, 65537),
    "128k": (131071, 131072),
}
HEADS = tuple((hq, hk) for hq in range(8, 65) for hk in range(1, 17)
              if hq % hk == 0 and hq // hk <= 32)
BUDGETS = {
    "max_abs": "<= 2 * bf16_pytorch_max_abs + 2 * max_abs(ref + 0.3 - 0.3 - ref)",
    "output_atol": 0.03, "output_rtol": 0.03, "lse_atol": 0.03, "lse_rtol": 0.03,
    "gradient_relative_l2_strictly_below": 0.01, "reference_rows": 1024,
    "fp32_reference": "all rows, heads, keys; no TF32; natural-log LSE",
    "baseline": "bf16 PyTorch slab/head GEMM+softmax+autograd; fp32 sum of slab/head gradients",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def shape_digest(sample):
    fields = ("dimension", "dimension_v", "query_heads", "kv_heads", "q_lengths", "k_lengths",
              "causal", "window", "dtype", "softmax_scale", "deterministic", "storage",
              "tensor_layout", "page_size", "direction", "api")
    return digest({field: sample[field] for field in fields})


def generate(seed, split, count, excluded):
    rng = random.Random(seed)
    samples = []
    identities = set(excluded)
    while len(samples) < count:
        i = len(samples)
        dim, dim_v = DIMENSIONS[i % len(DIMENSIONS)]
        if i % 8 == 7:
            dim, dim_v = rng.choice(range(8, 129, 8)), rng.choice(range(8, 129, 8))
        hq, hk = rng.choice(HEADS)
        bucket = tuple(LENGTH_BUCKETS)[(i // len(DIMENSIONS)) % len(LENGTH_BUCKETS)]
        nk = rng.choice(LENGTH_BUCKETS[bucket])
        mode = ("prefill", "decode", "rectangular")[(i // 3) % 3]
        nq = nk if mode == "prefill" else rng.choice((1, 2, 7, 17, 129)) if mode == "decode" else rng.choice(LENGTH_BUCKETS["boundary"])
        # Include full 128k while keeping its dense reference within the budget;
        # large-D/large-H 128k samples still examine every row of short-query KV.
        if bucket == "128k" and mode == "prefill" and (dim > 128 or hq > 16):
            mode, nq = "rectangular", rng.choice((127, 257, 1025))
        q_lengths, k_lengths = [nq], [nk]
        if i % 4 != 0:
            for _ in range(rng.choice((1, 2, 3))):
                q_lengths.append(rng.choice((0, 1, 63, 129, 257)))
                k_lengths.append(rng.choice((0, 1, 65, 127, 259)))
        mask = ("causal", "noncausal", "local")[(i // 2) % 3]
        window = [None, None]
        if mask == "local":
            window = [rng.choice((0, 1, 15, 63, 127, 255, 1023, 4095)), rng.choice((0, 1, 7, 31))]
        storage = ("packed", "packed", "dense", "paged")[(i // 5) % 4]
        tensor_layout = rng.choice(("contiguous", "strided", "equal_distinct", "alias"))
        if dim != dim_v and tensor_layout in ("equal_distinct", "alias"):
            tensor_layout = "strided"
        if storage != "packed":
            q_lengths, k_lengths = [nq] * rng.choice((1, 2)), [nk] * rng.choice((1, 2))
            k_lengths = [nk] * len(q_lengths)
        # Raw backward has no page_table argument. This is an inherited API
        # omission, not a success or an exclusion of another requested mode.
        direction = "forward" if storage == "paged" else "forward_backward"
        sample = {
            "id": f"{split}-{i:04d}", "dimension": dim, "dimension_v": dim_v,
            "query_heads": hq, "kv_heads": hk, "q_lengths": q_lengths, "k_lengths": k_lengths,
            "causal": mask == "causal", "window": window,
            "dtype": ("bfloat16", "float16")[(i // len(DIMENSIONS)) % 2],
            "softmax_scale": rng.choice((1.0, 1.0, dim ** -0.5, 0.37)),
            "deterministic": bool(i % 2), "storage": storage, "tensor_layout": tensor_layout,
            "page_size": rng.choice((64, 128, 256)) if storage == "paged" else None,
            "direction": direction, "expected": "pass", "input_seed": rng.randrange(2**31),
            "length_bucket": bucket, "query_mode": mode, "mask_class": mask,
            "distribution": rng.choice(("normalized", "normal", "peaky")),
            "api": "qkv", "unsupported_api": ["paged_backward"] if storage == "paged" else [],
        }
        identity = shape_digest(sample)
        if identity in identities:
            continue
        identities.add(identity)
        samples.append(sample)
    return samples, identities


def coverage(samples):
    return {
        "count": len(samples),
        "dimensions": sorted({f"{s['dimension']}/{s['dimension_v']}" for s in samples}),
        "dtypes": dict(Counter(s["dtype"] for s in samples)),
        "length_buckets": dict(Counter(s["length_bucket"] for s in samples)),
        "masks": dict(Counter(s["mask_class"] for s in samples)),
        "storage": dict(Counter(s["storage"] for s in samples)),
        "tensor_layout": dict(Counter(s["tensor_layout"] for s in samples)),
        "directions": dict(Counter(s["direction"] for s in samples)),
        "non_power_of_two_gqa_count": sum((s["query_heads"] // s["kv_heads"]) & ((s["query_heads"] // s["kv_heads"]) - 1) != 0 for s in samples),
        "query_head_range": [min(s["query_heads"] for s in samples), max(s["query_heads"] for s in samples)],
        "kv_head_range": [min(s["kv_heads"] for s in samples), max(s["kv_heads"] for s in samples)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, default=20261029)
    parser.add_argument("--count", type=int, default=196)
    args = parser.parse_args()
    assert args.count >= 98, "Each split must cover both dtypes and all seven length buckets"
    args.output.mkdir(parents=True, exist_ok=False)
    private = args.output / "private"
    private.mkdir(mode=0o700)
    train, used = generate(args.train_seed, "train", args.count, set())
    held_seed = secrets.randbits(64)
    heldout, _ = generate(held_seed, "heldout", args.count, used)
    schema = {
        "schema_version": 1, "architectures": [100, 103], "budgets": BUDGETS,
        "requested_modes": "All generated rows must pass; prototype restrictions are failures, never exclusions",
        "statuses": ["pending", "compile_error", "runtime_error", "numerical_fail", "pass", "baseline_unsupported"],
        "inherited_api_omissions": {"paged_backward": "_flash_attn_bwd has no page_table argument"},
        "heldout_policy": "No individual shapes, failures or metrics to implementers before final candidate freeze",
    }
    train_manifest = {**schema, "split": "train", "seed": args.train_seed, "samples": train}
    held_manifest = {**schema, "split": "heldout", "seed": held_seed, "samples": heldout}
    (args.output / "train.json").write_bytes(canonical(train_manifest) + b"\n")
    held_path = private / "heldout.json"
    held_path.write_bytes(canonical(held_manifest) + b"\n")
    held_path.chmod(0o400)
    summary = {"schema": schema, "train": {"sha256": digest(train_manifest), "coverage": coverage(train)},
               "heldout": {"sha256": digest(held_manifest), "coverage": coverage(heldout)}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
