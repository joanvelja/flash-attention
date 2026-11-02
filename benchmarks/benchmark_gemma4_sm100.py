"""Correctness-gated Gemma 4 timings, with explicit SDPA backend selection.

Example: uv run benchmarks/benchmark_gemma4_sm100.py --case global12 --pack short
         --peak-tflops <dense-bf16-peak-for-this-GPU> --json tmp/bench.json
Peak is mandatory: do not substitute a sparse-tensor-core marketing number.
Every invocation validates the target and the D128 anchor before timing either.
For the frozen general shape matrix, isolated pin/upstream comparisons, and
alternating query-tail overhead rounds, use benchmark_sm100_matrix.py.
"""

import argparse
import json
import statistics
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "cute"))
from benchmark_sm100_matrix import clocks
from gemma4_generalization import source_provenance
from test_gemma4_sm100 import (
    CASES,
    PACKS,
    Rungs,
    check_result,
    cu_seqlens,
    run_case,
)


def flop_counts(case, lengths):
    pairs = sum(sum(min(i + 1, case.window_left + 1) if case.window_left is not None else i + 1
                    for i in range(length)) for length in lengths)
    # QK + PV: 4D per pair/head; bwd recomputes QK plus dV,dP,dS*K,dS^T*Q: 10D.
    return {"fwd": 4 * case.heads * case.dim * pairs,
            "bwd": 10 * case.heads * case.dim * pairs,
            "total": 14 * case.heads * case.dim * pairs}


def timed(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    durations = []
    walls = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        walls.append((time.perf_counter() - wall) * 1000)
        durations.append(start.elapsed_time(end))
    return {"ms": statistics.median(durations), "min_ms": min(durations), "max_ms": max(durations),
            "wall_ms": statistics.median(walls), "all_ms": durations, "all_wall_ms": walls}


def flash_calls(case, tensors, lengths):
    """Precompute metadata, as the trainer does, outside the timed region."""
    from flash_attn.cute import flash_attn_varlen_func
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    q, k, v, dout = tensors
    cu = cu_seqlens(lengths)
    common = {"cu_seqlens_q": cu, "cu_seqlens_k": cu, "max_seqlen_q": max(lengths),
              "max_seqlen_k": max(lengths), "causal": True, "softmax_scale": 1.0}
    if case.shared:
        def forward():
            return flash_attn_varlen_func(q, k, v, **common, window_size=(case.window_left, 0), return_lse=True)

        held, _ = forward()

        def backward():
            return torch.autograd.grad(held, (q, k), dout, retain_graph=True)

        def total():
            out, _ = forward()
            return torch.autograd.grad(out, (q, k), dout)
    else:
        common.update(window_size_left=case.window_left, window_size_right=0)

        def forward():
            out, lse, _, _, _ = _flash_attn_fwd(q, k, v, **common, return_lse=True)
            return out, lse

        held, lse = forward()

        def backward():
            return _flash_attn_bwd(q, k, v, held, dout, lse, **common)

        def total():
            out, new_lse = forward()
            return _flash_attn_bwd(q, k, v, out, dout, new_lse, **common)
    return {"fwd": forward, "bwd": backward, "total": total}


def sdpa_calls(case, tensors, lengths, backend):
    q, k, v, dout = tensors
    inputs = (q, k) if case.shared else (q, k, v)
    masks = []
    for length in lengths:
        if case.window_left is None:
            masks.append(None)
        else:
            row = torch.arange(length, device=q.device)[:, None]
            col = torch.arange(length, device=q.device)[None, :]
            masks.append((col <= row) & (col >= row - case.window_left))

    def forward():
        parts = []
        start = 0
        with sdpa_kernel(backend):
            for length, mask in zip(lengths, masks):
                stop = start + length
                parts.append(F.scaled_dot_product_attention(
                    q[start:stop].transpose(0, 1).unsqueeze(0),
                    k[start:stop].transpose(0, 1).unsqueeze(0),
                    v[start:stop].transpose(0, 1).unsqueeze(0),
                    attn_mask=mask, is_causal=mask is None, dropout_p=0.0, scale=1.0,
                    enable_gqa=case.heads != case.kv_heads,
                ).squeeze(0).transpose(0, 1))
                start = stop
        return torch.cat(parts)

    held = forward()

    def backward():
        return torch.autograd.grad(held, inputs, dout, retain_graph=True)

    def total():
        return torch.autograd.grad(forward(), inputs, dout)

    gradients = backward()
    actual = {"O": held.detach(), "dQ": gradients[0]}
    if case.shared:
        actual["dKV"] = gradients[1]
    else:
        actual.update(dK=gradients[1], dV=gradients[2])
    return {"fwd": forward, "bwd": backward, "total": total}, actual


def benchmark_calls(rungs, name, calls, flops, peak, warmup, repeats):
    for direction, fn in calls.items():
        with rungs.rung(f"{name}-{direction}") as result:
            result.update(timed(fn, warmup, repeats))
            result["flops"] = flops[direction]
            result["tflops"] = flops[direction] / (result["ms"] * 1e9)
            result["dense_bf16_peak_tflops"] = peak
            result["percent_dense_bf16_peak"] = 100 * result["tflops"] / peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--pack", choices=PACKS, required=True)
    parser.add_argument("--peak-tflops", type=float, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    assert args.peak_tflops > 0 and args.repeats > 0 and args.warmup >= 0
    case, lengths = CASES[args.case], PACKS[args.pack]
    rungs = Rungs(f"benchmark-{case.name}-{args.pack}")
    report = {"case": case.name, "pack": args.pack, "peak_tflops": args.peak_tflops,
              "source_before": source_provenance(), "clocks_before": clocks()}
    try:
        validation, tensors, reference, baseline = run_case(
            case, lengths, return_artifacts=True, name=f"bench-validation-{case.name}-{args.pack}",
        )
        report["validation"] = validation
        flops = flop_counts(case, lengths)
        benchmark_calls(rungs, "fa4", flash_calls(case, tensors, lengths), flops,
                        args.peak_tflops, args.warmup, args.repeats)
        for name, backend in (("sdpa-cudnn", SDPBackend.CUDNN_ATTENTION),
                              ("sdpa-flash", SDPBackend.FLASH_ATTENTION)):
            # A backend's explicit rejection is a benchmark result, not a switch
            # to math SDPA. Numerical failures still fail the entire command.
            try:
                calls, actual = sdpa_calls(case, tensors, lengths, backend)
                torch.cuda.synchronize()
            except (RuntimeError, NotImplementedError) as exc:
                explicit_rejections = (
                    "No available kernel", "No viable backend", "No suitable kernel",
                    "does not support", "not supported", "not implemented for",
                )
                if not any(message in str(exc) for message in explicit_rejections):
                    raise
                traceback.print_exc()
                result = {"status": "UNSUPPORTED", "error": f"{type(exc).__name__}: {exc}"}
                report[name] = result
                print(f"[{name}] UNSUPPORTED {json.dumps(result)}", flush=True)
                continue
            with rungs.rung(f"{name}-correctness") as result:
                check_result(actual, {key: reference[key] for key in actual},
                             {key: baseline[key] for key in actual}, result)
            benchmark_calls(rungs, name, calls, flops, args.peak_tflops, args.warmup, args.repeats)
        del tensors, reference, baseline
        anchor = replace(case, name=case.name + "-hd128-anchor", dim=128)
        validation, tensors, _reference, _baseline = run_case(
            anchor, lengths, return_artifacts=True, name=f"bench-anchor-{case.name}-{args.pack}",
        )
        report["anchor_validation"] = validation
        benchmark_calls(rungs, "fa4-hd128", flash_calls(anchor, tensors, lengths), flop_counts(anchor, lengths),
                        args.peak_tflops, args.warmup, args.repeats)
    finally:
        report["source_after"] = source_provenance()
        report["source_unchanged"] = report["source_after"] == report["source_before"]
        report["clocks_after"] = clocks()
        report["benchmark"] = rungs.results
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        rungs.finish()
        assert report["source_unchanged"], "FA4 sources changed during correctness/timing; timings invalid"


if __name__ == "__main__":
    main()
