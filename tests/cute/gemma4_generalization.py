"""Execute frozen Gemma/Blackwell manifests without post-hoc exclusions.

CLI: uv run python tests/cute/gemma4_generalization.py --manifest PATH --ids ID... --output DIR
Each selected row writes a permanent source-stamped rung result. Failures remain
failures; unsupported prototype modes are never converted to skips.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from test_gemma4_sm100 import (
    Rungs,
    check_reference,
    check_result,
    cu_seqlens,
    dense_reference,
    environment,
)


def source_provenance():
    from flash_attn.cute import interface
    root = Path(interface.__file__).resolve().parents[1]
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted((root / "cute").rglob("*.py"))}
    return {"interface_path": str(Path(interface.__file__).resolve()),
            "package_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            "sources": sources}


def check_general_reference(device="cpu"):
    """Independent full-matrix graph checks Dv, MLA, scale and both mask edges."""
    torch.manual_seed(914)
    for causal in (False, True):
        for window in ((None, None), (2, 1)):
            for mla in (False, True):
                q = torch.randn(7, 6, 8, device=device, requires_grad=True)
                k = torch.randn(4, 2, 8, device=device, requires_grad=True)
                v = torch.randn(4, 2, 16, device=device, requires_grad=True)
                qv = torch.randn(7, 6, 16, device=device, requires_grad=True) if mla else None
                do = torch.randn(7, 6, 16, device=device)
                score = torch.einsum("ihd,jhd->hij", q, k.repeat_interleave(3, 1))
                if mla:
                    score = score + torch.einsum("ihd,jhd->hij", qv, v.repeat_interleave(3, 1))
                mask = torch.tensor([[((not causal) or j <= i - 3)
                                      and (window[0] is None or j >= i - 3 - window[0])
                                      and (window[1] is None or j <= i - 3 + window[1])
                                      for j in range(4)] for i in range(7)], device=device)
                score = (score * 0.37).masked_fill(~mask, -math.inf)
                valid = mask.any(-1)
                prob = torch.where(valid[None, :, None], score, 0).softmax(-1).masked_fill(~valid[None, :, None], 0)
                out = torch.einsum("hij,jhd->ihd", prob, v.repeat_interleave(3, 1))
                inputs = (q, k, v, qv) if mla else (q, k, v)
                grads = torch.autograd.grad(out, inputs, do)
                expected = {"O": out, "LSE": torch.logsumexp(score, -1),
                            **dict(zip(("dQ", "dK", "dV", "dQv"), grads))}
                actual = dense_reference(q, k, v, do, [7], [4], window[0], dtype=torch.float32,
                                         rows=3, causal=causal, window_right=window[1], softmax_scale=0.37, qv=qv)
                for name, expected_tensor in expected.items():
                    torch.testing.assert_close(actual[name], expected_tensor, atol=2e-5, rtol=2e-5)
    return {"cases": 8, "device": device}


def packed(tensor):
    return tensor.flatten(0, 1) if tensor.ndim == 4 else tensor


def effective(tensor, capacities, used):
    parts, start = [], 0
    for capacity, length in zip(capacities, used):
        parts.append(tensor[start:start + length])
        start += capacity
    return torch.cat(parts, 0)


def scatter_effective(tensor, capacities, used, *, lse=False):
    if lse:
        tensor = tensor.T
    shape = (sum(capacities), *tensor.shape[1:])
    result = torch.full(shape, -math.inf if lse else 0.0, dtype=tensor.dtype, device=tensor.device)
    source = target = 0
    for capacity, length in zip(capacities, used):
        result[target:target + length] = tensor[source:source + length]
        source += length
        target += capacity
    return result.T if lse else result


def used_results(sample, result):
    """Read-only judging selection; fully used canonical tensors may alias inputs."""
    selected = {}
    for name, tensor in result.items():
        side = "k" if name in ("dK", "dV", "dKV") else "q"
        capacities = sample[f"{side}_lengths"]
        used = sample.get(f"seqused_{side}", capacities)
        logical = tensor.T if name == "LSE" else tensor
        canonical, stride = [], 1
        for extent in reversed(logical.shape):
            canonical.append(stride)
            stride *= max(extent, 1)
        identity = (bool(capacities) and list(capacities) == list(used)
                    and sum(capacities) == logical.shape[0]
                    and logical.stride() == tuple(reversed(canonical)))
        selected[name] = logical if identity else effective(logical, capacities, used)
        if name == "LSE":
            selected[name] = selected[name].T
    return selected


def make_workload(sample):
    torch.manual_seed(sample["input_seed"])
    dtype = getattr(torch, sample["dtype"])
    qlens, klens = sample["q_lengths"], sample["k_lengths"]
    storage = sample["storage"]
    qdense = storage in ("dense", "dense_q_packed_k")
    kdense = storage in ("dense", "packed_q_dense_k")
    def make(lengths, heads, dimension, dense):
        shape = (len(lengths), lengths[0], heads, dimension) if dense else (sum(lengths), heads, dimension)
        tensor = torch.randn(shape, device="cuda", dtype=torch.float32)
        if sample["distribution"] == "normalized":
            tensor *= torch.rsqrt(tensor.square().mean(-1, keepdim=True) + 1e-6)
        elif sample["distribution"] == "peaky":
            tensor *= 4
        tensor = tensor.to(dtype)
        if sample["tensor_layout"] == "strided":
            backing = torch.empty((*shape[:-1], dimension * 2), device="cuda", dtype=dtype)
            backing[..., :dimension].copy_(tensor)
            tensor = backing[..., :dimension]
        return tensor.detach().requires_grad_(sample["direction"] != "forward")
    q = make(qlens, sample["query_heads"], sample["dimension"], qdense)
    k = make(klens, sample["kv_heads"], sample["dimension"], kdense)
    v = make(klens, sample["kv_heads"], sample["dimension_v"], kdense)
    if sample["tensor_layout"] == "alias":
        v = k
    elif sample["tensor_layout"] == "equal_distinct":
        v = k.detach().clone().requires_grad_(sample["direction"] != "forward")
    qv = make(qlens, sample["query_heads"], sample["dimension_v"], qdense) if sample["api"] == "mla" else None
    dout = torch.randn((*q.shape[:-1], sample["dimension_v"]), device="cuda", dtype=dtype)
    kwargs = {"cu_seqlens_q": None if qdense else cu_seqlens(qlens),
                  "cu_seqlens_k": None if kdense else cu_seqlens(klens),
                  "max_seqlen_q": max(qlens), "max_seqlen_k": max(klens),
                  "seqused_q": torch.tensor(sample["seqused_q"], device="cuda", dtype=torch.int32) if "seqused_q" in sample else None,
                  "seqused_k": torch.tensor(sample["seqused_k"], device="cuda", dtype=torch.int32) if "seqused_k" in sample else None,
                  "softmax_scale": sample["softmax_scale"], "causal": sample["causal"]}
    logical = (packed(q), packed(k), packed(v), packed(dout), packed(qv) if qv is not None else None)
    if storage == "paged":
        page = sample["page_size"]
        pages_per_seq = max(1, math.ceil(max(klens) / page))
        table = torch.randperm(len(klens) * pages_per_seq, device="cuda", dtype=torch.int32).reshape(len(klens), pages_per_seq)
        def page_tensor(tensor):
            # The public paged API requires finite padding in partially used
            # TMA pages; unused-page poison has a separate isolation test.
            result = torch.zeros((len(klens) * pages_per_seq, page, *tensor.shape[1:]), device="cuda", dtype=dtype)
            offset = 0
            for batch, length in enumerate(klens):
                for start in range(0, length, page):
                    stop = min(length, start + page)
                    result[table[batch, start // page].item(), :stop-start] = tensor[offset+start:offset+stop]
                offset += length
            if sample["tensor_layout"] == "strided":
                backing = torch.empty((*result.shape[:-1], result.shape[-1] * 2), device="cuda", dtype=dtype)
                backing[..., :result.shape[-1]].copy_(result)
                result = backing[..., :result.shape[-1]]
            return result
        k = page_tensor(logical[1])
        v = k if sample["tensor_layout"] == "alias" else page_tensor(logical[2])
        kwargs.update(cu_seqlens_k=None, page_table=table,
                      seqused_k=torch.tensor(klens, device="cuda", dtype=torch.int32))
    return (q, k, v, qv, dout), logical, kwargs


def execute(sample, tensors, kwargs):
    from flash_attn.cute.interface import (
        _flash_attn_bwd,
        _flash_attn_fwd,
        flash_attn_varlen_func,
    )
    q, k, v, qv, dout = tensors
    forward_only = sample["direction"] == "forward"
    public = sample["api"] == "mla" or sample["tensor_layout"] == "alias"
    if public:
        out, lse = flash_attn_varlen_func(q, k, v, qv=qv, **kwargs, window_size=tuple(sample["window"]),
                                        deterministic=sample["deterministic"], return_lse=True)
    else:
        out, lse, _, _, _ = _flash_attn_fwd(q, k, v, **kwargs, window_size_left=sample["window"][0],
                                          window_size_right=sample["window"][1], return_lse=True)
    if qv is not None:
        lse = lse.transpose(-1, -2)
    result = {"O": packed(out), "LSE": lse.permute(1, 0, 2).flatten(1) if lse.ndim == 3 else lse}
    if not forward_only:
        if public:
            inputs = (q, k) if v is k else (q, k, v)
            names = ["dQ", "dKV"] if v is k else ["dQ", "dK", "dV"]
            if qv is not None:
                inputs += (qv,)
                names.append("dQv")
            grads = torch.autograd.grad(out, inputs, dout)
        else:
            grads = _flash_attn_bwd(q, k, v, out, dout, lse, **kwargs,
                                   window_size_left=sample["window"][0], window_size_right=sample["window"][1],
                                   deterministic=sample["deterministic"])
            names = ["dQ", "dK", "dV"]
        result.update({name: packed(grad) for name, grad in zip(names, grads)})
    return result


def reference(sample, logical, dtype):
    q, k, v, dout, qv = logical
    qlens, klens = sample["q_lengths"], sample["k_lengths"]
    qused, kused = sample.get("seqused_q", qlens), sample.get("seqused_k", klens)
    q, dout = (effective(x, qlens, qused) for x in (q, dout))
    k, v = (effective(x, klens, kused) for x in (k, v))
    if qv is not None:
        qv = effective(qv, qlens, qused)
    result = dense_reference(q, k, v, dout, qused, kused, sample["window"][0], dtype=dtype,
                             causal=sample["causal"], window_right=sample["window"][1],
                             softmax_scale=sample["softmax_scale"], qv=qv)
    if sample["tensor_layout"] == "alias":
        result["dKV"] = result.pop("dK") + result.pop("dV")
    if sample["direction"] == "forward":
        result = {name: result[name] for name in ("O", "LSE")}
    return {name: scatter_effective(tensor, klens if name in ("dK", "dV", "dKV") else qlens,
                                    kused if name in ("dK", "dV", "dKV") else qused, lse=name == "LSE")
            for name, tensor in result.items()}


def run_sample(sample):
    rungs = Rungs(sample["id"])
    requested_sample = sample
    inherited_unsupported = []
    if sample["api"] == "mla" and sample["direction"] != "forward":
        # Public qv backward routes to sparse MLA, whose API requires gather
        # indices. Dense MLA backward was never implemented by this package.
        inherited_unsupported.append("dense_mla_backward_requires_gather_kv_indices")
        sample = {**sample, "direction": "forward"}
    try:
        with rungs.rung("environment") as result:
            result.update(environment(), **source_provenance(), sample=requested_sample,
                          executed_direction=sample["direction"], inherited_unsupported=inherited_unsupported,
                          baseline_dtype=sample["dtype"])
        with rungs.rung("inputs"):
            tensors, logical, kwargs = make_workload(sample)
        with rungs.rung("fp32-reference"):
            ref = reference(sample, logical, torch.float32)
        with rungs.rung("matching-dtype-baseline"):
            baseline = reference(sample, logical, getattr(torch, sample["dtype"]))
        with rungs.rung("compile-and-run"):
            actual = execute(sample, tensors, kwargs)
            torch.cuda.synchronize()
        with rungs.rung("correctness") as result:
            check_result(used_results(sample, actual), used_results(sample, ref), used_results(sample, baseline), result)
        if sample["deterministic"]:
            with rungs.rung("determinism"):
                again = execute(sample, tensors, kwargs)
                first_used, again_used = used_results(sample, actual), used_results(sample, again)
                for name in first_used:
                    assert torch.equal(first_used[name], again_used[name]), f"{name} deterministic replay differs"
        if "seqused_q" in sample or "seqused_k" in sample:
            with rungs.rung("unused-tail-poison"):
                with torch.no_grad():
                    for index, tensor in enumerate(tensors):
                        if tensor is None:
                            continue
                        side = "k" if index in (1, 2) else "q"
                        capacities = sample[f"{side}_lengths"]
                        used = sample.get(f"seqused_{side}", capacities)
                        offset = 0
                        for capacity, length in zip(capacities, used):
                            packed(tensor)[offset + length:offset + capacity].fill_(math.nan)
                            offset += capacity
                poisoned = execute(sample, tensors, kwargs)
                # Use the same frozen numerical criteria when deterministic
                # replay is not requested; do not demand undefined scheduling.
                check_result(used_results(sample, poisoned), used_results(sample, ref),
                             used_results(sample, baseline), {})
        rungs.results["status"] = "partial_api_unsupported" if inherited_unsupported else "pass"
        return rungs.results
    except Exception as exc:
        last = next(reversed(rungs.results["rungs"]))
        compile_failure = type(exc).__module__.startswith("cutlass") and any(
            word in type(exc).__name__ for word in ("Compile", "DSL", "MLIR"))
        rungs.results["status"] = ("numerical_fail" if last in ("correctness", "determinism", "unused-tail-poison")
                                   else "compile_error" if compile_failure else "runtime_error")
        raise
    finally:
        rungs.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["FA4_GEMMA_RESULTS_DIR"] = str(args.output)
    manifest = json.loads(args.manifest.read_text())
    selected = [s for s in manifest["samples"] if args.ids is None or s["id"] in args.ids]
    assert selected, "No manifest rows selected"
    if args.ids:
        assert {s["id"] for s in selected} == set(args.ids), "Unknown requested IDs"
    check_reference("cuda")
    check_general_reference("cuda")
    failures = []
    for sample in selected:
        try:
            run_sample(sample)
        except Exception as exc:  # noqa: BLE001 - preserve every failed row and fail the process
            failures.append({"id": sample["id"], "error": f"{type(exc).__name__}: {exc}"})
    print("failures = " + json.dumps(failures), flush=True)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
