"""Gemma 4's actual packed attention contract on SM100 and SM103.

Run with ``uv run pytest -s tests/cute/test_gemma4_sm100.py``. Select a bounded
batch with e.g. ``-k 'short and global12'``. FA4_GEMMA_EXPECT_ARCH=100 or 103
guards the hardware; FA4_GEMMA_RESULTS_DIR saves every rung, including failures.
The fp32 and bf16 references use the same fixed 1024-row decomposition, every
query head and every key. Neither reference calls SDPA or FlashAttention.
"""

import contextlib
import functools
import importlib.metadata
import itertools
import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu
USE_FAKE_TENSOR = os.environ.get("FLASH_ATTENTION_FAKE_TENSOR", "0") == "1"
REFERENCE_ROWS = 1024  # Part of the frozen bf16 baseline, not a tuning knob.


@dataclass(frozen=True)
class Case:
    name: str
    heads: int
    kv_heads: int
    dim: int
    window_left: int | None
    shared: bool = False


CASES = {
    case.name: case
    for case in (
        Case("global12", 16, 1, 512, None),
        Case("global26", 16, 2, 512, None),
        Case("global31", 32, 4, 512, None),
        Case("sliding12_26", 16, 8, 256, 1023),
        Case("sliding31", 32, 16, 256, 1023),
        Case("shared12", 16, 1, 512, None, True),
        Case("shared26", 16, 2, 512, None, True),
        Case("shared31", 32, 4, 512, None, True),
    )
}
PACKS = {
    "short": (128, 129, 257, 511, 1057),
    "medium": (2049, 4093, 8191),
    "long": (16381, 32749),
    "64k": (65536, 129),
}


def compilation_pass(fn):
    """Delay the FA import so the reference can also be checked on a CPU host."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        from flash_attn.cute.testing import maybe_fake_tensor_mode

        return maybe_fake_tensor_mode(USE_FAKE_TENSOR)(fn)(*args, **kwargs)

    return wrapped


def is_fake_mode():
    return torch._guards.active_fake_mode() is not None


class Rungs:
    def __init__(self, name):
        self.name = name
        self.results = {"name": name, "rungs": {}}

    @contextlib.contextmanager
    def rung(self, name):
        result = {}
        start = time.perf_counter()
        try:
            yield result
        except Exception as exc:
            result.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise
        else:
            result["ok"] = True
        finally:
            result["seconds"] = time.perf_counter() - start
            self.results["rungs"][name] = result
            print(f"[{name}] {'ok' if result['ok'] else 'FAILED'} {json.dumps(result)}", flush=True)
            self.save()

    def save(self):
        directory = os.environ.get("FA4_GEMMA_RESULTS_DIR")
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{self.name}.json").write_text(json.dumps(self.results, indent=2) + "\n")

    def finish(self):
        print("results = " + json.dumps(self.results, sort_keys=True), flush=True)
        self.save()


def environment():
    assert torch.cuda.is_available(), "Gemma 4 Blackwell verification requires CUDA"
    capability = torch.cuda.get_device_capability()
    assert capability in ((10, 0), (10, 3)), f"Expected SM100 or SM103, got {capability}"
    arch = capability[0] * 10 + capability[1]
    expected = os.environ.get("FA4_GEMMA_EXPECT_ARCH")
    if expected is not None:
        assert arch == int(expected), f"Expected SM{expected}, got SM{arch}"
    configured = os.environ.get("CUTE_DSL_ARCH")
    if configured is not None:
        assert configured.replace("_", "") == f"sm{arch}a", (
            f"CUTE_DSL_ARCH={configured} does not target this exact arch sm_{arch}a"
        )
    return {
        "gpu": torch.cuda.get_device_name(),
        "capability": capability,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "CUTE_DSL_ARCH": configured,
        "reference_rows": REFERENCE_ROWS,
        "fake_tensor": is_fake_mode(),
    }


def cu_seqlens(lengths, device="cuda"):
    return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device=device)


def make_inputs(case, q_lengths, k_lengths=None, seed=0, distribution="normalized", device="cuda"):
    k_lengths = q_lengths if k_lengths is None else k_lengths
    torch.manual_seed(seed)

    def sample(length, heads):
        tensor = torch.randn(length, heads, case.dim, device=device, dtype=torch.float32)
        if distribution == "normalized":
            tensor = tensor * torch.rsqrt(tensor.square().mean(-1, keepdim=True) + 1e-6)
        elif distribution == "peaky":
            tensor = tensor * 4.0
        elif distribution != "normal":
            raise ValueError(f"Unknown distribution: {distribution}")
        return tensor.to(torch.bfloat16).requires_grad_()

    q = sample(sum(q_lengths), case.heads)
    k = sample(sum(k_lengths), case.kv_heads)
    v = k if case.shared else sample(sum(k_lengths), case.kv_heads)
    dout = torch.randn_like(q)
    return q, k, v, dout


def raw_forward(case, q, k, v, q_lengths, k_lengths):
    """The upstream five-result ABI used by the updated PrimeRL custom op."""
    from flash_attn.cute.interface import _flash_attn_fwd

    out, lse, _, _, _ = _flash_attn_fwd(
        q, k, v, cu_seqlens_q=cu_seqlens(q_lengths), cu_seqlens_k=cu_seqlens(k_lengths),
        max_seqlen_q=max(q_lengths), max_seqlen_k=max(k_lengths), softmax_scale=1.0,
        causal=True, window_size_left=case.window_left, window_size_right=0, return_lse=True,
    )
    return out, lse


def raw_backward(case, q, k, v, out, lse, dout, q_lengths, k_lengths, deterministic=False):
    from flash_attn.cute.interface import _flash_attn_bwd

    dq, dk, dv = _flash_attn_bwd(
        q, k, v, out, dout.contiguous(), lse, softmax_scale=1.0, causal=True,
        window_size_left=case.window_left, window_size_right=0,
        cu_seqlens_q=cu_seqlens(q_lengths), cu_seqlens_k=cu_seqlens(k_lengths),
        max_seqlen_q=max(q_lengths), max_seqlen_k=max(k_lengths), deterministic=deterministic,
    )
    return dq, dk, dv


def kernel_result(case, q, k, v, dout, q_lengths, k_lengths, deterministic=False):
    if case.shared:
        from flash_attn.cute import flash_attn_varlen_func

        out, lse = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_seqlens(q_lengths), cu_seqlens_k=cu_seqlens(k_lengths),
            max_seqlen_q=max(q_lengths), max_seqlen_k=max(k_lengths), softmax_scale=1.0,
            causal=True, window_size=(case.window_left, 0), return_lse=True,
            deterministic=deterministic,
        )
        dq, dkv = torch.autograd.grad(out, (q, k), dout)
        return {"O": out.detach(), "LSE": lse.detach(), "dQ": dq, "dKV": dkv}
    out, lse = raw_forward(case, q, k, v, q_lengths, k_lengths)
    dq, dk, dv = raw_backward(case, q, k, v, out, lse, dout, q_lengths, k_lengths, deterministic)
    return {"O": out, "LSE": lse, "dQ": dq, "dK": dk, "dV": dv}


@contextlib.contextmanager
def fp32_math():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def dense_reference(q, k, v, dout, q_lengths, k_lengths, window_left, *, dtype, rows=REFERENCE_ROWS,
                    causal=True, window_right=0, softmax_scale=1.0, qv=None):
    """Exact dense per-row softmax; all key positions and all heads are evaluated.

    BF16 baseline: PyTorch bf16 GEMMs, bf16 softmax and autograd, with slab/head
    gradient contributions summed in fp32. FP32 reference uses the same graph
    entirely in fp32. No input/output quantization is hidden in the fp32 result.
    The final shared gradient is summed before its bf16 boundary conversion.
    """
    assert rows > 0 and len(q_lengths) == len(k_lengths)
    assert sum(q_lengths) == q.shape[0] and sum(k_lengths) == k.shape[0]
    outputs = torch.zeros((*q.shape[:-1], v.shape[-1]), device=q.device, dtype=torch.float32)
    lse = torch.full((q.shape[1], q.shape[0]), -math.inf, device=q.device, dtype=torch.float32)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    dqv = torch.zeros_like(qv, dtype=torch.float32) if qv is not None else None
    q_start = k_start = 0
    with fp32_math(), torch.enable_grad():
        for q_len, k_len in zip(q_lengths, k_lengths):
            if k_len:
                key_pos = torch.arange(k_len, device=q.device)[None, :]
                for head in range(q.shape[1]):
                    kv_head = head // (q.shape[1] // k.shape[1])
                    kh = k[k_start:k_start + k_len, kv_head].detach().to(dtype).requires_grad_()
                    vh = v[k_start:k_start + k_len, kv_head].detach().to(dtype).requires_grad_()
                    for begin in range(0, q_len, rows):
                        end = min(begin + rows, q_len)
                        qh = q[q_start + begin:q_start + end, head].detach().to(dtype).requires_grad_()
                        query_pos = torch.arange(begin, end, device=q.device)[:, None] + k_len - q_len
                        allowed = torch.ones((end - begin, k_len), device=q.device, dtype=torch.bool)
                        if causal:
                            allowed &= key_pos <= query_pos
                        if window_left is not None:
                            allowed &= key_pos >= query_pos - window_left
                        if window_right is not None:
                            allowed &= key_pos <= query_pos + window_right
                        valid = allowed.any(-1)
                        scores = qh @ kh.T
                        if qv is not None:
                            qvh = qv[q_start + begin:q_start + end, head].detach().to(dtype).requires_grad_()
                            scores = scores + qvh @ vh.T
                        scores = (scores * softmax_scale).masked_fill(~allowed, -math.inf)
                        # Avoid the undefined derivative of softmax([-inf,...,-inf]).
                        safe_scores = torch.where(valid[:, None], scores, torch.zeros_like(scores))
                        probabilities = safe_scores.softmax(-1).masked_fill(~valid[:, None], 0)
                        oh = probabilities @ vh
                        gh = dout[q_start + begin:q_start + end, head].to(dtype)
                        gradients = torch.autograd.grad(oh, (qh, kh, vh, qvh) if qv is not None else (qh, kh, vh), gh)
                        dqh, dkh, dvh = gradients[:3]
                        if qv is not None:
                            dqv[q_start + begin:q_start + end, head] = gradients[3].float()
                        outputs[q_start + begin:q_start + end, head] = oh.detach().float()
                        lse[head, q_start + begin:q_start + end] = torch.where(
                            valid, torch.logsumexp(scores.detach(), -1).float(), -math.inf,
                        )
                        dq[q_start + begin:q_start + end, head] = dqh.float()
                        dk[k_start:k_start + k_len, kv_head] += dkh.float()
                        dv[k_start:k_start + k_len, kv_head] += dvh.float()
            q_start += q_len
            k_start += k_len
    result = {"O": outputs, "LSE": lse, "dQ": dq, "dK": dk, "dV": dv}
    if dqv is not None:
        result["dQv"] = dqv
    return result


def as_shared(result):
    return {"O": result["O"], "LSE": result["LSE"], "dQ": result["dQ"],
            "dKV": result["dK"] + result["dV"]}


def tensor_chunks(*tensors, max_elements=2**20):
    """Bound metric scratch memory without sampling or dropping any elements."""
    assert all(t.shape == tensors[0].shape for t in tensors)
    if tensors[0].ndim == 0:
        yield tensors
        return
    row_elements = math.prod(tensors[0].shape[1:])
    rows = max(1, max_elements // max(1, row_elements))
    for start in range(0, tensors[0].shape[0], rows):
        yield tuple(t[start:start + rows] for t in tensors)


def metrics(actual, expected):
    assert actual.shape == expected.shape
    count = 0
    squared = ref_squared = actual_squared = dot = peak = max_abs = max_rel = 0.0
    for actual_chunk, expected_chunk in tensor_chunks(actual, expected):
        assert torch.equal(torch.isfinite(actual_chunk), torch.isfinite(expected_chunk)), "Finite masks differ"
        assert not torch.isnan(expected_chunk).any() and not torch.isnan(actual_chunk).any()
        assert torch.equal(torch.isneginf(actual_chunk), torch.isneginf(expected_chunk)), "-inf masks differ"
        assert torch.equal(torch.isposinf(actual_chunk), torch.isposinf(expected_chunk)), "+inf masks differ"
        finite = torch.isfinite(expected_chunk)
        a, b = actual_chunk[finite].float(), expected_chunk[finite].float()
        if not a.numel():
            continue
        delta = a - b
        relative = torch.where(b != 0, delta.abs() / b.abs(), torch.where(delta == 0, 0.0, math.inf))
        count += a.numel()
        squared += delta.double().square().sum().item()
        ref_squared += b.double().square().sum().item()
        actual_squared += a.double().square().sum().item()
        dot += (a.double() * b.double()).sum().item()
        peak = max(peak, b.abs().max().item())
        max_abs = max(max_abs, delta.abs().max().item())
        max_rel = max(max_rel, relative.max().item())
    rmse = math.sqrt(squared / count) if count else 0.0
    denominator = math.sqrt(actual_squared * ref_squared)
    return {
        "count": count, "nonfinite": expected.numel() - count,
        "max_abs": max_abs, "max_rel": max_rel,
        "rmse": rmse, "psnr": math.inf if rmse == 0 else 20 * math.log10(peak / rmse) if peak else -math.inf,
        "cosine": dot / denominator if denominator else float(squared == 0),
        "relative_l2": math.sqrt(squared / ref_squared) if ref_squared else (0.0 if squared == 0 else math.inf),
    }


def check_result(actual, reference, baseline, destination):
    """Both FA's 2x-baseline rule and the unchanged SM90 budgets are required."""
    assert actual.keys() == reference.keys() == baseline.keys()
    for name in actual:
        a, ref, pt = actual[name], reference[name], baseline[name]
        measured = metrics(a, ref)
        arithmetic_atol = 0.0
        for (chunk,) in tensor_chunks(ref):
            finite = chunk[torch.isfinite(chunk)]
            if finite.numel():
                arithmetic_atol = max(arithmetic_atol, 2 * (finite + 0.3 - 0.3 - finite).abs().max().item())
        baseline_error = metrics(pt, ref)["max_abs"]
        bound = 2 * baseline_error + arithmetic_atol
        measured.update(baseline_max_abs=baseline_error, arithmetic_atol=arithmetic_atol, max_abs_bound=bound)
        destination[name] = measured
    for name in actual:
        a, ref, measured = actual[name], reference[name], destination[name]
        assert measured["max_abs"] <= measured["max_abs_bound"], f"{name}: {measured}"
        if name in ("O", "LSE"):
            # Preserve SM90's output-boundary rounding for the actual dtype.
            for actual_chunk, ref_chunk in tensor_chunks(a, ref):
                sm90_ref = ref_chunk.to(a.dtype).float() if name == "O" else ref_chunk
                torch.testing.assert_close(actual_chunk.float(), sm90_ref, atol=3e-2, rtol=3e-2)
        else:
            sm90_ref = ref.to(a.dtype).float()
            relative_l2 = metrics(a, sm90_ref)["relative_l2"]
            measured["sm90_relative_l2"] = relative_l2
            assert relative_l2 < 1e-2, f"{name}: SM90 relative L2 {relative_l2} >= 0.01"


def check_reference(device="cpu"):
    """Independent full-matrix autograd checks the slab/head implementation."""
    torch.manual_seed(19)
    for q_lengths, k_lengths in (((7, 3), (4, 5)), ((0, 5, 2), (3, 0, 2))):
        for window in (None, 2):
            q = torch.randn(sum(q_lengths), 4, 8, device=device)
            k = torch.randn(sum(k_lengths), 2, 8, device=device)
            v = torch.randn_like(k)
            dout = torch.randn_like(q)
            chunked = dense_reference(q, k, v, dout, q_lengths, k_lengths, window, dtype=torch.float32, rows=3)
            expected = {"O": torch.zeros_like(q), "LSE": torch.full((4, q.shape[0]), -math.inf, device=device),
                        "dQ": torch.zeros_like(q), "dK": torch.zeros_like(k), "dV": torch.zeros_like(v)}
            qs = ks = 0
            with fp32_math():
                for nq, nk in zip(q_lengths, k_lengths):
                    if nq and nk:
                        qf = q[qs:qs+nq].clone().requires_grad_()
                        kf = k[ks:ks+nk].clone().requires_grad_()
                        vf = v[ks:ks+nk].clone().requires_grad_()
                        score = torch.einsum("ihd,jhd->hij", qf, kf.repeat_interleave(2, 1))
                        mask = torch.tensor([[j <= i + nk - nq and (window is None or j >= i + nk - nq - window)
                                              for j in range(nk)] for i in range(nq)], device=device)
                        score = score.masked_fill(~mask, -math.inf)
                        valid = mask.any(-1)
                        probability = torch.where(valid[None, :, None], score, 0.0).softmax(-1)
                        probability = probability.masked_fill(~valid[None, :, None], 0.0)
                        out = torch.einsum("hij,jhd->ihd", probability, vf.repeat_interleave(2, 1))
                        grads = torch.autograd.grad(out, (qf, kf, vf), dout[qs:qs+nq])
                        expected["O"][qs:qs+nq] = out.detach()
                        expected["LSE"][:, qs:qs+nq] = torch.logsumexp(score.detach(), -1)
                        expected["dQ"][qs:qs+nq], expected["dK"][ks:ks+nk], expected["dV"][ks:ks+nk] = grads
                    qs += nq
                    ks += nk
            for name, tensor in expected.items():
                torch.testing.assert_close(chunked[name], tensor, atol=2e-5, rtol=2e-5)
    return {"device": device, "packs": 2, "windows": 2, "chunk_rows": 3}


def run_case(case, q_lengths, k_lengths=None, *, seed=0, distribution="normalized", deterministic=False, name=None,
             return_artifacts=False):
    k_lengths = q_lengths if k_lengths is None else k_lengths
    rungs = Rungs(name or f"{case.name}-{max(q_lengths)}-s{seed}")
    try:
        with rungs.rung("environment") as result:
            result.update(environment(), case=asdict(case), q_lengths=q_lengths, k_lengths=k_lengths,
                          seed=seed, distribution=distribution, deterministic=deterministic)
        with rungs.rung("compile-and-run") as result:
            q, k, v, dout = make_inputs(case, q_lengths, k_lengths, seed, distribution)
            actual = kernel_result(case, q, k, v, dout, q_lengths, k_lengths, deterministic)
            if not is_fake_mode():
                torch.cuda.synchronize()
            result.update(shapes={name: tuple(t.shape) for name, t in actual.items()})
            result["dtypes"] = {name: str(t.dtype) for name, t in actual.items()}
            for tensor_name, tensor in actual.items():
                expected_dtype = torch.float32 if tensor_name == "LSE" else torch.bfloat16
                assert tensor.dtype == expected_dtype, f"{tensor_name}: expected {expected_dtype}, got {tensor.dtype}"
        if is_fake_mode():
            return rungs.results
        with rungs.rung("reference-self-check") as result:
            result.update(check_reference("cuda"))
        with rungs.rung("fp32-reference") as result:
            reference = dense_reference(q, k, v, dout, q_lengths, k_lengths, case.window_left, dtype=torch.float32)
            if case.shared:
                reference = as_shared(reference)
            torch.cuda.synchronize()
            result["all_elements"] = True
        with rungs.rung("bf16-pytorch-baseline") as result:
            baseline = dense_reference(q, k, v, dout, q_lengths, k_lengths, case.window_left, dtype=torch.bfloat16)
            if case.shared:
                baseline = as_shared(baseline)
            torch.cuda.synchronize()
            result["rows"] = REFERENCE_ROWS
        with rungs.rung("forward-correctness") as result:
            check_result({n: actual[n] for n in ("O", "LSE")}, {n: reference[n] for n in ("O", "LSE")},
                         {n: baseline[n] for n in ("O", "LSE")}, result)
        with rungs.rung("backward-correctness") as result:
            names = tuple(n for n in actual if n.startswith("d"))
            check_result({n: actual[n] for n in names}, {n: reference[n] for n in names},
                         {n: baseline[n] for n in names}, result)
        if return_artifacts:
            return rungs.results, (q, k, v, dout), reference, baseline
        return rungs.results
    finally:
        rungs.finish()


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("pack", PACKS)
@compilation_pass
def test_gemma4_sm100_varlen(case_name, pack):
    run_case(CASES[case_name], PACKS[pack], name=f"{case_name}-{pack}")


@pytest.mark.parametrize("case_name", ("global12", "global26", "global31", "sliding12_26", "sliding31"))
@pytest.mark.parametrize("seed,distribution", ((17, "normal"), (29, "peaky")))
@compilation_pass
def test_gemma4_sm100_distributions(case_name, seed, distribution):
    run_case(CASES[case_name], (129, 257), seed=seed, distribution=distribution,
             name=f"{case_name}-{distribution}-s{seed}")


@pytest.mark.parametrize("case_name", CASES)
@compilation_pass
def test_gemma4_sm100_fully_masked_and_empty_segments(case_name):
    run_case(CASES[case_name], (257, 0, 129, 65), (129, 64, 257, 0),
             name=f"{case_name}-masked-empty")


@pytest.mark.parametrize("case_name", ("global12", "global26", "global31", "sliding12_26", "sliding31"))
@compilation_pass
def test_gemma4_sm100_deterministic(case_name):
    case = CASES[case_name]
    lengths = (129, 257)
    rungs = Rungs(f"{case_name}-deterministic")
    try:
        with rungs.rung("repeat") as result:
            result.update(environment())
            q, k, v, dout = make_inputs(case, lengths)
            outputs = [kernel_result(case, q, k, v, dout, lengths, lengths, True) for _ in range(3)]
            if is_fake_mode():
                return
            torch.cuda.synchronize()
            for candidate in outputs[1:]:
                for key in outputs[0]:
                    assert torch.equal(outputs[0][key], candidate[key]), f"{key} is not deterministic"
            result["repeats"] = 3
        run_case(case, lengths, deterministic=True, name=f"{case_name}-deterministic-numerics")
    finally:
        rungs.finish()


@pytest.mark.parametrize("case_name", ("global12", "global26", "global31"))
@compilation_pass
def test_gemma4_sm100_alias_and_equal_distinct_storage(case_name):
    case = CASES[case_name]
    shared_case = Case(case.name + "-alias", case.heads, case.kv_heads, case.dim, None, True)
    lengths = (129, 257)
    rungs = Rungs(f"{case_name}-alias-equal-distinct")
    try:
        with rungs.rung("compile-and-run"):
            environment()
            q, k, _, dout = make_inputs(case, lengths)
            distinct = k.detach().clone().requires_grad_()
            ordinary = kernel_result(case, q, k, distinct, dout, lengths, lengths)
            shared = kernel_result(shared_case, q, k, k, dout, lengths, lengths)
        if is_fake_mode():
            return
        with rungs.rung("shared-gradient") as result:
            assert k.data_ptr() != distinct.data_ptr()
            reference = dense_reference(q, k, k, dout, lengths, lengths, None, dtype=torch.float32)
            baseline = dense_reference(q, k, k, dout, lengths, lengths, None, dtype=torch.bfloat16)
            check_result(ordinary, reference, baseline, result.setdefault("distinct", {}))
            check_result(shared, as_shared(reference), as_shared(baseline), result.setdefault("shared", {}))
            result["shared_vs_separate_sum"] = metrics(shared["dKV"], ordinary["dK"].float() + ordinary["dV"].float())
    finally:
        rungs.finish()


@pytest.mark.parametrize("case_name", ("global12", "sliding12_26"))
@compilation_pass
def test_gemma4_sm100_packed_sequence_isolation(case_name):
    case = CASES[case_name]
    lengths = (129, 257, 511)
    rungs = Rungs(f"{case_name}-sequence-isolation")
    try:
        with rungs.rung("compile-and-run"):
            environment()
            q, k, v, dout = make_inputs(case, lengths)
            ordinary = kernel_result(case, q, k, v, dout, lengths, lengths, True)
            changed = [x.detach().clone().requires_grad_() for x in (q, k, v)]
            with torch.no_grad():
                for x in changed:
                    x[:lengths[0]].mul_(3)
            perturbed = kernel_result(case, *changed, dout, lengths, lengths, True)
            # The final tile of a packed sequence can over-read the FOLLOWING
            # sequence. NaN/Inf detects contamination even when P is exactly 0.
            poisoned_results = []
            for poison in (math.nan, math.inf):
                poisoned = [x.detach().clone().requires_grad_() for x in (q, k, v)]
                with torch.no_grad():
                    for x in poisoned:
                        x[sum(lengths[:-1]):].fill_(poison)
                poisoned_results.append(kernel_result(case, *poisoned, dout, lengths, lengths, True))
        if is_fake_mode():
            return
        with rungs.rung("isolation") as result:
            for key in ordinary:
                a = ordinary[key][:, lengths[0]:] if key == "LSE" else ordinary[key][lengths[0]:]
                b = perturbed[key][:, lengths[0]:] if key == "LSE" else perturbed[key][lengths[0]:]
                assert torch.equal(a, b), f"{key} leaked between packed sequences"
                earlier = ordinary[key][:, :sum(lengths[:-1])] if key == "LSE" else ordinary[key][:sum(lengths[:-1])]
                for poisoned in poisoned_results:
                    candidate = poisoned[key][:, :sum(lengths[:-1])] if key == "LSE" else poisoned[key][:sum(lengths[:-1])]
                    assert torch.equal(earlier, candidate), f"{key} read NaN/Inf from the following packed sequence"
            result.update(unchanged_sequences=2, following_sequence_poison=["NaN", "+Inf"])
    finally:
        rungs.finish()
