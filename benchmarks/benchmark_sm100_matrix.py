"""Source-stamped, correctness-gated FA4 benchmark matrix.

The parent uses only the standard library. Normal-grid workers persist across
rows at fixed import roots, releasing backend graphs between calls and references
between groups. The overhead path uses fresh backend workers per row, retained
across its alternating timing rounds.
Example (run from the verifier's environment):
  uv run python benchmarks/benchmark_sm100_matrix.py --manifest MANIFEST \
    --candidate-root ROOT --pin-root PIN --upstream-root UPSTREAM --arch 103 \
    --output tmp/bench-run --overhead
Use --prepare-only to write the complete pending ledger without CUDA or imports.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROTOCOL = "FA4_BENCH_RESPONSE "
BACKENDS = (
    "candidate",
    "pin",
    "upstream",
    "sdpa-cudnn",
    "sdpa-flash",
    "sdpa-efficient",
    "sdpa-math",
    "anchor",
)
# NVIDIA's HGX table specifies eight SXM GPUs and sparse FP16/BF16 throughput.
# Footnote 2 states that dense throughput is half the listed value.
PEAK_SOURCE = {
    "url": "https://www.nvidia.com/en-au/data-center/hgx/",
    "title": "NVIDIA HGX: B200 and B300 specifications, footnote 2",
    "verified_utc_date": "2026-10-29",
    "system_sparse_fp16_bf16_pflops": 36.0,
    "gpus_per_system": 8,
    "sparse_to_dense_divisor": 2,
}
DEVICE_PEAK_BINDINGS = {
    "NVIDIA B200": ("HGX B200", (10, 0)),
    "NVIDIA B300 SXM6 PC": ("HGX B300", (10, 3)),
}


def nominal_dense_peak(device_name, capability, dtype):
    """Bind the published full-device nominal peak; never infer it from SM alone."""
    binding = DEVICE_PEAK_BINDINGS.get(device_name)
    if binding is None or tuple(capability) != binding[1]:
        raise ValueError(f"Unknown dense-peak binding: {device_name!r}, SM{capability}")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError(f"No FP16/BF16 dense-peak binding for dtype {dtype!r}")
    return {
        "device_name": device_name,
        "capability": list(capability),
        "product": binding[0],
        "dtype": dtype,
        "nominal_dense_tflops": (
            PEAK_SOURCE["system_sparse_fp16_bf16_pflops"]
            * 1000
            / PEAK_SOURCE["gpus_per_system"]
            / PEAK_SOURCE["sparse_to_dense_divisor"]
        ),
        "basis": "Published nominal full-GPU dense peak; not adjusted for measured clocks",
        "source": dict(PEAK_SOURCE),
    }


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def useful_pairs(sample):
    """Lower-right attention alignment, including empty and rectangular rows."""
    pairs = 0
    left, right = sample["window"]
    for nq, nk in zip(
        sample.get("seqused_q", sample["q_lengths"]),
        sample.get("seqused_k", sample["k_lengths"]),
    ):
        for i in range(nq):
            center = i + nk - nq
            low = max(0, center - left) if left is not None else 0
            high = nk - 1
            if right is not None:
                high = min(high, center + right)
            if sample["causal"]:
                high = min(high, center)
            pairs += max(0, high - low + 1)
    return pairs


def useful_flops(sample):
    d, dv = sample["dimension"], sample["dimension_v"]
    score_d = d + (dv if sample["api"] == "mla" else 0)
    pairs_heads = useful_pairs(sample) * sample["query_heads"]
    forward = pairs_heads * (2 * score_d + 2 * dv)
    backward = pairs_heads * (6 * score_d + 4 * dv)
    return {"fwd": forward, "bwd": backward, "total": forward + backward}


def distribution(values):
    return (
        {
            "count": len(values),
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }
        if values
        else {"count": 0, "min": None, "median": None, "max": None}
    )


def paired_interval(ratios):
    """Descriptive bootstrap over paired rounds, not independent inner samples."""
    if not ratios:
        return None
    rng = random.Random(20261029)
    draws = sorted(
        statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000)
    )
    return {
        "median": statistics.median(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "bootstrap_median_95_percentile": [draws[249], draws[9749]],
        "round_count": len(ratios),
        "all_round_ratios": ratios,
        "method": "10000 paired-round bootstrap resamples; descriptive interval",
    }


def overhead_assessment(candidate_rounds, baseline_rounds, expected_rounds=15):
    candidate = {r["round"]: r for r in candidate_rounds if r["status"] == "pass"}
    baseline = {r["round"]: r for r in baseline_rounds if r["status"] == "pass"}
    matched = sorted(candidate.keys() & baseline.keys())
    ratios = [
        candidate[i]["timings"]["bwd"]["device_ms"]["median"]
        / baseline[i]["timings"]["bwd"]["device_ms"]["median"]
        for i in matched
        if baseline[i]["timings"]["bwd"]["device_ms"]["median"] > 0
    ]
    interval = paired_interval(ratios)
    complete = len(ratios) == expected_rounds
    return {
        "status": "incomplete"
        if not complete
        else "within_2_percent"
        if 0.98 <= interval["median"] <= 1.02
        else "outside_2_percent",
        "expected_rounds": expected_rounds,
        "matched_round_ids": matched,
        "candidate_over_baseline_latency": interval,
        "overhead_percent": 100 * (interval["median"] - 1) if interval else None,
        "regression_over_2_percent": interval["median"] > 1.02 if complete else None,
    }


def explicit_unsupported(exc):
    # No OOM, illegal access, unknown assertion or internal error is a support rejection.
    text = str(exc)
    return isinstance(exc, (RuntimeError, NotImplementedError, AssertionError)) and any(
        marker in text
        for marker in (
            "No available kernel",
            "No viable backend",
            "No suitable kernel",
            "does not support",
            "not supported",
            "not implemented for",
            "Unsupported head",
            "Only supports",
            "only supports",
        )
    )


def tensor_hash(tensor):
    import torch

    cpu = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def tensor_hashes(tensors):
    return [
        {
            "shape": list(t.shape),
            "stride": list(t.stride()),
            "dtype": str(t.dtype),
            "sha256": tensor_hash(t),
        }
        if t is not None
        else None
        for t in tensors
    ]


def clocks():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def verify_baseline(root, revision, repository, frozen=None):
    """Verify archived baseline contents against immutable Git objects, not a label."""
    if frozen is None:
        full = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", revision], text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "-C", str(repository), "ls-tree", "-r", full, "flash_attn/cute"],
            text=True,
        )
        expected = {}
        for line in tree.splitlines():
            metadata, name = line.split("\t", 1)
            if name.endswith(".py"):
                expected[name] = metadata.split()[2]
    else:
        full, expected = frozen["revision"], frozen["git_blob_ids"]
        assert full.startswith(revision), (
            "Frozen baseline provenance names another revision"
        )
    actual = {
        str(path.relative_to(root)) for path in (root / "flash_attn/cute").rglob("*.py")
    }
    assert actual == set(expected), "Baseline package file set differs from revision"
    sources = {}
    for name, oid in expected.items():
        content = (root / name).read_bytes()
        actual_oid = hashlib.sha1(
            f"blob {len(content)}\0".encode() + content
        ).hexdigest()
        assert oid == actual_oid, f"Baseline differs from {full}: {name}"
        sources[str(Path(name).relative_to("flash_attn"))] = hashlib.sha256(
            content
        ).hexdigest()
    return {
        "revision": full,
        "git_blob_ids": expected,
        "package_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()
        ).hexdigest(),
    }


def native_calls(sample, tensors, kwargs, revision, implicit_shared_mla=False):
    import torch
    from gemma4_generalization import packed

    from flash_attn.cute.interface import (
        _flash_attn_bwd,
        _flash_attn_fwd,
        flash_attn_varlen_func,
    )

    q, k, v, qv, dout = tensors
    public = sample["api"] == "mla" or v is k
    inputs = (q, k) if v is k else (q, k, v)
    names = ["dQ", "dKV"] if v is k else ["dQ", "dK", "dV"]
    if qv is not None:
        inputs += (qv,)
        names.append("dQv")
    raw_kwargs = {
        **kwargs,
        "window_size_left": sample["window"][0],
        "window_size_right": sample["window"][1],
    }

    def forward():
        if public:
            return flash_attn_varlen_func(
                q,
                k,
                v,
                qv=qv,
                **kwargs,
                window_size=tuple(sample["window"]),
                deterministic=sample["deterministic"],
                return_lse=True,
            )
        returns = _flash_attn_fwd(q, k, v, **raw_kwargs, return_lse=True)
        # Only the isolated pinned baseline has the old raw ABI; candidate is strict.
        assert len(returns) == (4 if revision == "pin" else 5), (
            "Unexpected raw forward ABI"
        )
        return returns[0], returns[1]

    held, lse = forward()

    def gradients(out, stats, retain_graph=False):
        if public:
            return torch.autograd.grad(out, inputs, dout, retain_graph=retain_graph)
        return _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            stats,
            **raw_kwargs,
            deterministic=sample["deterministic"],
        )

    def backward():
        return gradients(held, lse, retain_graph=True)

    def total():
        out, stats = forward()
        return gradients(out, stats)

    mla_lse = qv is not None or (implicit_shared_mla and v is k and v.shape[-1] == 512)
    canonical_lse = lse.transpose(-1, -2) if mla_lse else lse
    actual = {
        "O": packed(held),
        "LSE": canonical_lse.permute(1, 0, 2).flatten(1)
        if canonical_lse.ndim == 3
        else canonical_lse,
    }
    calls = {"fwd": forward}
    if sample["direction"] != "forward":
        actual.update({name: packed(grad) for name, grad in zip(names, backward())})
        calls.update(bwd=backward, total=total)
    return calls, actual


def sdpa_calls(sample, tensors, logical, backend):
    import torch
    import torch.nn.functional as F
    from gemma4_generalization import scatter_effective
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if sample["api"] == "mla":
        raise NotImplementedError("SDPA does not support the qv MLA API")
    q, k, v, dout, _ = logical
    if tensors[1] is tensors[2]:
        v = k
    inputs = (q, k) if v is k else (q, k, v)
    names = ["dQ", "dKV"] if v is k else ["dQ", "dK", "dV"]
    qlens, klens = sample["q_lengths"], sample["k_lengths"]
    qused, kused = sample.get("seqused_q", qlens), sample.get("seqused_k", klens)
    entries, qi, ki = [], 0, 0
    left, right = sample["window"]
    for nq, nk, cq, ck in zip(qused, kused, qlens, klens):
        causal = sample["causal"] and nq == nk and left is None and right is None
        mask = None
        if (sample["causal"] and not causal) or left is not None or right is not None:
            row = torch.arange(nq, device=q.device)[:, None] + nk - nq
            col = torch.arange(nk, device=q.device)[None, :]
            mask = torch.ones((nq, nk), dtype=torch.bool, device=q.device)
            if sample["causal"]:
                mask &= col <= row
            if left is not None:
                mask &= col >= row - left
            if right is not None:
                mask &= col <= row + right
        entries.append((qi, ki, nq, nk, mask, causal))
        qi, ki = qi + cq, ki + ck
    selected = {
        "sdpa-cudnn": SDPBackend.CUDNN_ATTENTION,
        "sdpa-flash": SDPBackend.FLASH_ATTENTION,
        "sdpa-efficient": SDPBackend.EFFICIENT_ATTENTION,
        "sdpa-math": SDPBackend.MATH,
    }[backend]

    def forward():
        parts = []
        with sdpa_kernel(selected):
            for qi, ki, nq, nk, mask, causal in entries:
                qs = q[qi : qi + nq].transpose(0, 1).unsqueeze(0)
                ks = k[ki : ki + nk].transpose(0, 1).unsqueeze(0)
                vs = v[ki : ki + nk].transpose(0, 1).unsqueeze(0)
                gqa = sample["query_heads"] != sample["kv_heads"]
                if backend == "sdpa-efficient" and gqa:
                    # Efficient SDPA has no native GQA; include expansion in timing,
                    # as in the exercised D512 baseline. Autograd sums its gradients.
                    ratio = sample["query_heads"] // sample["kv_heads"]
                    ks = ks.repeat_interleave(ratio, dim=1)
                    vs = vs.repeat_interleave(ratio, dim=1)
                parts.append(
                    F.scaled_dot_product_attention(
                        qs,
                        ks,
                        vs,
                        attn_mask=mask,
                        is_causal=causal,
                        dropout_p=0.0,
                        scale=sample["softmax_scale"],
                        enable_gqa=gqa and backend != "sdpa-efficient",
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
        return scatter_effective(torch.cat(parts), qlens, qused)

    held = forward()

    def backward():
        return torch.autograd.grad(held, inputs, dout, retain_graph=True)

    def total():
        return torch.autograd.grad(forward(), inputs, dout)

    actual, calls = {"O": held}, {"fwd": forward}
    if sample["direction"] != "forward":
        actual.update(dict(zip(names, backward())))
        calls.update(bwd=backward, total=total)
    return calls, actual


def measure(fn, warmup, repeats):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    samples, walls = [], []
    for start, end in events:
        wall = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        walls.append((time.perf_counter() - wall) * 1000)
        samples.append(start.elapsed_time(end))
    return {
        "device_ms": distribution(samples),
        "wall_ms": distribution(walls),
        "all_device_ms": samples,
        "all_wall_ms": walls,
    }


def respond(value):
    print(PROTOCOL + json.dumps(value, sort_keys=True), flush=True)


def worker(args):
    # Set before importing either torch or FA4. Harness import is shared; package import is isolated.
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "tests" / "cute"))
    import torch
    from gemma4_generalization import (
        check_general_reference,
        check_reference,
        check_result,
        environment,
        make_workload,
        reference,
        source_provenance,
        used_results,
    )

    from flash_attn.cute import interface

    assert (
        Path(interface.__file__).resolve() == root / "flash_attn/cute/interface.py"
    ), interface.__file__
    requested = json.loads(Path(args.sample).read_text())
    sample = dict(requested)
    if args.backend == "anchor":
        sample.update(dimension=128, dimension_v=128)
    inherited_unsupported = []
    implicit_mla = (
        args.implicit_shared_mla
        and sample["tensor_layout"] == "alias"
        and sample["dimension_v"] == 512
    )
    if (sample["api"] == "mla" or implicit_mla) and sample["direction"] != "forward":
        inherited_unsupported = ["bwd", "total"]
        sample["direction"] = "forward"
    stage = "environment"
    gate = {
        "backend": args.backend,
        "requested_sample": requested,
        "sample": sample,
        "shape_input_sha256": digest(sample),
        "revision": args.revision,
        "inherited_unsupported_directions": inherited_unsupported,
        "inherited_unsupported_reason": "Dense MLA backward requires gather_kv_indices"
        if inherited_unsupported
        else None,
    }
    gate["harness_sha256"] = {
        name: hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "tests/cute" / name).read_bytes()
        ).hexdigest()
        for name in ("gemma4_generalization.py", "test_gemma4_sm100.py")
    }
    try:
        gate.update(
            environment=environment(),
            provenance=source_provenance(),
            clocks_before=clocks(),
        )
        gate["dense_peak"] = nominal_dense_peak(
            gate["environment"]["gpu"],
            gate["environment"]["capability"],
            sample["dtype"],
        )
        check_reference("cuda")
        check_general_reference("cuda")
        stage = "inputs"
        tensors, logical, kwargs = make_workload(sample)
        gate["input_hashes"] = tensor_hashes(tensors)
        stage = "reference"
        ref = reference(sample, logical, torch.float32)
        baseline = reference(sample, logical, getattr(torch, sample["dtype"]))
        stage = "compile-and-run"
        if args.backend.startswith("sdpa-"):
            calls, actual = sdpa_calls(sample, tensors, logical, args.backend)
        else:
            calls, actual = native_calls(
                sample, tensors, kwargs, args.backend, args.implicit_shared_mla
            )
        torch.cuda.synchronize()
        stage = "correctness"
        gate["correctness"] = {}
        check_result(
            used_results(sample, actual),
            used_results(sample, {k: ref[k] for k in actual}),
            used_results(sample, {k: baseline[k] for k in actual}),
            gate["correctness"],
        )
        if not args.backend.startswith("sdpa-"):
            replay_calls, again = native_calls(
                sample, tensors, kwargs, args.backend, args.implicit_shared_mla
            )
            gate["replay_correctness"] = {}
            check_result(
                used_results(sample, again),
                used_results(sample, ref),
                used_results(sample, baseline),
                gate["replay_correctness"],
            )
            gate["replay_output_hashes"] = {
                name: tensor_hash(tensor)
                for name, tensor in used_results(sample, again).items()
            }
            gate["within_build_bitexact"] = {
                name: torch.equal(tensor, used_results(sample, again)[name])
                for name, tensor in used_results(sample, actual).items()
            }
            if sample["deterministic"]:
                assert all(gate["within_build_bitexact"].values()), (
                    "Deterministic replay differs"
                )
            del replay_calls, again
        gate["output_hashes"] = {
            name: tensor_hash(tensor)
            for name, tensor in used_results(sample, actual).items()
        }
        if args.save_outputs:
            torch.save(
                {
                    name: tensor.detach().cpu()
                    for name, tensor in used_results(sample, actual).items()
                },
                args.save_outputs,
            )
        assert source_provenance() == gate["provenance"], (
            "Sources changed during correctness"
        )
        assert tensor_hashes(tensors) == gate["input_hashes"], (
            "Inputs changed during correctness"
        )
        gate["status"] = "pass"
        gate["directions"] = list(calls)
        del ref, baseline, actual
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - persist every failure; never mark it successful
        traceback.print_exc()
        support = (
            args.backend != "candidate"
            and explicit_unsupported(exc)
            and stage == "compile-and-run"
        )
        gate.update(
            status="baseline_unsupported"
            if support
            else "numerical_fail"
            if stage == "correctness"
            else "compile_error"
            if type(exc).__module__.startswith("cutlass")
            else "runtime_error",
            stage=stage,
            error=f"{type(exc).__name__}: {exc}",
        )
        respond(gate)
        return
    respond(gate)
    for line in sys.stdin:
        command = json.loads(line)
        if command["command"] == "close":
            return
        result = {"round": command["round"], "status": "pass", "timings": {}}
        try:
            assert source_provenance() == gate["provenance"], (
                "Source changed after correctness gate"
            )
            assert tensor_hashes(tensors) == gate["input_hashes"], (
                "Input changed after correctness gate"
            )
            result["clocks_before"] = clocks()
            for direction in command["directions"]:
                if direction not in calls:
                    raise NotImplementedError(f"Direction not supported: {direction}")
                result["timings"][direction] = measure(
                    calls[direction], command["warmup"], command["repeats"]
                )
            result["clocks_after"] = clocks()
            assert source_provenance() == gate["provenance"], (
                "Source changed during timing"
            )
            assert tensor_hashes(tensors) == gate["input_hashes"], (
                "Input changed during timing"
            )
        except Exception as exc:  # noqa: BLE001 - return failure to the full coverage ledger
            traceback.print_exc()
            result.update(status="runtime_error", error=f"{type(exc).__name__}: {exc}")
        respond(result)


class Worker:
    def __init__(self, command, root, log, env):
        self.log = open(log, "w")  # noqa: SIM115 - lifetime matches persistent worker
        self.process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.gate = self.receive()

    def receive(self):
        for line in self.process.stdout:
            self.log.write(line)
            self.log.flush()
            if line.startswith(PROTOCOL):
                return json.loads(line[len(PROTOCOL) :])
        return {
            "status": "runtime_error",
            "error": f"Worker exited {self.process.wait()} without response",
        }

    def send(self, command):
        try:
            self.process.stdin.write(json.dumps(command) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            return {
                "status": "runtime_error",
                "error": f"Worker exited {self.process.wait()} before command",
            }
        return self.receive()

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"close"}\n')
                self.process.stdin.flush()
            except BrokenPipeError:
                pass  # Worker already returned its failure and exited; still reap below.
        self.process.wait()
        self.log.close()


# Normal-grid workers never import another package root during their lifetime.
GRID_ORDER = (
    "candidate",
    "sdpa-cudnn",
    "sdpa-flash",
    "sdpa-efficient",
    "sdpa-math",
    "pin",
    "anchor",
)


class Lifecycle:
    allowed = {
        "EMPTY": {"PREPARING"},
        "PREPARING": {"GATED", "RELEASING"},
        "GATED": {"TIMING", "RELEASING"},
        "TIMING": {"RELEASING"},
        "RELEASING": {"EMPTY"},
    }

    def __init__(self, root, generation):
        self.root, self.generation, self.state = root, generation, "EMPTY"

    def assert_identity(self, root, generation):
        assert (root, generation) == (self.root, self.generation), (
            "Worker identity changed"
        )

    def transition(self, state):
        assert state in self.allowed[self.state], (
            f"Invalid lifecycle {self.state}->{state}"
        )
        self.state = state

    def begin(self, row, backend, shape, source):
        self.transition("PREPARING")
        self.row, self.backend, self.shape, self.source = row, backend, shape, source
        self.inputs = None

    def bind(self, shape, source, inputs):
        assert self.shape == shape and self.source == source, (
            "Sample/source binding changed"
        )
        assert self.inputs is None or self.inputs == inputs, "Input binding changed"
        self.inputs = inputs


def normal_support(exc, backend, stage):
    message = str(exc).lower()
    forbidden = (
        "out of memory",
        "illegal",
        "device-side",
        "cuda error",
        "assert triggered",
    )
    return (
        backend != "candidate"
        and stage == "compile-and-run"
        and not any(token in message for token in forbidden)
        and explicit_unsupported(exc)
    )


def file_binding(root, harness):
    sources = {
        str(p.relative_to(root / "flash_attn")): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sorted((root / "flash_attn/cute").rglob("*.py"))
    }
    return {
        "package_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "harness_sha256": {
            n: hashlib.sha256((harness / n).read_bytes()).hexdigest()
            for n in ("gemma4_generalization.py", "test_gemma4_sm100.py")
        },
    }


def effective_sample(requested, backend, implicit_shared_mla):
    sample = dict(requested)
    if backend == "anchor":
        sample.update(dimension=128, dimension_v=128)
    inherited = []
    implicit = (
        implicit_shared_mla
        and sample["tensor_layout"] == "alias"
        and sample["dimension_v"] == 512
    )
    if (sample["api"] == "mla" or implicit) and sample["direction"] != "forward":
        sample["direction"] = "forward"
        inherited = ["bwd", "total"]
    return sample, inherited


def assert_reference_identity(cache, group, key):
    if cache:
        assert cache["group"] == group, "Reference group was not released"
        assert cache["key"] == key, "Reference identity mismatch"


def persistent_worker(args):
    import gc

    root, harness = (
        Path(args.root).resolve(),
        Path(args.harness_root).resolve() / "tests/cute",
    )
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(harness))
    import torch
    import gemma4_generalization as g
    from flash_attn.cute import interface

    assert Path(interface.__file__).resolve() == root / "flash_attn/cute/interface.py"
    assert Path(g.__file__).resolve() == harness / "gemma4_generalization.py"
    life = Lifecycle(str(root), args.generation)
    binding = file_binding(root, harness)
    env = g.environment()

    def math_flags():
        return {
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }

    flags = math_flags()
    self_checks = {
        "status": "pass",
        "reference": g.check_reference("cuda"),
        "general": g.check_general_reference("cuda"),
        "binding": binding,
    }
    cache = {}

    def clean():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "free": free,
            "total": total,
        }

    baseline_memory = clean()

    def verify_binding(command):
        life.assert_identity(command["root"], command["generation"])
        assert command["binding"] == binding == file_binding(root, harness), (
            "Source/harness/runner changed"
        )
        assert g.environment() == env and math_flags() == flags, (
            "Toolchain/device/math flags changed"
        )

    def inputs_binding(tensors, logical, kwargs):
        def value(v):
            if isinstance(v, torch.Tensor):
                return tensor_hashes([v])[0]
            return v

        # Identity relationships and storage relationships are separate from hashes.
        objects = list(tensors) + list(logical)
        alias = [[a is b for b in objects] for a in objects]
        storage = [
            [
                a is not None
                and b is not None
                and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
                for b in objects
            ]
            for a in objects
        ]
        return {
            "physical": tensor_hashes(tensors),
            "logical": tensor_hashes(logical),
            "metadata": {k: value(v) for k, v in sorted(kwargs.items())},
            "object_alias": alias,
            "storage_alias": storage,
        }

    def storage_bytes(values):
        seen = {}
        for mapping in values:
            for tensor in mapping.values():
                storage = tensor.untyped_storage()
                seen[(tensor.device, storage.data_ptr())] = storage.nbytes()
        return sum(seen.values())

    def execute(command):
        backend, requested = command["backend"], command["sample"]
        implicit = command["implicit_shared_mla"]
        sample, inherited = effective_sample(requested, backend, implicit)
        stage = "environment"
        gate = {
            "backend": backend,
            "requested_sample": requested,
            "sample": sample,
            "shape_input_sha256": digest(sample),
            "revision": command["revision"],
            "inherited_unsupported_directions": inherited,
            "inherited_unsupported_reason": "Dense MLA backward requires gather_kv_indices"
            if inherited
            else None,
            "harness_sha256": binding["harness_sha256"],
            "environment": env,
            "provenance": g.source_provenance(),
            "reference_self_checks": self_checks,
            "clocks_before": clocks(),
            "math_flags": flags,
        }
        result = {"round": 0, "status": "pass", "timings": {}}
        try:
            verify_binding(command)
            assert command["effective_sample_sha256"] == digest(sample), (
                "Effective sample changed"
            )
            life.begin(requested["id"], backend, digest(sample), digest(binding))
            gate["dense_peak"] = nominal_dense_peak(
                env["gpu"], env["capability"], sample["dtype"]
            )
            torch.cuda.reset_peak_memory_stats()
            stage = "inputs"
            tensors, logical, kwargs = g.make_workload(sample)
            identity = inputs_binding(tensors, logical, kwargs)
            gate["input_hashes"] = identity["physical"]
            gate["complete_input_binding"] = identity
            life.bind(digest(sample), digest(binding), digest(identity))
            reference_key = digest(
                {
                    "sample": sample,
                    "inputs": identity,
                    "binding": binding,
                    "environment": env,
                    "math_flags": flags,
                }
            )
            gate["reference_key"] = reference_key
            assert_reference_identity(cache, command["group"], reference_key)
            gate["reference_retained_before_submission"] = {
                "storage_bytes": cache.get("storage_bytes", 0),
                "allocated_bytes": cache.get("allocated", 0),
            }
            stage = "compile-and-run"
            if backend.startswith("sdpa-"):
                calls, actual = sdpa_calls(sample, tensors, logical, backend)
            else:
                calls, actual = native_calls(sample, tensors, kwargs, backend, implicit)
            torch.cuda.synchronize()
            stage = "reference"
            reused = bool(cache)
            if not cache:
                before = torch.cuda.memory_allocated()
                ref = {
                    k: v.detach()
                    for k, v in g.reference(sample, logical, torch.float32).items()
                }
                baseline = {
                    k: v.detach()
                    for k, v in g.reference(
                        sample, logical, getattr(torch, sample["dtype"])
                    ).items()
                }
                torch.cuda.synchronize()
                cache.update(
                    group=command["group"],
                    key=reference_key,
                    ref=ref,
                    baseline=baseline,
                    allocated=torch.cuda.memory_allocated() - before,
                    storage_bytes=storage_bytes((ref, baseline)),
                )
            ref, baseline = cache["ref"], cache["baseline"]
            gate["reference_reused"] = reused
            gate["reference_storage_bytes"] = cache["storage_bytes"]
            gate["reference_allocated_bytes"] = cache["allocated"]
            stage = "correctness"
            gate["correctness"] = {}
            g.check_result(
                g.used_results(sample, actual),
                g.used_results(sample, {k: ref[k] for k in actual}),
                g.used_results(sample, {k: baseline[k] for k in actual}),
                gate["correctness"],
            )
            if not backend.startswith("sdpa-"):
                replay_calls, again = native_calls(
                    sample, tensors, kwargs, backend, implicit
                )
                gate["replay_correctness"] = {}
                g.check_result(
                    g.used_results(sample, again),
                    g.used_results(sample, ref),
                    g.used_results(sample, baseline),
                    gate["replay_correctness"],
                )
                gate["replay_output_hashes"] = {
                    k: tensor_hash(v) for k, v in g.used_results(sample, again).items()
                }
                gate["within_build_bitexact"] = {
                    k: torch.equal(v, g.used_results(sample, again)[k])
                    for k, v in g.used_results(sample, actual).items()
                }
                if sample["deterministic"]:
                    assert all(gate["within_build_bitexact"].values()), (
                        "Deterministic replay differs"
                    )
                del replay_calls, again
            gate["output_hashes"] = {
                k: tensor_hash(v) for k, v in g.used_results(sample, actual).items()
            }
            verify_binding(command)
            life.bind(
                digest(sample),
                digest(binding),
                digest(inputs_binding(tensors, logical, kwargs)),
            )
            gate.update(status="pass", directions=list(calls))
            life.transition("GATED")
            del actual
            torch.cuda.empty_cache()
            life.transition("TIMING")
            stage = "timing"
            result["clocks_before"] = clocks()
            for direction in calls:
                result["timings"][direction] = measure(
                    calls[direction], command["warmup"], command["repeats"]
                )
            result["clocks_after"] = clocks()
            verify_binding(command)
            life.bind(
                digest(sample),
                digest(binding),
                digest(inputs_binding(tensors, logical, kwargs)),
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            support = normal_support(exc, backend, stage)
            status = (
                "baseline_unsupported"
                if support
                else "numerical_fail"
                if stage == "correctness"
                else "compile_error"
                if type(exc).__module__.startswith("cutlass")
                else "runtime_error"
            )
            gate.update(
                status=status, stage=stage, error=f"{type(exc).__name__}: {exc}"
            )
            result.update(status=status, error=gate["error"])
            # Every execution exception retires this generation conservatively.
            # Even a support rejection gets a new context, never a fallback/retry.
            cache.clear()
        finally:
            gate["peak_memory"] = {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            }
        return {"gate": gate, "result": result, "status": gate["status"]}

    respond(
        {
            "status": "ready",
            "generation": args.generation,
            "root": str(root),
            "binding": binding,
            "environment": env,
            "baseline_memory": baseline_memory,
        }
    )
    last_request = None
    for line in sys.stdin:
        command = json.loads(line)
        if command["command"] == "close":
            cache.clear()
            clean()
            return
        if command["command"] == "ack":
            assert command["request_id"] == last_request, "Stale acknowledgement"
            respond(
                {
                    "status": "ack",
                    "generation": args.generation,
                    "request_id": last_request,
                }
            )
            continue
        verify_binding(command)
        if command["command"] == "release-group":
            assert life.state == "EMPTY"
            cache.clear()
            memory = clean()
            assert memory["allocated"] == baseline_memory["allocated"], (
                "Live tensors remain after group release"
            )
            response = {"status": "pass", "memory": memory, "released": True}
        else:
            assert command["command"] == "execute"
            response = execute(command)
            # execute() has returned: its frames, callables, autograd graphs,
            # input tensors and exception traceback references are gone.
            if life.state != "EMPTY":
                life.transition("RELEASING")
                life.transition("EMPTY")
            try:
                memory = clean()
                expected = baseline_memory["allocated"] + cache.get("allocated", 0)
                response["post_release_memory"] = {
                    **memory,
                    "expected_allocated": expected,
                    "retained_reference_storage_bytes": cache.get("storage_bytes", 0),
                }
                assert memory["allocated"] == expected, (
                    "Unexplained live allocation after graph release"
                )
                response["released"] = True
            except Exception as exc:  # noqa: BLE001
                response.update(
                    status="release_error",
                    error=f"{type(exc).__name__}: {exc}",
                    released=False,
                )
        last_request = command["request_id"]
        response.update(generation=args.generation, request_id=last_request)
        response["retire"] = response["status"] != "pass"
        respond(response)
        if response["retire"]:
            return


class PersistentWorker(Worker):
    def __init__(self, command, root, log, env, generation, identity_root):
        super().__init__(command, root, log, env)
        self.generation, self.identity_root = generation, identity_root
        try:
            assert self.gate.get("status") == "ready", self.gate
            assert (
                self.gate.get("generation") == generation
                and self.gate.get("root") == identity_root
            ), self.gate
        except Exception:
            self.close()
            raise

    def execute(self, command):
        response = self.send(command)
        valid = (
            response.get("generation") == self.generation
            and response.get("request_id") == command["request_id"]
        )
        if not valid:
            return {
                "status": "worker_error",
                "error": "Missing/stale worker response",
                "response": response,
                "retire": True,
            }
        if response.get("retire"):
            code = self.process.wait()
            response["worker_exit_code"] = code
            if code != 0:
                response.update(
                    status="worker_error",
                    error=f"Worker failed to retire cleanly: {code}",
                )
            return response
        ack = self.send({"command": "ack", "request_id": command["request_id"]})
        if ack != {
            "status": "ack",
            "generation": self.generation,
            "request_id": command["request_id"],
        }:
            return {
                "status": "worker_error",
                "error": "Release acknowledgement missing",
                "response": response,
                "ack": ack,
                "retire": True,
            }
        return response

    def close(self):
        super().close()
        return self.process.returncode


def normal_grid(args, report, samples, roots, output):
    workers, generations = {}, {"candidate": 0, "pin": 0}
    report["protocol"].update(
        alternation="Fixed serial order; normal-grid comparisons are not paired measurements",
        acceptance="Every backend requires its own unchanged strict numerical gate before timing",
        backend_order=list(GRID_ORDER),
        reference_groups=[
            ["candidate", "sdpa-cudnn", "sdpa-flash", "sdpa-efficient", "sdpa-math"],
            ["pin"],
            ["anchor"],
        ],
    )
    report["worker_generations"], report["execution_order"] = [], []
    frozen_bindings = {
        key: file_binding(
            roots[key].resolve(), args.candidate_root.resolve() / "tests/cute"
        )
        for key in ("candidate", "pin")
        if roots[key] is not None
    }
    report["frozen_worker_bindings"] = frozen_bindings
    export(report, output)

    def get_worker(key):
        if key in workers:
            return workers[key]
        root = roots[key].resolve()
        generations[key] += 1
        generation = generations[key]
        env = {
            **os.environ,
            "PYTHONPATH": str(root),
            "CUTE_DSL_ARCH": f"sm_{args.arch}a",
            "FA4_GEMMA_EXPECT_ARCH": str(args.arch),
            "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
            "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": str(output / "cache" / key),
        }
        command = [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            str(Path(__file__).resolve()),
            "--persistent-worker",
            "--root",
            str(root),
            "--harness-root",
            str(args.candidate_root.resolve()),
            "--generation",
            str(generation),
        ]
        instance = PersistentWorker(
            command,
            root,
            output / f"{key}-generation-{generation}.log",
            env,
            generation,
            str(root),
        )
        expected = (
            args.candidate_package_sha256
            if key == "candidate"
            else report["baseline_verification"]["pin"]["package_sha256"]
        )
        workers[key] = instance
        assert (
            instance.gate["binding"]
            == frozen_bindings[key]
            == file_binding(root, args.candidate_root.resolve() / "tests/cute")
        ), "Worker source binding mismatch"
        if expected:
            assert instance.gate["binding"]["package_sha256"] == expected, (
                "Frozen package mismatch"
            )
        report["worker_generations"].append(
            {
                "role": key,
                "generation": generation,
                "pid": instance.process.pid,
                "ready": instance.gate,
                "command": command,
            }
        )
        return instance

    def invalidate(rows, status, error):
        for row in rows:
            gate = report["gates"].get(f"{row['id']}/{row['backend']}", {})
            gate.setdefault("lifecycle_failures", []).append(
                {"status": status, "error": error}
            )
            if row["status"] == "pass":
                row.update(status=status, error=error)

    def retire(key, reason):
        instance = workers.pop(key)
        code = instance.close()
        report.setdefault("worker_shutdown", []).append(
            {
                "role": key,
                "generation": instance.generation,
                "exit_code": code,
                "reason": reason,
            }
        )
        if code:
            affected = [
                row
                for row in report["rows"]
                if (
                    report["gates"]
                    .get(f"{row['id']}/{row['backend']}", {})
                    .get("worker_role"),
                    report["gates"]
                    .get(f"{row['id']}/{row['backend']}", {})
                    .get("worker_generation"),
                )
                == (key, instance.generation)
            ]
            invalidate(
                affected,
                "worker_error",
                f"Worker {key} generation {instance.generation} exited {code}: {reason}",
            )
        return code

    def request(key, backend, sample, command_name="execute"):
        try:
            return request_inner(key, backend, sample, command_name)
        except Exception as exc:  # noqa: BLE001 - a failed backend never erases later ledger rows
            traceback.print_exc()
            if key in workers:
                retire(key, "request exception")
            response = {
                "status": "worker_error",
                "error": f"{type(exc).__name__}: {exc}",
                "retire": True,
            }
            report["execution_order"].append(
                {
                    "backend": backend,
                    "row": sample["id"],
                    "command": command_name,
                    "response": response,
                    "end_time_ns": time.time_ns(),
                }
            )
            return response

    def request_inner(key, backend, sample, command_name="execute"):
        instance = get_worker(key)
        implicit = key == "pin" or (
            backend == "candidate"
            and args.candidate_base_revision
            and args.candidate_base_revision.startswith("98c1cee")
        )
        effective, _ = effective_sample(sample, backend, implicit)
        request_id = f"{sample['id']}/{backend}/{command_name}/{generations[key]}"
        group = f"{sample['id']}/" + ("anchor" if backend == "anchor" else key)
        command = {
            "command": command_name,
            "request_id": request_id,
            "root": instance.identity_root,
            "generation": instance.generation,
            "binding": instance.gate["binding"],
            "backend": backend,
            "sample": sample,
            "effective_sample_sha256": digest(effective),
            "group": group,
            "implicit_shared_mla": bool(implicit),
            "revision": "14e3c18" if key == "pin" else "candidate",
            "warmup": 3,
            "repeats": 20,
        }
        entry = {
            "request_id": request_id,
            "backend": backend,
            "command": command_name,
            "start_time_ns": time.time_ns(),
        }
        report["execution_order"].append(entry)
        export(report, output)
        response = instance.execute(command)
        entry.update(end_time_ns=time.time_ns(), response=response)
        if response.get("retire"):
            code = retire(key, "worker retirement response")
            if code:
                response.update(
                    status="worker_error", error=f"Worker retirement exited {code}"
                )
        return response

    try:
        for sample in samples:
            if args.ids and sample["id"] not in args.ids:
                continue
            selected = [r for r in report["rows"] if r["id"] == sample["id"]]
            for backend in GRID_ORDER:
                key = "pin" if backend == "pin" else "candidate"
                if roots[key] is None:
                    continue
                response = request(key, backend, sample)
                gate, result = response.get("gate", {}), response.get("result", {})
                if gate:
                    gate.update(
                        worker_role=key, worker_generation=response["generation"]
                    )
                report["gates"][f"{sample['id']}/{backend}"] = gate or response
                report["rounds"][f"{sample['id']}/{backend}"] = (
                    [result] if result else []
                )
                for row in selected:
                    if row["backend"] != backend:
                        continue
                    row.update(
                        status=response["status"],
                        source_sha256=gate.get("provenance", {}).get("package_sha256"),
                        shape_input_sha256=gate.get(
                            "shape_input_sha256", digest(sample)
                        ),
                        dense_peak=gate.get("dense_peak"),
                        nominal_dense_peak_tflops=gate.get("dense_peak", {}).get(
                            "nominal_dense_tflops"
                        ),
                        dense_peak_source=gate.get("dense_peak", {})
                        .get("source", {})
                        .get("url"),
                    )
                    if response["status"] != "pass":
                        row["error"] = response.get(
                            "error", gate.get("error", "Worker failed")
                        )
                    elif row["direction"] in gate.get(
                        "inherited_unsupported_directions", []
                    ):
                        row.update(
                            status="inherited_api_unsupported",
                            error=gate["inherited_unsupported_reason"],
                        )
                    else:
                        measured = result["timings"][row["direction"]]
                        ms = measured["device_ms"]["median"]
                        flops = useful_flops(gate["sample"])[row["direction"]]
                        tflops = flops / (ms * 1e9) if ms > 0 else None
                        row.update(
                            median_ms=ms,
                            min_ms=measured["device_ms"]["min"],
                            max_ms=measured["device_ms"]["max"],
                            flops=flops,
                            tflops=tflops,
                            percent_dense_peak=100
                            * tflops
                            / row["nominal_dense_peak_tflops"]
                            if tflops is not None
                            else None,
                        )
                # No reference tensors cross candidate-group -> pin -> anchor.
                if backend in ("sdpa-math", "pin", "anchor") and key in workers:
                    released = request(key, backend, sample, "release-group")
                    if released["status"] != "pass":
                        members = (
                            GRID_ORDER[:5] if backend == "sdpa-math" else (backend,)
                        )
                        invalidate(
                            [row for row in selected if row["backend"] in members],
                            "release_error",
                            released.get("error", "Group release failed"),
                        )
                export(report, output)
            for row in selected:
                candidate = next(
                    r
                    for r in selected
                    if r["backend"] == "candidate"
                    and r["direction"] == row["direction"]
                )
                if (
                    candidate["status"] == row["status"] == "pass"
                    and candidate["median_ms"] > 0
                ):
                    row["ratio_vs_candidate"] = (
                        row["median_ms"] / candidate["median_ms"]
                    )
            export(report, output)
    finally:
        for key in list(workers):
            retire(key, "normal shutdown")
        export(report, output)
    failed = any(
        r["status"]
        not in ("pass", "pending", "baseline_unsupported", "inherited_api_unsupported")
        for r in report["rows"]
    )
    raise SystemExit(int(failed))


def strata(sample):
    return {
        "dimension": f"{sample['dimension']}/{sample['dimension_v']}",
        "dtype": sample["dtype"],
        "heads": f"{sample['query_heads']}/{sample['kv_heads']}",
        "storage": sample["storage"],
        "mask": "causal-local"
        if sample["causal"] and sample["window"] != [None, None]
        else "local"
        if sample["window"] != [None, None]
        else "causal"
        if sample["causal"]
        else "noncausal",
        "query_mode": sample.get(
            "query_mode",
            "prefill" if sample["q_lengths"] == sample["k_lengths"] else "rectangular",
        ),
        "length": sample.get("length_bucket", str(max(sample["k_lengths"]))),
        "deterministic": str(sample["deterministic"]),
    }


def export(report, output):
    rows, summaries = report["rows"], {}
    for row in rows:
        keys = ["all"] + [f"{key}={value}" for key, value in row["strata"].items()]
        for key in keys:
            group = summaries.setdefault(
                f"{row['backend']}/{row['direction']}/{key}", {"counts": {}, "rows": []}
            )
            group["counts"][row["status"]] = group["counts"].get(row["status"], 0) + 1
            group["rows"].append(row)
    for group in summaries.values():
        members = group.pop("rows")
        for field in (
            "median_ms",
            "tflops",
            "percent_dense_peak",
            "ratio_vs_candidate",
        ):
            group[field] = distribution(
                [
                    r[field]
                    for r in members
                    if r["status"] == "pass" and r.get(field) is not None
                ]
            )
    report["distributions"] = summaries
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = (
        "id",
        "arch",
        "backend",
        "direction",
        "status",
        "median_ms",
        "min_ms",
        "max_ms",
        "flops",
        "tflops",
        "percent_dense_peak",
        "nominal_dense_peak_tflops",
        "dense_peak_source",
        "ratio_vs_candidate",
        "source_sha256",
        "shape_input_sha256",
        "error",
    )
    with (output / "coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "coverage.md").open("w") as handle:
        handle.write(
            "# Complete benchmark coverage\n\nEvery requested row/backend/direction remains in this ledger.\n\n"
        )
        handle.write(
            "| ID | Arch | Backend | Direction | Status | Median ms | TFLOP/s | % dense peak | Baseline/candidate time |\n"
        )
        handle.write("|---|---|---|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(str(row.get(k, "—")) for k in fields[:8] if k != "min_ms")
                + f" | {row.get('tflops', '—')} | {row.get('percent_dense_peak', '—')} | {row.get('ratio_vs_candidate', '—')} |\n"
            )


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    samples, manifests = [], []
    for path in args.manifest:
        manifest = json.loads(path.read_text())
        manifests.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        samples.extend(manifest["samples"])
    assert len({s["id"] for s in samples}) == len(samples), "Duplicate manifest IDs"
    if args.ids:
        assert set(args.ids) <= {s["id"] for s in samples}, "Unknown ID"
    roots = {
        "candidate": args.candidate_root,
        "pin": args.pin_root,
        "upstream": args.upstream_root,
        "sdpa-cudnn": args.candidate_root,
        "sdpa-flash": args.candidate_root,
        "sdpa-efficient": args.candidate_root,
        "sdpa-math": args.candidate_root,
        "anchor": args.candidate_root,
    }
    backends = (
        ("candidate", "upstream", "pin")
        if args.overhead
        else tuple(b for b in BACKENDS if b != "upstream")
    )
    rows = []
    for sample in samples:
        directions = (
            ("bwd",)
            if args.overhead
            else ("fwd",)
            if sample["direction"] == "forward"
            else ("fwd", "bwd", "total")
        )
        for backend in backends:
            for direction in directions:
                rows.append(
                    {
                        "id": sample["id"],
                        "arch": args.arch,
                        "backend": backend,
                        "direction": direction,
                        "status": "pending",
                        "strata": strata(sample),
                        "shape_input_sha256": digest(sample),
                    }
                )
    report = {
        "arch": args.arch,
        "manifests": manifests,
        "rows": rows,
        "protocol": {
            "overhead": args.overhead,
            "rounds": 15 if args.overhead else 1,
            "iterations": 30 if args.overhead else 20,
            "warmup": 3,
            "dense_peak_policy": "Bound to observed device name, capability and dtype in each worker gate; unknown bindings fail before timing",
            "dense_peak_source": PEAK_SOURCE,
            "alternation": "candidate/upstream then upstream/candidate each successive round; pin afterward",
            "acceptance": "upstream/PR backward median overhead within +/-2%; report interval and every round",
            "sdpa_boundary": "Per-sequence calls on logical tensors; paged storage conversion outside timing; efficient GQA K/V expansion inside timed forward; other backends use native enable_gqa; exactly one explicit backend, no fallback",
            "baseline_revisions": {"pin": "14e3c18", "upstream": "98c1cee"},
            "useful_flops": "QK+PV 2(D+Dv); backward 6D+4Dv per unmasked head pair; MLA adds Dv to score D",
        },
        "gates": {},
        "rounds": {},
    }
    report["baseline_verification"] = {}
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["candidate_provenance"] = {
        "root": str(args.candidate_root.resolve()),
        "base_revision": args.candidate_base_revision,
        "patch_sha256": hashlib.sha256(args.candidate_patch.read_bytes()).hexdigest()
        if args.candidate_patch
        else None,
        "patch_path": str(args.candidate_patch.resolve())
        if args.candidate_patch
        else None,
        "expected_package_sha256": args.candidate_package_sha256,
    }
    frozen_baselines = (
        json.loads(args.baseline_provenance.read_text())
        if args.baseline_provenance
        else {}
    )
    if not args.prepare_only:
        for backend, revision in (("pin", "14e3c18"), ("upstream", "98c1cee")):
            if backend in backends and roots[backend] is not None:
                report["baseline_verification"][backend] = verify_baseline(
                    roots[backend].resolve(),
                    revision,
                    args.candidate_root.resolve(),
                    frozen_baselines.get(backend),
                )
    export(report, output)
    if args.prepare_only:
        return
    if not args.overhead:
        return normal_grid(args, report, samples, roots, output)
    for sample in samples:
        if args.ids and sample["id"] not in args.ids:
            continue
        directory = output / sample["id"]
        directory.mkdir()
        sample_file = directory / "sample.json"
        sample_file.write_text(json.dumps(sample, sort_keys=True) + "\n")
        workers = {}
        selected = [row for row in rows if row["id"] == sample["id"]]
        try:
            for backend in backends:
                root = roots[backend]
                if root is None:
                    continue  # Missing baseline is explicitly pending, never unsupported.
                root = root.resolve()
                env = {
                    **os.environ,
                    "PYTHONPATH": str(root),
                    "CUTE_DSL_ARCH": f"sm_{args.arch}a",
                    "FA4_GEMMA_EXPECT_ARCH": str(args.arch),
                    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
                    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": str(
                        output / "cache" / backend
                    ),
                }
                command = [
                    "uv",
                    "run",
                    "--no-project",
                    "--python",
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--root",
                    str(root),
                    "--backend",
                    backend,
                    "--revision",
                    {"pin": "14e3c18", "upstream": "98c1cee"}.get(backend, "candidate"),
                    "--sample",
                    str(sample_file),
                ]
                if args.overhead:
                    command += [
                        "--save-outputs",
                        str(directory / f"{backend}-outputs.pt"),
                    ]
                if backend in ("pin", "upstream") or (
                    backend == "candidate"
                    and args.candidate_base_revision
                    and args.candidate_base_revision.startswith("98c1cee")
                ):
                    command += ["--implicit-shared-mla"]
                worker_instance = Worker(
                    command, root, directory / f"{backend}.log", env
                )
                workers[backend] = worker_instance
                gate = worker_instance.gate
                if (
                    backend == "candidate"
                    and args.candidate_package_sha256
                    and (
                        gate.get("provenance", {}).get("package_sha256")
                        != args.candidate_package_sha256
                    )
                ):
                    gate.update(
                        status="source_mismatch",
                        error="Candidate differs from frozen expected package hash",
                    )
                verified = report["baseline_verification"].get(backend)
                if (
                    verified
                    and gate.get("provenance", {}).get("package_sha256")
                    != verified["package_sha256"]
                ):
                    gate.update(
                        status="source_mismatch",
                        error="Loaded baseline differs from verified Git snapshot",
                    )
                report["gates"][f"{sample['id']}/{backend}"] = gate
                for row in selected:
                    if row["backend"] == backend:
                        row.update(
                            status=gate["status"]
                            if gate["status"] != "pass"
                            else "pending",
                            source_sha256=gate.get("provenance", {}).get(
                                "package_sha256"
                            ),
                            shape_input_sha256=gate.get(
                                "shape_input_sha256", digest(sample)
                            ),
                            dense_peak=gate.get("dense_peak"),
                            nominal_dense_peak_tflops=gate.get("dense_peak", {}).get(
                                "nominal_dense_tflops"
                            ),
                            dense_peak_source=gate.get("dense_peak", {})
                            .get("source", {})
                            .get("url"),
                        )
                        if "error" in gate:
                            row["error"] = gate["error"]
                        if row["direction"] in gate.get(
                            "inherited_unsupported_directions", []
                        ):
                            row.update(
                                status="inherited_api_unsupported",
                                error=gate["inherited_unsupported_reason"],
                            )
                export(report, output)
            if args.overhead:
                cg = report["gates"].get(f"{sample['id']}/candidate", {})
                ug = report["gates"].get(f"{sample['id']}/upstream", {})
                compared = cg.get("status") == ug.get("status") == "pass"
                evidence = {
                    "status": "compared" if compared else "pending",
                    "input_hashes_equal": cg.get("input_hashes")
                    == ug.get("input_hashes")
                    if compared
                    else None,
                    "output_hashes_equal": cg.get("output_hashes")
                    == ug.get("output_hashes")
                    if compared
                    else None,
                    "candidate": cg.get("output_hashes"),
                    "upstream": ug.get("output_hashes"),
                }
                report.setdefault("bitwise_comparisons", {})[sample["id"]] = evidence
                if compared and not (
                    evidence["input_hashes_equal"] and evidence["output_hashes_equal"]
                ):
                    # Preserve complete outputs; report the mismatch before timing.
                    print(
                        f"FINITE_INPUT_DIFFERENCE {sample['id']} {json.dumps(evidence)}",
                        flush=True,
                    )
                    for backend in ("candidate", "upstream"):
                        workers[backend].gate["status"] = "finite_input_difference"
                        for row in selected:
                            if row["backend"] == backend:
                                row.update(
                                    status="finite_input_difference",
                                    error="Main/PR finite outputs are not bit-exact; timing withheld",
                                )
                    export(report, output)
            for round_index in range(report["protocol"]["rounds"]):
                order = (
                    ("candidate", "upstream", "pin")
                    if round_index % 2 == 0
                    else ("upstream", "candidate", "pin")
                )
                if not args.overhead:
                    order = backends
                for backend in order:
                    if (
                        backend not in workers
                        or workers[backend].gate["status"] != "pass"
                    ):
                        continue
                    target_rows = [
                        r
                        for r in selected
                        if r["backend"] == backend
                        and r["direction"]
                        in workers[backend].gate.get("directions", [])
                        and r["status"] in ("pending", "pass")
                    ]
                    if not target_rows:
                        continue
                    result = workers[backend].send(
                        {
                            "command": "time",
                            "round": round_index,
                            "directions": [r["direction"] for r in target_rows],
                            "warmup": 3,
                            "repeats": report["protocol"]["iterations"],
                        }
                    )
                    report["rounds"].setdefault(f"{sample['id']}/{backend}", []).append(
                        result
                    )
                    for row in target_rows:
                        if result["status"] != "pass":
                            row.update(status=result["status"], error=result["error"])
                            continue
                        rounds = report["rounds"][f"{sample['id']}/{backend}"]
                        values = [
                            r["timings"][row["direction"]]["device_ms"]["median"]
                            for r in rounds
                            if r["status"] == "pass"
                        ]
                        all_values = [
                            value
                            for r in rounds
                            if r["status"] == "pass"
                            for value in r["timings"][row["direction"]]["all_device_ms"]
                        ]
                        measured_sample = workers[backend].gate["sample"]
                        flops = useful_flops(measured_sample)[row["direction"]]
                        median = statistics.median(values)
                        row.update(
                            status="pass",
                            median_ms=median,
                            min_ms=min(all_values),
                            max_ms=max(all_values),
                            flops=flops,
                            tflops=flops / (median * 1e9) if median > 0 else None,
                        )
                        row["percent_dense_peak"] = (
                            row["tflops"] * 100 / row["nominal_dense_peak_tflops"]
                            if row["tflops"] is not None
                            else None
                        )
                    export(report, output)
            for row in selected:
                candidate = next(
                    (
                        r
                        for r in selected
                        if r["backend"] == "candidate"
                        and r["direction"] == row["direction"]
                    ),
                    None,
                )
                if (
                    candidate
                    and candidate["status"] == row["status"] == "pass"
                    and candidate["median_ms"] > 0
                ):
                    row["ratio_vs_candidate"] = (
                        row["median_ms"] / candidate["median_ms"]
                    )
            if args.overhead:
                cr = report["rounds"].get(f"{sample['id']}/candidate", [])
                for baseline in ("upstream", "pin"):
                    br = report["rounds"].get(f"{sample['id']}/{baseline}", [])
                    report.setdefault(f"{baseline}_overhead", {})[sample["id"]] = (
                        overhead_assessment(cr, br)
                    )
            export(report, output)
        finally:
            for instance in workers.values():
                instance.close()
    failed = any(
        row["status"]
        not in ("pass", "pending", "baseline_unsupported", "inherited_api_unsupported")
        for row in rows
    )
    if args.overhead:
        assessed = report.get("upstream_overhead", {})
        report["upstream_performance_gate"] = {
            "status": "incomplete"
            if len(assessed) != len(samples)
            or any(r["status"] == "incomplete" for r in assessed.values())
            else "within_2_percent"
            if all(r["status"] == "within_2_percent" for r in assessed.values())
            else "outside_2_percent",
            "requested_shapes": len(samples),
            "assessed_shapes": len(assessed),
        }
        failed |= any(r["regression_over_2_percent"] is True for r in assessed.values())
        export(report, output)
    raise SystemExit(int(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--candidate-base-revision")
    parser.add_argument("--candidate-patch", type=Path)
    parser.add_argument(
        "--candidate-package-sha256",
        help="Independently frozen expected candidate package hash",
    )
    parser.add_argument("--pin-root", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument(
        "--baseline-provenance",
        type=Path,
        help="Frozen Git blob IDs generated on a checkout; required for archived remote roots without .git",
    )
    parser.add_argument("--arch", type=int, choices=(100, 103))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--overhead", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--persistent-worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--harness-root")
    parser.add_argument("--generation", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--revision")
    parser.add_argument("--sample")
    parser.add_argument("--save-outputs")
    parser.add_argument(
        "--implicit-shared-mla", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.persistent_worker:
        persistent_worker(args)
    elif args.worker:
        worker(args)
    else:
        assert args.manifest and args.candidate_root and args.arch and args.output
        run(args)


if __name__ == "__main__":
    main()
