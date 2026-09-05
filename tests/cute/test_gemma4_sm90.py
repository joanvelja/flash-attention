import cutlass
import cutlass.cute as cute
import pytest
import torch
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func, utils
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

pytestmark = pytest.mark.gpu


def _assert_sm90() -> None:
    assert torch.cuda.is_available(), "Gemma 4 FA4 canaries require CUDA"
    assert torch.cuda.get_device_capability() == (9, 0), "Gemma 4 FA4 canaries require SM90"


def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).norm() / expected.float().norm())


def _reference_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
    window: tuple[int | None, int | None],
    softcap: float | None = None,
    dense_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_q, sequence_k = q.shape[0], k.shape[0]
    repeats = q.shape[1] // k.shape[1]
    reference_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q_float = q.to(reference_dtype).transpose(0, 1)
    k_float = k.to(reference_dtype).repeat_interleave(repeats, dim=1).transpose(0, 1)
    v_float = v.to(reference_dtype).repeat_interleave(repeats, dim=1).transpose(0, 1)
    scores = torch.matmul(q_float, k_float.transpose(-1, -2)) * scale
    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)

    query_idx = torch.arange(sequence_q, device=q.device).unsqueeze(1) + (sequence_k - sequence_q)
    key_idx = torch.arange(sequence_k, device=q.device).unsqueeze(0)
    allowed = torch.ones((sequence_q, sequence_k), dtype=torch.bool, device=q.device)
    if causal:
        allowed &= key_idx <= query_idx
    window_left, window_right = window
    if window_left is not None:
        allowed &= key_idx >= query_idx - window_left
    if window_right is not None:
        allowed &= key_idx <= query_idx + window_right
    if dense_mask is not None:
        allowed &= dense_mask
    scores = scores.masked_fill(~allowed, -torch.inf)
    # Avoid evaluating softmax on all -inf rows: masking NaNs afterward also
    # leaves NaNs in backward. Empty attention has zero output and derivatives.
    valid_rows = allowed.any(-1, keepdim=True)
    safe_scores = torch.where(valid_rows, scores, 0.0)
    probabilities = safe_scores.softmax(-1) * valid_rows
    output = torch.matmul(probabilities, v_float).transpose(0, 1)
    return output, torch.logsumexp(scores, dim=-1)


def _reference_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lengths: list[int],
    k_lengths: list[int],
    scale: float,
    causal: bool,
    window: tuple[int | None, int | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    logsumexp = []
    q_start = 0
    k_start = 0
    for q_length, k_length in zip(q_lengths, k_lengths):
        output, lse = _reference_dense(
            q[q_start : q_start + q_length],
            k[k_start : k_start + k_length],
            v[k_start : k_start + k_length],
            scale,
            causal,
            window,
        )
        outputs.append(output.to(q.dtype))
        logsumexp.append(lse)
        q_start += q_length
        k_start += k_length
    return torch.cat(outputs), torch.cat(logsumexp, dim=1)


@pytest.mark.parametrize(
    ("head_dim", "query_lengths", "key_lengths", "causal", "scale", "window", "check_backward"),
    [
        (64, [197, 65], [197, 65], False, None, (None, None), True),
        (128, [73, 129], [137, 193], True, 0.37, (None, None), True),
        (256, [1057, 511], [1057, 511], True, 1.0, (1023, 0), True),
        (512, [257, 129], [513, 385], True, 1.0, (None, None), True),
        (512, [200, 129], [264, 385], False, 1.0, (127, 15), False),
        (512, [0, 257, 129], [64, 513, 385], True, 1.0, (None, None), False),
    ],
    ids=(
        "legacy-d64",
        "legacy-d128-bottom-right",
        "gemma-local-d256",
        "gemma-global-d512",
        "gemma-local-d512-forward",
        "gemma-zerolen-d512-forward",
    ),
)
def test_gemma4_sm90_varlen_canary(
    head_dim: int,
    query_lengths: list[int],
    key_lengths: list[int],
    causal: bool,
    scale: float | None,
    window: tuple[int | None, int | None],
    check_backward: bool,
) -> None:
    _assert_sm90()
    torch.manual_seed(head_dim)
    query_heads = 16 if head_dim >= 256 else 8
    key_value_heads = 8 if head_dim == 256 else 2
    q = torch.randn(
        sum(query_lengths), query_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=check_backward
    )
    k = torch.randn(
        sum(key_lengths), key_value_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=check_backward
    )
    v = torch.randn_like(k, requires_grad=check_backward)
    q_ref = q.detach().clone().requires_grad_(check_backward)
    k_ref = k.detach().clone().requires_grad_(check_backward)
    v_ref = v.detach().clone().requires_grad_(check_backward)
    effective_scale = head_dim**-0.5 if scale is None else scale

    output, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=_cu_seqlens(query_lengths),
        cu_seqlens_k=_cu_seqlens(key_lengths),
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=max(key_lengths),
        softmax_scale=scale,
        causal=causal,
        window_size=window,
        return_lse=True,
    )
    output_ref, lse_ref = _reference_varlen(
        q_ref,
        k_ref,
        v_ref,
        query_lengths,
        key_lengths,
        effective_scale,
        causal,
        window,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output.float(), output_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse.float(), lse_ref.float(), atol=3e-2, rtol=3e-2)
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()

    if check_backward:
        output_gradient = torch.randn_like(output)
        output.backward(output_gradient)
        output_ref.backward(output_gradient)
        for name, actual, expected in (
            ("dQ", q.grad, q_ref.grad),
            ("dK", k.grad, k_ref.grad),
            ("dV", v.grad, v_ref.grad),
        ):
            relative_l2 = _relative_l2(actual, expected)
            print(
                f"{name}: relative_l2={relative_l2:.8f}, "
                f"max_abs={float((actual.float() - expected.float()).abs().max()):.8f}",
                flush=True,
            )
            assert relative_l2 < 1e-2
            assert torch.isfinite(actual).all()


@pytest.mark.parametrize(("query_heads", "key_value_heads"), [(4, 1), (8, 1)], ids=("gqa4", "gqa8"))
def test_gemma4_sm90_d512_production_bf16_varlen(query_heads: int, key_value_heads: int) -> None:
    """The Gemma 4 global-attention production contract, and that the fast path is taken.

    BF16, head dim 512, causal varlen, global window, softmax scale 1.0. The ring wrapper
    gathers one KV head per call, so the inner call is GQA4 (E4B) or GQA8 (26B-A4B, 31B).
    Tensors here are compact; the noncompact ring head slices are the replay harness's job.

    The selector assertion is deliberate coupling: a numerical check alone still passes if
    the score-publication gate is accidentally closed, since the generic path is correct too.
    """
    _assert_sm90()
    torch.manual_seed(512 + query_heads)
    lengths = [4096, 3000, 1096]
    q = torch.randn(sum(lengths), query_heads, 512, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(lengths), key_value_heads, 512, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    # The gate is only consulted when a kernel is actually compiled, so an earlier test that
    # already cached this configuration would leave `selected` empty and make the assertion
    # vacuous. Force the compile.
    _flash_attn_fwd.compile_cache.clear()

    selected: list[bool] = []
    original = FlashAttentionForwardSm90._supports_cta_score_publication

    def record(self, *args, **kwargs):
        chosen = original(self, *args, **kwargs)
        selected.append(bool(chosen))
        return chosen

    FlashAttentionForwardSm90._supports_cta_score_publication = record
    try:
        output, lse = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=_cu_seqlens(lengths),
            cu_seqlens_k=_cu_seqlens(lengths),
            max_seqlen_q=max(lengths),
            max_seqlen_k=max(lengths),
            softmax_scale=1.0,
            causal=True,
            return_lse=True,
        )
    finally:
        FlashAttentionForwardSm90._supports_cta_score_publication = original

    assert selected, "the score-publication gate was never consulted"
    assert all(selected), (
        "score publication must stay selected for the production contract "
        f"(bf16, causal varlen, global, GQA{query_heads // key_value_heads})"
    )

    output_ref, lse_ref = _reference_varlen(
        q, k, v, lengths, lengths, 1.0, True, (None, None)
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), output_ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse.float(), lse_ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=("fp16", "bf16"))
@pytest.mark.parametrize(
    ("query_heads", "key_value_heads"),
    [(8, 2), (16, 2), (16, 1)],
    ids=("gqa4", "gqa8", "gqa16"),
)
def test_gemma4_sm90_d512_batch_backward(
    query_heads: int,
    key_value_heads: int,
    dtype: torch.dtype,
) -> None:
    _assert_sm90()
    torch.manual_seed(2512 + query_heads + key_value_heads)
    query_length, key_length = 129, 257
    q = torch.randn(1, query_length, query_heads, 512, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(1, key_length, key_value_heads, 512, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()

    output, _ = flash_attn_func(q, k, v, causal=True, return_lse=True)
    output_ref, _ = _reference_dense(
        q_ref[0],
        k_ref[0],
        v_ref[0],
        512**-0.5,
        True,
        (None, None),
    )
    output_ref = output_ref.to(q.dtype).unsqueeze(0)
    torch.testing.assert_close(output.float(), output_ref.float(), atol=3e-2, rtol=3e-2)

    output_gradient = torch.randn_like(output)
    output.backward(output_gradient)
    output_ref.backward(output_gradient)
    for name, actual, expected in (
        ("dQ", q.grad, q_ref.grad),
        ("dK", k.grad, k_ref.grad),
        ("dV", v.grad, v_ref.grad),
    ):
        relative_l2 = _relative_l2(actual, expected)
        print(
            f"{dtype}-d512 {name}: relative_l2={relative_l2:.8f}, "
            f"max_abs={float((actual.float() - expected.float()).abs().max()):.8f}",
            flush=True,
        )
        assert relative_l2 < 1e-2
        assert torch.isfinite(actual).all()


def test_gemma4_sm90_d512_batch_softcap_fp16() -> None:
    _assert_sm90()
    torch.manual_seed(1512)
    q = torch.randn(2, 257, 4, 512, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 257, 4, 512, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    output, lse = flash_attn_func(
        q,
        k,
        v,
        softmax_scale=0.05,
        causal=True,
        softcap=30.0,
        return_lse=True,
    )
    torch.cuda.synchronize()
    for batch_idx in range(2):
        output_ref, lse_ref = _reference_dense(
            q[batch_idx],
            k[batch_idx],
            v[batch_idx],
            0.05,
            True,
            (None, None),
            softcap=30.0,
        )
        torch.testing.assert_close(output[batch_idx].float(), output_ref, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(lse[batch_idx].float(), lse_ref, atol=3e-2, rtol=3e-2)


def test_gemma4_sm90_d512_batch_decode_and_auto_num_splits() -> None:
    _assert_sm90()
    torch.manual_seed(2512)
    q = torch.randn(3, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 999, 2, 512, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    output, lse = flash_attn_func(q, k, v, causal=True, return_lse=True)
    output_auto, lse_auto = flash_attn_func(q, k, v, causal=True, return_lse=True, num_splits=0)
    torch.cuda.synchronize()
    assert torch.equal(output, output_auto)
    assert torch.equal(lse, lse_auto)
    for batch_idx in range(3):
        output_ref, lse_ref = _reference_dense(
            q[batch_idx],
            k[batch_idx],
            v[batch_idx],
            512**-0.5,
            True,
            (None, None),
        )
        torch.testing.assert_close(output[batch_idx].float(), output_ref, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(lse[batch_idx].float(), lse_ref, atol=3e-2, rtol=3e-2)


@cute.jit
def _mask_mod_causal_hole(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
    offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
    offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
    lower = utils.scalar_to_ssa(32, cutlass.Int32)
    upper = utils.scalar_to_ssa(48, cutlass.Int32)
    return (n_idx <= (m_idx + offset_ssa)) & ((n_idx < lower) | (n_idx >= upper))


@cute.jit
def _mask_mod_causal(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
    offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
    offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
    return n_idx <= (m_idx + offset_ssa)


def test_gemma4_sm90_d512_mask_mod() -> None:
    _assert_sm90()
    torch.manual_seed(3512)
    sequence_q, sequence_k = 200, 264
    q = torch.randn(2, sequence_q, 4, 512, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, sequence_k, 4, 512, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    output, lse, *_ = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        softmax_scale=0.08,
        causal=False,
        mask_mod=_mask_mod_causal_hole,
        return_lse=True,
    )
    torch.cuda.synchronize()
    query_idx = torch.arange(sequence_q, device="cuda").unsqueeze(1) + (sequence_k - sequence_q)
    key_idx = torch.arange(sequence_k, device="cuda").unsqueeze(0)
    dense_mask = (key_idx <= query_idx) & ((key_idx < 32) | (key_idx >= 48))
    for batch_idx in range(2):
        output_ref, lse_ref = _reference_dense(
            q[batch_idx],
            k[batch_idx],
            v[batch_idx],
            0.08,
            False,
            (None, None),
            dense_mask=dense_mask,
        )
        torch.testing.assert_close(output[batch_idx].float(), output_ref, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(lse[batch_idx].float(), lse_ref, atol=3e-2, rtol=3e-2)


def test_gemma4_sm90_d512_block_sparse() -> None:
    _assert_sm90()
    torch.manual_seed(4512)
    batch, heads, sequence_q, sequence_k = 1, 4, 320, 320
    q = torch.randn(batch, sequence_q, heads, 512, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, sequence_k, heads, 512, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sparse_tensors = compute_block_sparsity(
        tile_m=64,
        tile_n=64,
        batch_size=batch,
        num_heads=heads,
        seqlen_q=sequence_q,
        seqlen_k=sequence_k,
        mask_mod=_mask_mod_causal,
        aux_tensors=None,
        device="cuda",
    )
    mask_count, mask_index, full_count, full_index, *_ = sparse_tensors
    block_sparse = BlockSparseTensorsTorch(
        mask_block_cnt=mask_count,
        mask_block_idx=mask_index,
        full_block_cnt=full_count,
        full_block_idx=full_index,
        block_size=(64, 64),
    )
    output, lse, *_ = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        softmax_scale=0.07,
        causal=False,
        mask_mod=_mask_mod_causal,
        block_sparse_tensors=block_sparse,
        return_lse=True,
    )
    torch.cuda.synchronize()
    output_ref, lse_ref = _reference_dense(q[0], k[0], v[0], 0.07, True, (None, None))
    torch.testing.assert_close(output[0].float(), output_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse[0].float(), lse_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=("bf16", "fp16"))
@pytest.mark.parametrize(("query_heads", "parent_heads"), [(4, 8), (8, 16), (16, 16)])
def test_gemma4_sm90_d512_strided_rms_backward(
    dtype: torch.dtype, query_heads: int, parent_heads: int
) -> None:
    """Scale-one backward with the noncompact head slices used by ring attention."""
    _assert_sm90()
    torch.manual_seed(20260905 + query_heads)
    sequence_q, sequence_k, head_dim = 129, 257, 512
    q_parent = torch.randn(sequence_q, parent_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(sequence_k, 1, head_dim, device="cuda", dtype=dtype)
    q_parent = (q_parent.float() * q_parent.float().square().mean(-1, keepdim=True).rsqrt()).to(dtype)
    k = (k.float() * k.float().square().mean(-1, keepdim=True).rsqrt()).to(dtype)
    v = torch.randn_like(k)
    offset = query_heads if parent_heads > query_heads else 0
    head_slice = slice(offset, offset + query_heads)
    q = q_parent[:, head_slice]
    out = torch.empty_like(q_parent)[:, head_slice]
    dout = torch.randn_like(q_parent)[:, head_slice]
    dq = torch.empty_like(q_parent)[:, head_slice]
    q_ref, k_ref, v_ref = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    kwargs = dict(
        cu_seqlens_q=_cu_seqlens([sequence_q]),
        cu_seqlens_k=_cu_seqlens([sequence_k]),
        max_seqlen_q=sequence_q,
        max_seqlen_k=sequence_k,
        softmax_scale=1.0,
        causal=True,
    )
    actual_out, lse, *_ = _flash_attn_fwd(q=q, k=k, v=v, out=out, return_lse=True, **kwargs)
    gradients = _flash_attn_bwd(q, k, v, out, dout, lse, dq=dq, **kwargs)
    expected_out, expected_lse = _reference_dense(q_ref, k_ref, v_ref, 1.0, True, (None, None))
    expected_out.backward(dout.float())
    assert actual_out.data_ptr() == out.data_ptr()
    assert gradients[0].data_ptr() == dq.data_ptr()
    assert out.stride() == dout.stride() == dq.stride() == q.stride()
    for name, actual, expected in (
        ("O", actual_out, expected_out),
        ("LSE", lse, expected_lse),
        ("dQ", gradients[0], q_ref.grad),
        ("dK", gradients[1], k_ref.grad),
        ("dV", gradients[2], v_ref.grad),
    ):
        assert torch.isfinite(actual).all()
        difference = actual.double() - expected.double()
        relative_l2 = float(difference.norm() / expected.double().norm())
        max_abs = float(difference.abs().max())
        print(f"{name}: relative_l2={relative_l2:.8f}, max_abs={max_abs:.8f}", flush=True)
        assert relative_l2 <= 1e-2
        if name in ("O", "LSE"):
            assert max_abs <= 3e-2


@pytest.mark.parametrize(
    ("head_dim", "query_heads", "parent_heads", "window_left", "dtype"),
    [
        (256, 2, 32, 1023, torch.bfloat16),
        (256, 4, 8, 511, torch.bfloat16),
        (256, 8, 8, 511, torch.bfloat16),
        (256, 4, 8, 511, torch.float16),
        (512, 8, 32, None, torch.bfloat16),
    ],
    ids=("local-gqa2", "local-gqa4", "local-gqa8", "local-gqa4-fp16", "global-gqa8"),
)
def test_gemma4_sm90_ragged_empty_strided_backward(
    head_dim: int,
    query_heads: int,
    parent_heads: int,
    window_left: int | None,
    dtype: torch.dtype,
) -> None:
    """Empty rows, window eviction and ring head views against FP64 attention."""
    _assert_sm90()
    torch.manual_seed(20260905)
    query_lengths = [0, 65, 17, 129]
    key_lengths = [31, 31, 0, 257 if window_left is None else window_left + 129]
    q_parent = torch.randn(sum(query_lengths), parent_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(sum(key_lengths), 1, head_dim, device="cuda", dtype=dtype)
    q_parent = (q_parent.float() * q_parent.float().square().mean(-1, keepdim=True).rsqrt()).to(dtype)
    k = (k.float() * k.float().square().mean(-1, keepdim=True).rsqrt()).to(dtype)
    v = torch.randn_like(k)
    offset = query_heads if parent_heads > query_heads else 0
    head_slice = slice(offset, offset + query_heads)
    out_parent = torch.full_like(q_parent, torch.nan)
    dq_parent = torch.full_like(q_parent, torch.nan)
    q = q_parent[:, head_slice]
    out = out_parent[:, head_slice]
    dout = torch.randn_like(q_parent)[:, head_slice]
    dq = dq_parent[:, head_slice]
    q_ref, k_ref, v_ref = (x.detach().double().requires_grad_(True) for x in (q, k, v))
    window = (window_left, 0 if window_left is not None else None)
    kwargs = dict(
        cu_seqlens_q=_cu_seqlens(query_lengths),
        cu_seqlens_k=_cu_seqlens(key_lengths),
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=max(key_lengths),
        softmax_scale=1.0,
        causal=True,
        window_size_left=window[0],
        window_size_right=window[1],
    )
    actual_out, lse, *_ = _flash_attn_fwd(q=q, k=k, v=v, out=out, return_lse=True, **kwargs)
    gradients = _flash_attn_bwd(q, k, v, out, dout, lse, dq=dq, **kwargs)
    expected_out, expected_lse = _reference_varlen(
        q_ref, k_ref, v_ref, query_lengths, key_lengths, 1.0, True, window
    )
    expected_out.backward(dout.double())
    assert actual_out.data_ptr() == out.data_ptr()
    assert gradients[0].data_ptr() == dq.data_ptr()
    assert out.stride() == dout.stride() == dq.stride() == q.stride()
    for parent in (out_parent, dq_parent):
        assert torch.isnan(parent[:, :offset]).all()
        assert torch.isnan(parent[:, offset + query_heads :]).all()

    # Document 1 has 65 queries / 31 keys: its first 34 rows are fully masked.
    # Document 2 has queries but no keys. Document 0 has keys but no queries.
    for rows in (slice(0, 34), slice(65, 82)):
        assert torch.count_nonzero(actual_out[rows]) == 0
        assert torch.count_nonzero(gradients[0][rows]) == 0
        assert torch.isneginf(lse[:, rows]).all()
    for gradient in gradients[1:]:
        assert torch.count_nonzero(gradient[:31]) == 0

    for name, actual, expected in (
        ("O", actual_out, expected_out),
        ("LSE", lse, expected_lse),
        ("dQ", gradients[0], q_ref.grad),
        ("dK", gradients[1], k_ref.grad),
        ("dV", gradients[2], v_ref.grad),
    ):
        assert not torch.isnan(actual).any()
        assert not torch.isposinf(actual).any()
        assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
        finite = torch.isfinite(expected)
        difference = actual.double()[finite] - expected[finite]
        relative_l2 = float(difference.norm() / expected[finite].norm())
        max_abs = float(difference.abs().max())
        print(f"{name}: relative_l2={relative_l2:.8f}, max_abs={max_abs:.8f}", flush=True)
        assert relative_l2 <= 1e-2
        if name in ("O", "LSE"):
            assert max_abs <= 3e-2


def test_gemma4_sm90_d512_deterministic_backward_is_rejected() -> None:
    _assert_sm90()
    q = torch.randn(8, 16, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(8, 2, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    cu_seqlens = _cu_seqlens([8])
    output, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=8,
        max_seqlen_k=8,
        softmax_scale=1.0,
        causal=True,
        deterministic=True,
    )

    with pytest.raises(NotImplementedError, match="deterministic backward"):
        output.sum().backward()


@pytest.mark.parametrize(("head_dim", "head_dim_v"), [(384, 384), (512, 256)])
def test_sm90_backward_rejects_unvalidated_large_head_dims(head_dim: int, head_dim_v: int) -> None:
    _assert_sm90()
    q = torch.zeros(1, 8, 16, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 8, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros(1, 8, 2, head_dim_v, device="cuda", dtype=torch.bfloat16)
    out = torch.zeros(1, 8, 16, head_dim_v, device="cuda", dtype=torch.bfloat16)
    dout = torch.zeros_like(out)
    lse = torch.zeros(1, 16, 8, device="cuda", dtype=torch.float32)

    with pytest.raises(NotImplementedError, match=r"up to 256 or exactly \(512, 512\)"):
        _flash_attn_bwd(q, k, v, out, dout, lse, causal=True)


def test_gemma4_sm90_rejects_unvalidated_intermediate_head_dim() -> None:
    _assert_sm90()
    q = torch.randn(8, 16, 384, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(8, 2, 384, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_seqlens = _cu_seqlens([8])

    with pytest.raises((AssertionError, ValueError), match="384"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=8,
            max_seqlen_k=8,
        )


def test_gemma4_sm90_d512_rejects_split_kv() -> None:
    _assert_sm90()
    q = torch.randn(8, 16, 512, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(8, 2, 512, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_seqlens = _cu_seqlens([8])

    with pytest.raises(NotImplementedError, match="SplitKV"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=8,
            max_seqlen_k=8,
            num_splits=2,
        )


def test_gemma4_sm90_d512_rejects_paged_kv() -> None:
    _assert_sm90()
    q = torch.randn(1, 8, 16, 512, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 16, 2, 512, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    page_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")

    with pytest.raises(NotImplementedError, match="paged KV"):
        flash_attn_varlen_func(q, k, v, page_table=page_table)
