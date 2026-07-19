import cutlass
import cutlass.cute as cute
import pytest
import torch
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func, utils
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
from flash_attn.cute.interface import _flash_attn_fwd

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
    q_float = q.float().transpose(0, 1)
    k_float = k.float().repeat_interleave(repeats, dim=1).transpose(0, 1)
    v_float = v.float().repeat_interleave(repeats, dim=1).transpose(0, 1)
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
    output = torch.matmul(scores.softmax(-1), v_float).transpose(0, 1)
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
        (512, [257, 129], [513, 385], True, 1.0, (None, None), False),
        (512, [200, 129], [264, 385], False, 1.0, (127, 15), False),
        (512, [0, 257, 129], [64, 513, 385], True, 1.0, (None, None), False),
    ],
    ids=(
        "legacy-d64",
        "legacy-d128-bottom-right",
        "gemma-local-d256",
        "gemma-global-d512-forward",
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
    output, lse = _flash_attn_fwd(
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
    _, sparse_tensors = compute_block_sparsity(
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
    output, lse = _flash_attn_fwd(
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


def test_gemma4_sm90_d512_backward_fails_before_compilation() -> None:
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
    )

    with pytest.raises(NotImplementedError, match="D512 backward"):
        output.sum().backward()


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
