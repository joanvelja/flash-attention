"""FA4's forward and backward as torch.library ops, so torch.compile traces one opaque node.

The ops cover the tensor/scalar surface: dense or packed varlen, causal/local windows, softcap,
learnable sinks, GQA, deterministic backward. Callables (score_mod, mask_mod), block sparsity,
aux tensors, qv/MLA, gathered KV, paged KV, FP8 and scheduler metadata keep the eager
autograd.Function in interface.py. Only ``flash_attn_varlen_func`` routes: a call that uses none
of those goes to the ops, and a dense (batch, seqlen, heads, dim) layout gets them through
``flash_attn_varlen_func`` with cu_seqlens_q=cu_seqlens_k=None; ``flash_attn_func`` is
interface's, unchanged.

Under FakeTensorMode the ops run only their fake impls and compile no CuTe kernel, so a
compile-only warm-up (FLASH_ATTENTION_FAKE_TENSOR=1) must call ``interface`` directly.
"""

import inspect
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from flash_attn.cute import interface


# The ops' outputs are allocated here, once for both impls: the real impl hands them to
# interface (out=, lse=; dq=, dk=, dv=), which checks their shapes and writes into them, and
# the fake returns them. So the strides inductor plans with are the real ones by construction,
# whatever the inputs' strides (left to itself, _flash_attn_bwd allocates
# empty_like(maybe_contiguous(x)), which keeps an input's strides when its last one is 1).
def _fwd_outputs(q: Tensor, v: Tensor, cu_seqlens_q: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    """out and lse, contiguous, in _flash_attn_fwd's shapes (varlen iff cu_seqlens_q, as there)."""
    num_heads, head_dim_v = q.shape[-2], v.shape[-1]
    if cu_seqlens_q is None:  # dense (batch, seqlen, heads, head_dim)
        batch, seqlen_q = q.shape[:2]
        return (
            q.new_empty((batch, seqlen_q, num_heads, head_dim_v)),
            q.new_empty((batch, num_heads, seqlen_q), dtype=torch.float32),
        )
    total_q = q.shape[0]  # packed varlen (total_q, heads, head_dim)
    return (
        q.new_empty((total_q, num_heads, head_dim_v)),
        q.new_empty((num_heads, total_q), dtype=torch.float32),
    )


def _bwd_outputs(q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """dq, dk, dv, contiguous. dsink is _flash_attn_bwd's empty_like of the 1-D sink, as the
    fake's: a 1-D empty_like is contiguous."""
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)


@torch.library.custom_op("flash_attn_cute::fwd", mutates_args=(), device_types="cuda")
def _fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Optional[Tensor],
    cu_seqlens_k: Optional[Tensor],
    seqused_q: Optional[Tensor],
    seqused_k: Optional[Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    softcap: float,
    learnable_sink: Optional[Tensor],
    num_splits: int,
    pack_gqa: Optional[bool],
    deterministic: bool,
    return_lse: bool,
) -> Tuple[Tensor, Tensor]:
    # deterministic and return_lse steer only the backward; setup_context reads them.
    # lse always: a compiled forward sees no input that requires grad, and _flash_attn_fwd
    # would otherwise return lse=None, which the schema cannot carry.
    out, lse = _fwd_outputs(q, v, cu_seqlens_q)
    interface._flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        softcap=softcap,
        learnable_sink=learnable_sink,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        return_lse=True,
        out=out,
        lse=lse,
    )
    return out, lse


@_fwd.register_fake
def _fwd_fake(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    learnable_sink,
    num_splits,
    pack_gqa,
    deterministic,
    return_lse,
):
    return _fwd_outputs(q, v, cu_seqlens_q)


@torch.library.custom_op("flash_attn_cute::bwd", mutates_args=(), device_types="cuda")
def _bwd(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    dlse: Optional[Tensor],
    cu_seqlens_q: Optional[Tensor],
    cu_seqlens_k: Optional[Tensor],
    seqused_q: Optional[Tensor],
    seqused_k: Optional[Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    softcap: float,
    learnable_sink: Optional[Tensor],
    deterministic: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    dq, dk, dv = _bwd_outputs(q, k, v)
    grads = interface._flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        softmax_scale,
        causal,
        softcap,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        deterministic=deterministic,
        dq=dq,
        dk=dk,
        dv=dv,
        dlse=dlse,
        learnable_sink=learnable_sink,
    )
    # The schema has no optional outputs: an empty tensor stands for "no sink".
    dsink = grads[3] if learnable_sink is not None else q.new_empty(0)
    return dq, dk, dv, dsink


@_bwd.register_fake
def _bwd_fake(
    dout,
    q,
    k,
    v,
    out,
    lse,
    dlse,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    learnable_sink,
    deterministic,
):
    dsink = torch.empty_like(learnable_sink) if learnable_sink is not None else q.new_empty(0)
    return (*_bwd_outputs(q, k, v), dsink)


def _setup_context(ctx, inputs, output):
    (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        learnable_sink,
        _num_splits,
        _pack_gqa,
        deterministic,
        return_lse,
    ) = inputs
    out, lse = output
    ctx.save_for_backward(
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink
    )
    ctx.scalars = (
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
    )
    ctx.return_lse = return_lse
    ctx.set_materialize_grads(False)


def _backward(ctx, dout, dlse):
    q, k, v, out, lse, cu_q, cu_k, used_q, used_k, sink = ctx.saved_tensors
    max_q, max_k, scale, causal, left, right, softcap, deterministic = ctx.scalars
    # An lse dropped inside a compiled graph, or in eager, arrives here as None. One that leaves
    # a compiled region as an output and is dropped outside it arrives as a materialized zero
    # tensor. flash_attn_varlen_func returns lse=None when return_lse is False, so only a direct
    # torch.ops.flash_attn_cute.fwd caller can do that; the gate keeps such a dlse out of the
    # kernel, where the SM90 D512 backward would raise NotImplementedError on it.
    if not ctx.return_lse:
        dlse = None
    if dout is None:
        dout = torch.zeros_like(out)
    dq, dk, dv, dsink = _bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dlse,
        cu_q,
        cu_k,
        used_q,
        used_k,
        max_q,
        max_k,
        scale,
        causal,
        left,
        right,
        softcap,
        sink,
        deterministic,
    )
    # Inputs: q k v | cu_q cu_k used_q used_k max_q max_k scale causal left right softcap | sink | 4 flags
    return (dq, dk, dv) + (None,) * 11 + (dsink if sink is not None else None,) + (None,) * 4


_fwd.register_autograd(_backward, setup_context=_setup_context)


# Arguments the ops do not take: a call that sets any of them keeps interface's eager path.
_EAGER_ONLY_NONE = (
    "qv",
    "gather_kv_indices",
    "page_table",
    "min_seqlen_k",
    "score_mod",
    "score_mod_bwd",
    "mask_mod",
    "block_sparse_tensors",
    "scheduler_metadata",
    "seqlen_k_per_split",
)


def _op_representable(call: dict) -> bool:
    q, k, v = call["q"], call["k"], call["v"]
    return (
        q is not None
        and q.dtype in (torch.float16, torch.bfloat16)
        and not (k is v and v.shape[-1] == 512)  # interface's MLA specialization
        and all(call[name] is None for name in _EAGER_ONLY_NONE)
        and not call["aux_tensors"]
        and not call["aux_scalars"]
        and not call["disable_scheduler_metadata"]
        and not torch.is_tensor(call["max_seqlen_q"])
        and not torch.is_tensor(call["max_seqlen_k"])
    )


def flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    qv: Optional[Tensor] = None,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    seqused_q: Optional[Tensor] = None,
    seqused_k: Optional[Tensor] = None,
    gather_kv_indices: Optional[Tensor] = None,
    page_table: Optional[Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors=None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    return_lse: bool = False,
    scheduler_metadata=None,
    seqlen_k_per_split: Optional[int] = None,
    disable_scheduler_metadata: bool = False,
):
    """interface.flash_attn_varlen_func, through the ops when the call allows.

    Returns (out, lse), with lse=None unless return_lse: a caller that uses an lse it did not
    ask for crashes instead of training on a gradient the backward drops.
    """
    call = dict(locals())  # first statement: exactly the parameters, which match interface's
    if not _op_representable(call):
        out, lse = interface.flash_attn_varlen_func(**call)
        return out, (lse if return_lse else None)
    out, lse = _fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        learnable_sink,
        num_splits,
        pack_gqa,
        deterministic,
        return_lse,
    )
    return out, (lse if return_lse else None)


def _parameter_rows(fn: Callable) -> list:
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


# The wrapper forwards **call to interface and is exported in its place: a parameter interface
# gains, drops, reorders or re-defaults must be mirrored above, or the import fails here.
if _parameter_rows(flash_attn_varlen_func) != _parameter_rows(interface.flash_attn_varlen_func):
    raise RuntimeError(
        "flash_attn.cute.torch_ops.flash_attn_varlen_func no longer mirrors "
        "interface.flash_attn_varlen_func:\n"
        f"  torch_ops: {_parameter_rows(flash_attn_varlen_func)}\n"
        f"  interface: {_parameter_rows(interface.flash_attn_varlen_func)}"
    )
