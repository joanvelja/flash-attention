# Copyright (c) 2026, FlashAttention contributors.
"""Bounded TMA operands for the symmetric D512 backward cluster.

The descriptor lifecycle follows SM100 backward's per-CTA tensor-map workspace
(upstream #3071). Seven 128-byte slots belong to each *physical* CTA, including
padding in the launch grid. No descriptor storage is added to shared memory.

Slot order is K, V, Kt, Q, dO, Qt, dOt. Normal operands have logical modes
(sequence, D, head[, batch]); transposed operands swap the first two modes.
The source pointer and its strides stay unchanged when bounding a sequence:
packed operands use the absolute endpoint offset + used_length, while dense
operands use used_length. TMA tile coordinates include the packed offset.
Thus a partial tile zero-fills instead of fetching the following sequence.

Initialization may be called by the whole CTA: the manager guards descriptor
copies with the load-warp ID. Only the load warp updates descriptors and issues
loads. The caller must drain prior TMA loads before rewriting a descriptor and
must issue no loads for an empty sequence (the minimum legal extent is one).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils import TensorMapManager, TensorMapUpdateMode


@cute.jit
def _sequence_major(tensor: cute.Tensor):
    """Preserve arbitrary head/sequence strides; only permute logical modes."""
    if const_expr(cute.rank(tensor) == 3):
        order = (0, 2, 1)
    else:
        assert cute.rank(tensor) == 4
        order = (1, 3, 2, 0)
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            tuple(tensor.shape[i] for i in order),
            stride=tuple(tensor.stride[i] for i in order),
        ),
    )


@cute.jit
def _transpose(tensor: cute.Tensor):
    order = (1, 0, 2) if const_expr(cute.rank(tensor) == 3) else (1, 0, 2, 3)
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            tuple(tensor.shape[i] for i in order),
            stride=tuple(tensor.stride[i] for i in order),
        ),
    )


@cute.jit
def make_d512_tma_atoms(Q, K, V, dO, normal_layout, transpose_layout):
    """Create single-CTA 32-KiB loads for the seven operand orientations.

    Layouts describe a single stage, respectively (64, 256) and (256, 64).
    Both cluster CTAs use these atoms; dimension-tile coordinates choose their
    disjoint 256-column halves. The returned original tensors are passed to
    ``update_d512_tensormaps``; returned TMA tensors go to ``load_d512_tile``.
    """
    q, k, v, do = tuple(_sequence_major(t) for t in (Q, K, V, dO))
    descriptor_tensors = (k, v, _transpose(k), q, do, _transpose(q), _transpose(do))
    atoms, tma_tensors = [], []
    for i, tensor in enumerate(descriptor_tensors):
        transposed = const_expr(i in (2, 5, 6))
        layout = transpose_layout if transposed else normal_layout
        tiler = (256, 64) if transposed else (64, 256)
        atom, tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            tensor,
            layout,
            tiler,
        )
        atoms.append(atom)
        tma_tensors.append(tma_tensor)
    return tuple(atoms), tuple(tma_tensors), descriptor_tensors


@cute.jit
def init_d512_tensormaps(atoms, workspace: cute.Tensor, load_warp_id: cutlass.Constexpr):
    """Initialize one descriptor set per physical CTA, with a bounds trap.

    ``workspace`` has shape (product(padded_launch_grid), 7, 16), dtype Int64,
    with contiguous final modes and a 128-byte-aligned base. Call collectively
    from the load warp or the whole CTA. Only the load warp may subsequently
    update/use these descriptors. Return global update and generic TMA pointers.
    """
    assert len(atoms) == 7
    assert workspace.element_type == cutlass.Int64
    manager = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
    # Interface conversion can leave these dimensions dynamic. Validate them
    # on device instead of requiring a Python compile-time truth value.
    if workspace.shape[1] != 7:
        cute.arch.inline_ptx("trap;", read_only_args=[])
    if workspace.shape[2] != 16:
        cute.arch.inline_ptx("trap;", read_only_args=[])
    bx, by, bz = cute.arch.block_idx()
    gx, gy, _ = cute.arch.grid_dim()
    cta_idx = bx + by * gx + bz * gx * gy
    if cta_idx >= cute.size(workspace, mode=[0]):
        cute.arch.inline_ptx("trap;", read_only_args=[])
    gmem_ptrs = tuple(
        manager.get_tensormap_ptr(workspace[(cta_idx, i, None)].iterator) for i in range(7)
    )
    desc_ptrs = tuple(manager.get_tensormap_ptr(p, cute.AddressSpace.generic) for p in gmem_ptrs)
    for atom, ptr in zip(atoms, gmem_ptrs):
        manager.init_tensormap_from_atom(atom, ptr, load_warp_id)
    manager.fence_tensormap_initialization()
    return gmem_ptrs, desc_ptrs


@cute.jit
def update_d512_tensormaps(
    atoms,
    descriptor_tensors,
    gmem_ptrs,
    q_rows_end: Int32,
    k_rows_end: Int32,
    load_warp_id: cutlass.Constexpr,
):
    """Publish sequence-bounded descriptors before any operand prefetch/load.

    The CUTLASS manager emits one release after all seven updates. Each
    descriptor then requires its own 128-byte acquire, matching #3071.
    """
    manager = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
    q_rows_end, k_rows_end = cutlass.max(q_rows_end, 1), cutlass.max(k_rows_end, 1)
    bounded = []
    for i, tensor in enumerate(descriptor_tensors):
        rows_mode = const_expr(1 if i in (2, 5, 6) else 0)
        shape = list(tensor.shape)
        shape[rows_mode] = q_rows_end if const_expr(i >= 3) else k_rows_end
        bounded.append(
            cute.make_tensor(tensor.iterator, cute.make_layout(tuple(shape), stride=tensor.stride))
        )
    manager.update_tensormap(tuple(bounded), atoms, gmem_ptrs, load_warp_id, (None,) * 7)
    for ptr in gmem_ptrs:
        manager.fence_tensormap_update(ptr)
        cute.arch.inline_ptx("prefetch.tensormap [{$r0}];", read_only_args=[ptr.toint()])


@cute.jit
def load_d512_tile(
    atom,
    tma_tensor: cute.Tensor,
    smem: cute.Tensor,
    seq_offset: Int32,
    tile_idx: Int32,
    head_idx: Int32,
    half_idx: Int32,
    batch_idx: Int32,
    barrier,
    desc_ptr,
    transpose: cutlass.Constexpr = False,
):
    """Issue one bounded 64-row by 256-dimension load from the whole load warp.

    ``seq_offset`` is the packed start or zero for dense. ``smem`` is the
    single-stage 2D destination view. The caller owns expect_tx(32768), barrier
    phase management, and producer/consumer lifetime synchronization. Call
    expect_tx from one elected thread, but do not wrap this helper in that
    election: ``cute.copy`` performs its own election and warp collectives.
    """
    if const_expr(cute.rank(tma_tensor) == 3):
        source = tma_tensor[None, None, head_idx]
    else:
        source = tma_tensor[None, None, head_idx, batch_idx]
    if const_expr(transpose):
        source = cute.domain_offset((0, seq_offset), source)
        tile = cute.local_tile(source, (256, 64), (half_idx, tile_idx))
    else:
        source = cute.domain_offset((seq_offset, 0), source)
        tile = cute.local_tile(source, (64, 256), (tile_idx, half_idx))
    dst, src = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0, 2),
        cute.group_modes(tile, 0, 2),
    )
    cute.copy(atom, src, dst, tma_bar_ptr=barrier, tma_desc_ptr=desc_ptr)
