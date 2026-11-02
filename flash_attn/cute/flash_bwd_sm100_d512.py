# Copyright (c) 2026, FlashAttention contributors.
"""Symmetric two-CTA D512 backward, functional pipeline (Stage A).

A cluster owns one 64-key tile. Each independent cta_group::1 CTA owns 256
columns. It exchanges fp32 partial S/dP before conversion, and retains its
fp32 dK/dV accumulators through every query tile and GQA head. No global P/dS
or dK/dV partial workspace is used. This is the deliberately serialized first
stage of the approved pipeline; the second S/dP TMEM bank is reserved.

Reuse: blackwell_helpers' Layout E weight-stationary UMMA; MLA's column-split
cluster ownership; SM100 backward's per-CTA bounded tensor-map lifecycle and
cp.reduce.async.bulk.add.f32 dQ drain. See flash_bwd_sm100_d512_tma.py.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import tcgen05
from cutlass._mlir.dialects import llvm

from flash_attn.cute import barrier, copy_utils, utils
from flash_attn.cute.blackwell_helpers import gemm_ws_ptx_partial, i64_to_i32x2
import flash_attn.cute.mma_sm100_desc as sm100_desc
from flash_attn.cute.flash_bwd_sm100_d512_tma import (
    init_d512_tensormaps,
    load_d512_tile,
    make_d512_tma_atoms,
    update_d512_tensormaps,
)


@cute.struct
class _SharedStorage:
    load_barrier: Int64
    mma_barrier: Int64
    exchange_barrier: Int64
    tmem_handle: Int32
    owner_id: Int32


class FlashAttentionBackwardSm100D512:
    """Causal long-prefill D512 backward; no unsupported-mode fallback."""

    def __init__(
        self,
        head_dim=512,
        head_dim_v=512,
        qhead_per_kvhead=1,
        deterministic=False,
        is_causal=True,
        window_size_left=None,
        window_size_right=None,
        num_sms=148,
    ):
        assert head_dim == head_dim_v == 512, "D512 backward requires Dq=Dv=512"
        assert qhead_per_kvhead >= 1, "D512 backward requires integral positive GQA"
        self.deterministic = deterministic
        assert is_causal, "D512 backward v2 currently supports causal long-prefill only"
        assert window_size_left is None or window_size_left < 0, (
            "D512 backward v2 sliding windows are not implemented yet"
        )
        assert window_size_right is None or window_size_right <= 0
        assert num_sms >= 2
        self.qhead_per_kvhead = qhead_per_kvhead
        self.num_clusters = num_sms // 2
        self.load_warp = 4
        self.mma_warp = 5
        self.num_threads = 256

    @staticmethod
    def _acc_layout(n):
        # Layout E: 64 logical rows packed into 128 lanes, N/2 columns.
        return cute.make_layout((64, (n // 2, 2)), stride=(65536, (1, 64 * 65536)))

    @staticmethod
    def _acc_coords(n):
        return cute.composition(
            cute.make_identity_tensor((64, n)),
            cute.make_layout((64, (n // 2, 2)), stride=(1, (64, 64 * (n // 2)))),
        )

    @cute.jit
    def _logical(self, tensor, rows: cutlass.Constexpr, cols: cutlass.Constexpr):
        return cute.composition(tensor, cute.make_ordered_layout((rows, cols), order=(0, 1)))

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dO: cute.Tensor,
        lse_log2: cute.Tensor,
        dpsum: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        scale_softmax: Float32,
        cu_seqlens_q: cute.Tensor | None,
        cu_seqlens_k: cute.Tensor | None,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        seqused_q: cute.Tensor | None = None,
        seqused_k: cute.Tensor | None = None,
        tensormap_workspace: cute.Tensor | None = None,
        owner_counter: cute.Tensor | None = None,
        dq_semaphores: cute.Tensor | None = None,
        key_tile_prefix: cute.Tensor | None = None,
        owner_limit: Int32 = 0,
        stream: cuda.CUstream = None,
    ):
        self.dtype = Q.element_type
        assert self.dtype in (cutlass.BFloat16, cutlass.Float16)
        assert all(t.element_type == self.dtype for t in (K, V, dO, dK, dV))
        assert dQ.element_type == Float32, "D512 v2 needs zero-initialized fp32 dQaccum"
        assert tensormap_workspace is not None, "D512 v2 needs per-CTA tensor-map workspace"
        assert cute.rank(dQ) == (4 if cu_seqlens_q is not None else 5)
        assert owner_counter is not None, "D512 backward needs a zero-initialized owner counter"
        assert owner_counter.element_type == Int32 and cute.rank(owner_counter) == 1
        assert (key_tile_prefix is not None) == (cu_seqlens_k is not None or seqused_k is not None)
        if const_expr(key_tile_prefix is not None):
            assert key_tile_prefix.element_type == Int32 and cute.rank(key_tile_prefix) == 1
        if const_expr(self.deterministic):
            assert dq_semaphores is not None and dq_semaphores.element_type == Int32
            assert cute.rank(dq_semaphores) == (3 if cu_seqlens_q is not None else 4)
        mma_score = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (64, 64),
            tcgen05.OperandSource.SMEM,
        )
        mma_grad = sm100_utils.make_trivial_tiled_mma(
            self.dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            (64, 256),
            tcgen05.OperandSource.SMEM,
        )
        normal = sm100_utils.make_smem_layout_a(mma_score, (64, 64, 256), self.dtype, 1)
        transpose = sm100_utils.make_smem_layout_b(mma_grad, (64, 256, 64), self.dtype, 1)
        prob = sm100_utils.make_smem_layout_a(mma_grad, (64, 256, 64), self.dtype, 1)
        assert cute.cosize(normal) * 2 == 32768
        assert cute.cosize(transpose) * 2 == 32768
        assert cute.cosize(prob) * 2 == 8192
        # The 32-byte header rounds to the first operand's 1024-byte alignment.
        # Every following allocation size is a multiple of its alignment.
        smem_bytes = (
            (4 * cute.cosize(normal) + cute.cosize(transpose) + 3 * cute.cosize(prob)) * 2
            + 32768
            + 1024
        )
        assert smem_bytes <= 232448, f"D512 backward SMEM budget exceeded: {smem_bytes}"
        normal_2d = cute.composition(normal, cute.make_ordered_layout((64, 256), order=(0, 1)))
        transpose_2d = cute.composition(
            transpose, cute.make_ordered_layout((256, 64), order=(0, 1))
        )
        atoms, tma_tensors, descriptor_tensors = make_d512_tma_atoms(
            Q,
            K,
            V,
            dO,
            normal_2d,
            transpose_2d,
        )
        batch = (
            cu_seqlens_q.shape[0] - 1
            if cu_seqlens_q is not None
            else cu_seqlens_k.shape[0] - 1
            if cu_seqlens_k is not None
            else Q.shape[0]
        )
        self.kernel(
            Q,
            K,
            dQ,
            dK,
            dV,
            lse_log2,
            dpsum,
            scale_softmax,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_k,
            batch,
            atoms,
            tma_tensors,
            descriptor_tensors,
            tensormap_workspace,
            owner_counter,
            dq_semaphores,
            key_tile_prefix,
            owner_limit,
            mma_score,
            mma_grad,
            normal,
            transpose,
            prob,
        ).launch(
            grid=(self.num_clusters * 2, 1, 1),
            block=(self.num_threads, 1, 1),
            cluster=(2, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _stat(self, src, batch, head, row):
        if const_expr(cute.rank(src) == 2):
            value = src[head, row]
        else:
            value = src[batch, head, row]
        return value

    @cute.jit
    def _after_sync(self):
        llvm.inline_asm(
            None,
            [],
            "tcgen05.fence::after_thread_sync;",
            "",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @cute.jit
    def _mma(self, mma, a, b, address, zero: cutlass.Constexpr = False):
        if cute.arch.warp_idx() == self.mma_warp:
            ar = mma.make_fragment_A(a)
            br = mma.make_fragment_B(b)
            gemm_ws_ptx_partial(
                mma.op,
                address,
                ar[None, None, None, 0],
                br[None, None, None, 0],
                a[None, None, None, 0],
                b[None, None, None, 0],
                zero_init=zero,
            )

    @cute.jit
    def _mma_score_rolled(self, mma, a, b, address, zero: cutlass.Constexpr = False):
        # D512-local equivalent of gemm_ws_ptx_partial for score K=256 only.
        # The real PTX loop prevents retaining all 16 invariant descriptors
        # across the query loop. Gradient issue remains on the upstream helper.
        if cute.arch.warp_idx() == self.mma_warp:
            op = mma.op
            assert isinstance(op, tcgen05.mma.MmaF16BF16Op)
            assert op.a_src == tcgen05.OperandSource.SMEM
            assert op.a_major_mode == op.b_major_mode == tcgen05.OperandMajorMode.K
            assert op.a_dtype.width == op.b_dtype.width == 16
            ar, br = mma.make_fragment_A(a), mma.make_fragment_B(b)
            ar, br = ar[None, None, None, 0], br[None, None, None, 0]
            sa, sb = a[None, None, None, 0], b[None, None, None, 0]
            assert cute.size(ar.shape[2]) == cute.size(br.shape[2]) == 16
            offsets_a = tuple(cute.crd2idx((0, 0, k), ar.layout) for k in range(16))
            offsets_b = tuple(cute.crd2idx((0, 0, k), br.layout) for k in range(16))
            expected = tuple(2 * (k & 3) + 0x200 * (k >> 2) for k in range(16))
            # These are the exact offset units used by gemm_ptx_partial, not
            # guessed byte offsets. Every actual A/B layout entry must match.
            assert offsets_a == expected, "D512 score A descriptor walk differs from layout"
            assert offsets_b == expected, "D512 score B descriptor walk differs from layout"
            idesc = const_expr(sm100_desc.mma_op_to_idesc(op))
            base_a = const_expr(
                sm100_desc.make_smem_desc_base(
                    cute.recast_layout(128, op.a_dtype.width, sa.layout[0]),
                    sa.iterator.type.swizzle_type,
                    sm100_desc.Major.K,
                )
            )
            base_b = const_expr(
                sm100_desc.make_smem_desc_base(
                    cute.recast_layout(128, op.b_dtype.width, sb.layout[0]),
                    sb.iterator.type.swizzle_type,
                    sm100_desc.Major.K,
                )
            )
            base_a_lo, base_a_hi = i64_to_i32x2(base_a)
            base_b_lo, base_b_hi = i64_to_i32x2(base_b)
            start_a = Int32(
                base_a_lo | sm100_desc.make_smem_desc_start_addr(sa[None, None, 0].iterator)
            )
            start_b = Int32(
                base_b_lo | sm100_desc.make_smem_desc_start_addr(sb[None, None, 0].iterator)
            )
            # Whole warp participates in elect.sync. Only its elected lane
            # issues MMA; all lanes execute the same 16 ordered K iterations.
            llvm.inline_asm(
                None,
                [
                    Int32(cute.arch.make_warp_uniform(start_a)).ir_value(),
                    Int32(cute.arch.make_warp_uniform(start_b)).ir_value(),
                    Int32(not zero).ir_value(),
                    Int32(cute.arch.make_warp_uniform(address)).ir_value(),
                ],
                "{\n\t"
                ".reg .pred leader, initialize_accum, accumulate, more;\n\t"
                ".reg .b32 idesc, tmem_acc, a_start, b_start, a_lo, b_lo, a_hi, b_hi;\n\t"
                ".reg .b32 k, low_step, high_step, offset;\n\t"
                ".reg .b64 a_desc, b_desc;\n\t"
                "elect.sync _|leader, -1;\n\t"
                f"mov.b32 idesc, {hex(idesc)};\n\t"
                "mov.b32 tmem_acc, $3;\n\t"
                "mov.b32 a_start, $0;\n\t"
                "mov.b32 b_start, $1;\n\t"
                f"mov.b32 a_hi, {hex(base_a_hi)};\n\t"
                f"mov.b32 b_hi, {hex(base_b_hi)};\n\t"
                "setp.ne.b32 initialize_accum, $2, 0;\n\t"
                "mov.u32 k, 0;\n\t"
                "SCORE_K_LOOP:\n\t"
                '.pragma "nounroll";\n\t'
                "and.b32 low_step, k, 3;\n\t"
                "shl.b32 low_step, low_step, 1;\n\t"
                "shr.u32 high_step, k, 2;\n\t"
                "shl.b32 high_step, high_step, 9;\n\t"
                "add.u32 offset, low_step, high_step;\n\t"
                "add.u32 a_lo, a_start, offset;\n\t"
                "add.u32 b_lo, b_start, offset;\n\t"
                "mov.b64 a_desc, {a_lo, a_hi};\n\t"
                "mov.b64 b_desc, {b_lo, b_hi};\n\t"
                "setp.ne.u32 accumulate, k, 0;\n\t"
                "or.pred accumulate, accumulate, initialize_accum;\n\t"
                "@leader tcgen05.mma.ws.cta_group::1.kind::f16 "
                "[tmem_acc], a_desc, b_desc, idesc, accumulate;\n\t"
                "add.u32 k, k, 1;\n\t"
                "setp.lt.u32 more, k, 16;\n\t"
                "@more bra SCORE_K_LOOP;\n\t"
                "}\n",
                "r,r,r,r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )

    @cute.jit
    def _mma_wait(self, barrier, phase):
        if cute.arch.warp_idx() == self.mma_warp:
            with cute.arch.elect_one():
                tcgen05.commit(barrier)
        cute.arch.mbarrier_wait(barrier, phase)
        cute.arch.sync_threads()
        self._after_sync()

    @cute.jit
    def _claim_owner(self, storage, counter, half, tid):
        # Called by the whole initialized resident pair, never a role branch.
        local = cute.make_tensor(storage.owner_id.ptr, cute.make_layout(1))
        remote = cute.arch.map_dsmem_ptr(storage.owner_id.ptr, 0)
        if half == 0 and tid == 0:
            local[0] = utils.atomic_add_i32(1, counter.iterator)
        # Non-relaxed release/acquire publishes CTA0's ordinary SMEM store.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if half == 1 and tid == 0:
            # CuTe tensor indexing does not support DSMEM scalar loads. Keep
            # the mapped shared::cluster address and issue the synchronous PTX
            # load after the release/acquire cluster publication above.
            local[0] = Int32(
                llvm.inline_asm(
                    Int32.mlir_type,
                    [Int32(remote.toint()).ir_value()],
                    "ld.shared::cluster.b32 $0, [$1];",
                    "=r,r,~{memory}",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            )
        cute.arch.sync_threads()
        work = local[0]
        if work < 0:
            cute.arch.inline_ptx("trap;", read_only_args=[])
        return work

    @cute.jit
    def _wait_owner(self, semaphore, tile):
        # Only the reduction issuer calls this. No timeout: a predecessor may
        # legitimately be slow. A value past our turn is a protocol violation.
        observed = barrier.ld_acquire(semaphore)
        while observed != tile:
            if observed > tile:
                cute.arch.inline_ptx("trap;", read_only_args=[])
            observed = barrier.ld_acquire(semaphore)

    @cute.jit
    def _drain_sync(self):
        # Only warps 0..3 and 6 participate. PTX .aligned would require a
        # CTA-uniform condition; the explicit unaligned form counts 160.
        llvm.inline_asm(
            None,
            [],
            "barrier.cta.sync 2, 160;",
            "~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @cute.jit
    def _load(
        self,
        slot: cutlass.Constexpr,
        atoms,
        tensors,
        pointers,
        dst,
        offset,
        tile,
        head,
        half,
        batch,
        barrier,
        transpose: cutlass.Constexpr = False,
    ):
        load_d512_tile(
            atoms[slot],
            tensors[slot],
            dst,
            offset,
            tile,
            head,
            half,
            batch,
            barrier,
            pointers[slot],
            transpose,
        )

    @cute.jit
    def _load_query_pair(
        self, atoms, tensors, dptrs, sA, sB, qstart, qt, qhead, half, batch, barrier
    ):
        if cute.arch.warp_idx() == self.load_warp:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, 65536)
            self._load(
                3,
                atoms,
                tensors,
                dptrs,
                self._logical(sA, 64, 256),
                qstart,
                qt,
                qhead,
                half,
                batch,
                barrier,
            )
            self._load(
                4,
                atoms,
                tensors,
                dptrs,
                self._logical(sB, 64, 256),
                qstart,
                qt,
                qhead,
                half,
                batch,
                barrier,
            )

    @cute.jit
    def _epilogue(
        self,
        source,
        destination,
        stage,
        batch,
        start,
        keytile,
        head,
        half,
        nk,
        tid,
        ld_grad,
        grad_coords,
    ):
        # TMEM -> registers -> swizzled SMEM, then coalesced vector global stores.
        if tid < 128:
            r = cute.make_rmem_tensor(grad_coords.shape, Float32)
            cute.copy(ld_grad, ld_grad.partition_S(source), r)
            cute.arch.fence_view_async_tmem_load()
            for i in cutlass.range_constexpr(cute.size(r)):
                row, col = grad_coords[i]
                stage[row, col] = self.dtype(r[i])
        cute.arch.sync_threads()
        if tid < 128:
            row = tid // 2
            col = (tid % 2) * 128
            if keytile * 64 + row < nk:
                src = cute.local_tile(stage[row, None], (128,), (tid % 2,))
                if const_expr(cute.rank(destination) == 3):
                    dst = destination[Int64(start) + keytile * 64 + row, head, None]
                else:
                    dst = destination[batch, Int64(start) + keytile * 64 + row, head, None]
                dst = cute.local_tile(dst, (128,), (half * 2 + tid % 2,))
                temp = cute.make_rmem_tensor((128,), self.dtype)
                cute.autovec_copy(src, temp)
                cute.autovec_copy(temp, dst)
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        dQ: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        LSE: cute.Tensor,
        D: cute.Tensor,
        scale: Float32,
        cuQ: cute.Tensor | None,
        cuK: cute.Tensor | None,
        usedQ: cute.Tensor | None,
        usedK: cute.Tensor | None,
        max_k: Int32,
        batch_count: Int32,
        atoms,
        tensors,
        originals,
        workspace: cute.Tensor,
        counter: cute.Tensor,
        semaphores: cute.Tensor | None,
        key_prefix: cute.Tensor | None,
        owner_limit: Int32,
        mma_score: cute.TiledMma,
        mma_grad: cute.TiledMma,
        normal: cute.ComposedLayout,
        transpose: cute.ComposedLayout,
        prob: cute.ComposedLayout,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bx = cute.arch.block_idx()[0]
        half = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        peer = half ^ 1
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(_SharedStorage)
        sK = smem.allocate_tensor(
            self.dtype, normal.outer, byte_alignment=1024, swizzle=normal.inner
        )
        sV = smem.allocate_tensor(
            self.dtype, normal.outer, byte_alignment=1024, swizzle=normal.inner
        )
        sKt = smem.allocate_tensor(
            self.dtype, transpose.outer, byte_alignment=1024, swizzle=transpose.inner
        )
        sA = smem.allocate_tensor(
            self.dtype, normal.outer, byte_alignment=1024, swizzle=normal.inner
        )
        sB = smem.allocate_tensor(
            self.dtype, normal.outer, byte_alignment=1024, swizzle=normal.inner
        )
        incoming = smem.allocate_tensor(
            Float32, cute.make_layout((2, 64, 64), stride=(4096, 1, 64)), byte_alignment=128
        )
        sP = smem.allocate_tensor(self.dtype, prob.outer, byte_alignment=1024, swizzle=prob.inner)
        sDS = smem.allocate_tensor(self.dtype, prob.outer, byte_alignment=1024, swizzle=prob.inner)
        sDST = smem.allocate_tensor(self.dtype, prob.outer, byte_alignment=1024, swizzle=prob.inner)
        # 216 KiB payload + alignment/header <= 218 KiB, below 227 KiB.
        sAt = cute.make_tensor(
            cute.recast_ptr(sA.iterator, swizzle_=transpose.inner), transpose.outer
        )
        sBt = cute.make_tensor(
            cute.recast_ptr(sB.iterator, swizzle_=transpose.inner), transpose.outer
        )
        outgoing = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=Float32, swizzle_=None), incoming.layout
        )
        # All three 8KiB probability views are dead after the gradient wait.
        # Reuse their contiguous 24KiB range, preserving the global dQ layout.
        reduce_stage = cute.make_tensor(
            cute.recast_ptr(sP.iterator, dtype=Float32, swizzle_=None),
            cute.make_layout(6144),
        )
        p_view, ds_view, dst_view = (self._logical(t, 64, 64) for t in (sP, sDS, sDST))
        if tid == 0:
            cute.arch.mbarrier_init(storage.load_barrier.ptr, 1)
            cute.arch.mbarrier_init(storage.mma_barrier.ptr, 1)
            cute.arch.mbarrier_init(storage.exchange_barrier.ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        gptrs, dptrs = init_d512_tensormaps(atoms, workspace, self.load_warp)
        allocator = cutlass.utils.TmemAllocator(
            storage.tmem_handle.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(1, self.num_threads),
            allocator_warp_id=self.mma_warp,
        )
        allocator.allocate(512)
        allocator.wait_for_alloc()
        ptr = allocator.retrieve_ptr(Float32)
        allocator.relinquish_alloc_permit()
        tDK = cute.make_tensor(ptr, self._acc_layout(256))
        tDV = cute.make_tensor(ptr + 128, self._acc_layout(256))
        tDQ = cute.make_tensor(ptr + 256, self._acc_layout(256))
        # Copy-layout prototype; each loop selects one complete 64-column pair.
        tScore = cute.make_tensor(ptr + 384, self._acc_layout(64))
        # S/dP banks: 384/416 and 448/480, both within the 512-column allocation.
        ld_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        st_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        ld = tcgen05.make_tmem_copy(ld_atom, tScore).get_slice(tid % 128)
        coords = ld.partition_D(self._acc_coords(64))
        ld_grad = tcgen05.make_tmem_copy(ld_atom, tDK).get_slice(tid % 128)
        st_grad = tcgen05.make_tmem_copy(st_atom, tDK).get_slice(tid % 128)
        grad_coords = ld_grad.partition_D(self._acc_coords(256))
        # The register view below assumes one row and consecutive columns.
        # Check the actual copy partition, not just the intended MMA layout.
        assert cute.size(grad_coords) == 128
        assert cute.coalesce(grad_coords.layout).stride == cute.E(1), (
            f"D512 dQ register columns are not consecutive: {grad_coords.layout}"
        )
        dq_store = copy_utils.tiled_copy_1d(Float32, 64, num_copy_elems=4).get_slice(tid % 64)
        dq_store_dst96 = dq_store.partition_D(reduce_stage)
        dq_store_dst32 = dq_store.partition_D(
            cute.make_tensor(reduce_stage.iterator, cute.make_layout(2048))
        )
        dq_store_dst64 = dq_store.partition_D(
            cute.make_tensor(reduce_stage.iterator, cute.make_layout(4096))
        )
        dq_store_dst64_hi = dq_store.partition_D(
            cute.make_tensor(reduce_stage.iterator + 2048, cute.make_layout(4096))
        )
        assert cute.size(dq_store_dst96) == 96
        assert cute.size(dq_store_dst32) == 32
        assert cute.size(dq_store_dst64) == cute.size(dq_store_dst64_hi) == 64
        load_phase, mma_phase, exchange_phase = Int32(0), Int32(0), Int32(0)
        if const_expr(self.deterministic):
            if (
                semaphores.shape[-2] != 2
                or semaphores.shape[-3] != Q.shape[-2]
                or Int64(semaphores.shape[-1]) * 64 != Int64(dQ.shape[-2])
            ):
                cute.arch.inline_ptx("trap;", read_only_args=[])
            if const_expr(cuQ is None):
                if semaphores.shape[0] != batch_count:
                    cute.arch.inline_ptx("trap;", read_only_args=[])
        if counter.shape[0] != 1 or owner_limit < 0:
            cute.arch.inline_ptx("trap;", read_only_args=[])
        if const_expr(key_prefix is not None):
            if key_prefix.shape[0] != Int64(batch_count) + 1:
                cute.arch.inline_ptx("trap;", read_only_args=[])
            key_tiles = Int64(key_prefix[batch_count])
            if key_prefix[0] != 0 or key_tiles < 0:
                cute.arch.inline_ptx("trap;", read_only_args=[])
            total_work = Int64(K.shape[-2]) * key_tiles
        else:
            key_tiles = cute.ceil_div(Int64(K.shape[1]), 64)
            total_work = Int64(batch_count) * Int64(K.shape[-2]) * key_tiles
        if total_work > Int64(owner_limit):
            cute.arch.inline_ptx("trap;", read_only_args=[])
        # Every owner, including the first, is claimed only after both CTAs
        # finish initialization. There are no reserved ranges or lookahead.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        work = self._claim_owner(storage, counter, half, tid)
        # Resident pairs claim one monotonic owner at a time. Ascending actual
        # key tiles retain longest-first causal work within each KV group.
        while work < total_work:
            if const_expr(key_prefix is not None):
                batch = utils.get_batch_from_cu_tensor(
                    Int32(Int64(work) // Int64(K.shape[-2])), key_prefix
                )
                if batch < 0 or batch >= batch_count:
                    cute.arch.inline_ptx("trap;", read_only_args=[])
                start = Int64(key_prefix[batch])
                end = Int64(key_prefix[batch + 1])
                tiles = end - start
                if start < 0 or tiles <= 0 or end > key_tiles:
                    cute.arch.inline_ptx("trap;", read_only_args=[])
                local_work = Int64(work) - Int64(K.shape[-2]) * start
            else:
                tiles = key_tiles
                batch = Int32(Int64(work) // (Int64(K.shape[-2]) * tiles))
                local_work = Int64(work) % (Int64(K.shape[-2]) * tiles)
            kv_head64 = local_work // tiles
            tile64 = local_work % tiles
            if (
                local_work < 0
                or kv_head64 < 0
                or kv_head64 >= Int64(K.shape[-2])
                or tile64 < 0
                or tile64 >= tiles
                or batch < 0
                or batch >= batch_count
            ):
                cute.arch.inline_ptx("trap;", read_only_args=[])
            kv_head, tile = Int32(kv_head64), Int32(tile64)
            qstart, kstart = Int32(0), Int32(0)
            if const_expr(cuQ is not None):
                qstart = cuQ[batch]
                nq = cuQ[batch + 1] - qstart
            else:
                nq = Q.shape[1]
            if const_expr(cuK is not None):
                kstart = cuK[batch]
                nk = cuK[batch + 1] - kstart
            else:
                nk = K.shape[1]
            if const_expr(usedQ is not None):
                nq = usedQ[batch]
            if const_expr(usedK is not None):
                nk = usedK[batch]
            if Int64(tile) * 64 >= Int64(nk) or tiles != cute.ceil_div(Int64(nk), 64):
                cute.arch.inline_ptx("trap;", read_only_args=[])
            if tile * 64 < nk:
                if tid < 128:
                    zero = cute.make_rmem_tensor(grad_coords.shape, Float32)
                    zero.fill(0.0)
                    cute.copy(st_grad, zero, st_grad.partition_D(tDK))
                    cute.copy(st_grad, zero, st_grad.partition_D(tDV))
                    cute.arch.fence_view_async_tmem_store()
                cute.arch.sync_threads()
                self._after_sync()
                if warp == self.load_warp:
                    update_d512_tensormaps(
                        atoms, originals, gptrs, qstart + nq, kstart + nk, self.load_warp
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(storage.load_barrier.ptr, 98304)
                    self._load(
                        0,
                        atoms,
                        tensors,
                        dptrs,
                        self._logical(sK, 64, 256),
                        kstart,
                        tile,
                        kv_head,
                        half,
                        batch,
                        storage.load_barrier.ptr,
                    )
                    self._load(
                        1,
                        atoms,
                        tensors,
                        dptrs,
                        self._logical(sV, 64, 256),
                        kstart,
                        tile,
                        kv_head,
                        half,
                        batch,
                        storage.load_barrier.ptr,
                    )
                    self._load(
                        2,
                        atoms,
                        tensors,
                        dptrs,
                        self._logical(sKt, 256, 64),
                        kstart,
                        tile,
                        kv_head,
                        half,
                        batch,
                        storage.load_barrier.ptr,
                        True,
                    )
                cute.arch.mbarrier_wait(storage.load_barrier.ptr, load_phase)
                load_phase ^= 1
                cute.arch.sync_threads()
                self._after_sync()
                first_q_tile = cutlass.max(0, (tile * 64 + nq - nk) // 64)
                for h in cutlass.range(self.qhead_per_kvhead):
                    qhead = kv_head * self.qhead_per_kvhead + h
                    score_bank = Int32(0)
                    if first_q_tile < cute.ceil_div(nq, 64):
                        self._load_query_pair(
                            atoms,
                            tensors,
                            dptrs,
                            sA,
                            sB,
                            qstart,
                            first_q_tile,
                            qhead,
                            half,
                            batch,
                            storage.load_barrier.ptr,
                        )
                        cute.arch.mbarrier_wait(storage.load_barrier.ptr, load_phase)
                        load_phase ^= 1
                        cute.arch.sync_threads()
                        self._after_sync()
                        self._mma_score_rolled(mma_score, sA, sK, ptr.toint() + 384, True)
                        self._mma_score_rolled(mma_score, sB, sV, ptr.toint() + 416, True)
                        self._mma_wait(storage.mma_barrier.ptr, mma_phase)
                        mma_phase ^= 1
                    for qt in cutlass.range(first_q_tile, cute.ceil_div(nq, 64)):
                        # Loop invariant: this tile's S/dP pair is complete.
                        current_s = cute.make_tensor(
                            ptr + 384 + score_bank * 64, self._acc_layout(64)
                        )
                        current_p = cute.make_tensor(
                            ptr + 416 + score_bank * 64, self._acc_layout(64)
                        )
                        if tid < 128:
                            rs = cute.make_rmem_tensor(coords.shape, Float32)
                            rp = cute.make_rmem_tensor(coords.shape, Float32)
                            cute.copy(ld, ld.partition_S(current_s), rs)
                            cute.copy(ld, ld.partition_S(current_p), rp)
                            cute.arch.fence_view_async_tmem_load()
                            for i in cutlass.range_constexpr(cute.size(rs)):
                                row, col = coords[i]
                                outgoing[0, row, col] = rs[i]
                                outgoing[1, row, col] = rp[i]
                        cute.arch.sync_threads()
                        cute.arch.fence_view_async_shared()
                        if tid == 192:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                storage.exchange_barrier.ptr, 32768, peer_cta_rank_in_cluster=peer
                            )
                            copy_utils.cpasync_bulk_s2cluster(
                                outgoing.iterator,
                                incoming.iterator,
                                storage.exchange_barrier.ptr,
                                32768,
                                peer_cta_rank_in_cluster=peer,
                            )
                        # Statistics are independent of the incoming partials:
                        # overlap their guarded load chain with the DSM transfer.
                        row_lse, row_dpsum = Float32(float("inf")), Float32(0.0)
                        if tid < 128:
                            # Layout E with Ld32x32b maps each thread to one
                            # logical row and 32 columns. Load row statistics
                            # once; no valid key means no LSE/D read at all.
                            stat_row, _ = coords[0]
                            stat_qi = qt * 64 + stat_row
                            if stat_qi < nq and tile * 64 <= stat_qi + nk - nq:
                                row_lse = self._stat(LSE, batch, qhead, Int64(qstart) + stat_qi)
                                if row_lse != Float32(float("inf")) and row_lse != Float32(
                                    -float("inf")
                                ):
                                    row_dpsum = self._stat(D, batch, qhead, Int64(qstart) + stat_qi)
                        cute.arch.mbarrier_wait(storage.exchange_barrier.ptr, exchange_phase)
                        exchange_phase ^= 1
                        # Consume peer partials before ACK permits either CTA to
                        # send the next tile into this same incoming buffer.
                        merged_s = cute.make_rmem_tensor(coords.shape, Float32)
                        merged_p = cute.make_rmem_tensor(coords.shape, Float32)
                        if tid < 128:
                            cute.copy(ld, ld.partition_S(current_s), merged_s)
                            cute.copy(ld, ld.partition_S(current_p), merged_p)
                            cute.arch.fence_view_async_tmem_load()
                            for i in cutlass.range_constexpr(cute.size(merged_s)):
                                row, col = coords[i]
                                peer_s = incoming[0, row, col]
                                peer_p = incoming[1, row, col]
                                if half == 0:
                                    merged_s[i] = merged_s[i] + peer_s
                                    merged_p[i] = merged_p[i] + peer_p
                                else:
                                    merged_s[i] = peer_s + merged_s[i]
                                    merged_p[i] = peer_p + merged_p[i]
                        # Whole-cluster ACK protects both the outgoing source
                        # from overwrite and incoming from the next peer send.
                        cute.arch.cluster_arrive()
                        cute.arch.cluster_wait()
                        # The DSM ACK frees the old Q/send slot. Qt/dOt write
                        # only sA/sB, disjoint from the incoming partials and
                        # P/dS views, so TMA can overlap their conversion.
                        if warp == self.load_warp:
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    storage.load_barrier.ptr, 65536
                                )
                            self._load(
                                5,
                                atoms,
                                tensors,
                                dptrs,
                                self._logical(sAt, 256, 64),
                                qstart,
                                qt,
                                qhead,
                                half,
                                batch,
                                storage.load_barrier.ptr,
                                True,
                            )
                            self._load(
                                6,
                                atoms,
                                tensors,
                                dptrs,
                                self._logical(sBt, 256, 64),
                                qstart,
                                qt,
                                qhead,
                                half,
                                batch,
                                storage.load_barrier.ptr,
                                True,
                            )
                        if tid < 128:
                            for i in cutlass.range_constexpr(cute.size(merged_s)):
                                row, col = coords[i]
                                score, dp = merged_s[i], merged_p[i]
                                qi, ki = qt * 64 + row, tile * 64 + col
                                p, ds = Float32(0.0), Float32(0.0)
                                if qi < nq and ki < nk and ki <= qi + nk - nq:
                                    if row_lse != Float32(float("inf")) and row_lse != Float32(
                                        -float("inf")
                                    ):
                                        p = cute.math.exp2(
                                            score * (scale * math.log2(math.e)) - row_lse,
                                            fastmath=True,
                                        )
                                        ds = p * (dp - row_dpsum) * scale
                                p_view[col, row] = self.dtype(p)
                                dst_view[col, row] = self.dtype(ds)
                                ds_view[row, col] = self.dtype(ds)
                        cute.arch.sync_threads()
                        cute.arch.fence_view_async_shared()
                        cute.arch.mbarrier_wait(storage.load_barrier.ptr, load_phase)
                        load_phase ^= 1
                        cute.arch.sync_threads()
                        self._after_sync()
                        self._mma(mma_grad, sDST, sAt, ptr.toint(), False)
                        self._mma(mma_grad, sP, sBt, ptr.toint() + 128, False)
                        self._mma(mma_grad, sDS, sKt, ptr.toint() + 256, True)
                        self._mma_wait(storage.mma_barrier.ptr, mma_phase)
                        mma_phase ^= 1
                        # Separate roles: warp4 loads, warp5 computes next S/dP,
                        # and warps0..3 +warp6 drain current dQ concurrently.
                        has_next = qt + 1 < cute.ceil_div(nq, 64)
                        if has_next:
                            self._load_query_pair(
                                atoms,
                                tensors,
                                dptrs,
                                sA,
                                sB,
                                qstart,
                                qt + 1,
                                qhead,
                                half,
                                batch,
                                storage.load_barrier.ptr,
                            )
                            if warp == self.mma_warp:
                                cute.arch.mbarrier_wait(storage.load_barrier.ptr, load_phase)
                                self._after_sync()
                                next_score = ptr.toint() + 384 + (score_bank ^ 1) * 64
                                self._mma_score_rolled(mma_score, sA, sK, next_score, True)
                                self._mma_score_rolled(mma_score, sB, sV, next_score + 32, True)
                                with cute.arch.elect_one():
                                    tcgen05.commit(storage.mma_barrier.ptr)
                        if const_expr(cuQ is not None):
                            if const_expr(self.deterministic):
                                padded_start = ((Int64(qstart) + Int64(batch) * 64) // 64) * 64
                            else:
                                padded_start = ((qstart + batch * 64) // 64) * 64
                            dq_cur = dQ[qhead, half, None, None]
                        else:
                            padded_start = Int32(0)
                            dq_cur = dQ[batch, qhead, half, None, None]
                        if warp < 4 or warp == 6:
                            if const_expr(self.deterministic):
                                sem_row = Int64(padded_start) // 64 + Int64(qt)
                                if sem_row < 0 or sem_row >= Int64(semaphores.shape[-1]):
                                    cute.arch.inline_ptx("trap;", read_only_args=[])
                                if const_expr(cuQ is not None):
                                    sem_ptr = cute.domain_offset(
                                        (Int64(qhead), Int64(half), sem_row), semaphores
                                    ).iterator
                                else:
                                    sem_ptr = cute.domain_offset(
                                        (Int64(batch), Int64(qhead), Int64(half), sem_row),
                                        semaphores,
                                    ).iterator
                                if tid == 192:
                                    self._wait_owner(sem_ptr, tile)
                            # The first 160-thread ready barrier includes the
                            # acquiring issuer; producers may fill SMEM meanwhile.
                            # TMEM_FRAG_1CTA: 96,96,64 columns are contiguous
                            # 24,24,16KiB spans in the unchanged global tile.
                            # The middle span crosses the 128-column lane halves.
                            # Source reuse still waits for the bulk reads.
                            for chunk in cutlass.range_constexpr(3):
                                if tid < 128:
                                    rg = cute.make_rmem_tensor(grad_coords.shape, Float32)
                                    cute.copy(ld_grad, ld_grad.partition_S(tDQ), rg)
                                    cute.arch.fence_view_async_tmem_load()
                                    if const_expr(chunk == 0):
                                        if tid // 64 == 0:
                                            rg_chunk = cute.make_tensor(
                                                rg.iterator, dq_store_dst96.shape
                                            )
                                            cute.copy(dq_store, rg_chunk, dq_store_dst96)
                                    elif const_expr(chunk == 1):
                                        if tid // 64 == 0:
                                            rg_chunk = cute.make_tensor(
                                                rg.iterator + 96, dq_store_dst32.shape
                                            )
                                            cute.copy(dq_store, rg_chunk, dq_store_dst32)
                                        else:
                                            rg_chunk = cute.make_tensor(
                                                rg.iterator, dq_store_dst64_hi.shape
                                            )
                                            cute.copy(dq_store, rg_chunk, dq_store_dst64_hi)
                                    else:
                                        if tid // 64 == 1:
                                            rg_chunk = cute.make_tensor(
                                                rg.iterator + 64, dq_store_dst64.shape
                                            )
                                            cute.copy(dq_store, rg_chunk, dq_store_dst64)
                                cute.arch.fence_view_async_shared()
                                self._drain_sync()
                                if tid == 192:
                                    gptr = (
                                        cute.domain_offset(
                                            (Int64(padded_start) + qt * 64, 0), dq_cur
                                        ).iterator
                                        + chunk * 6144
                                    )
                                    copy_utils.cpasync_reduce_bulk_add_f32(
                                        reduce_stage.iterator,
                                        gptr,
                                        24576 if const_expr(chunk < 2) else 16384,
                                    )
                                    cute.arch.cp_async_bulk_commit_group()
                                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                                self._drain_sync()
                            if const_expr(self.deterministic):
                                if tid == 192:
                                    # The same issuer owns all three bulk groups.
                                    # A source-read wait does not publish dQ adds.
                                    cute.arch.cp_async_bulk_wait_group(0, read=False)
                                    barrier.arrive_inc(sem_ptr, 0, 0, 1)
                        # Full CTA join is outside all role-specific paths.
                        # It proves the score commit was issued and dQ source
                        # consumption finished before any next-tile export.
                        cute.arch.sync_threads()
                        if has_next:
                            # Warp5 already observed this load; all roles advance
                            # the phase once here, then wait WITHOUT a second commit.
                            cute.arch.mbarrier_wait(storage.load_barrier.ptr, load_phase)
                            load_phase ^= 1
                            cute.arch.mbarrier_wait(storage.mma_barrier.ptr, mma_phase)
                            mma_phase ^= 1
                            cute.arch.sync_threads()
                            self._after_sync()
                            score_bank ^= 1
                self._epilogue(
                    tDK,
                    dK,
                    self._logical(sA, 64, 256),
                    batch,
                    kstart,
                    tile,
                    kv_head,
                    half,
                    nk,
                    tid,
                    ld_grad,
                    grad_coords,
                )
                self._epilogue(
                    tDV,
                    dV,
                    self._logical(sA, 64, 256),
                    batch,
                    kstart,
                    tile,
                    kv_head,
                    half,
                    nk,
                    tid,
                    ld_grad,
                    grad_coords,
                )
                # Pair remains in lockstep across owner changes and descriptor
                # reuse; no CTA may exit while its peer can still target DSMEM.
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            # No lookahead: both CTAs complete this owner before claiming next.
            work = self._claim_owner(storage, counter, half, tid)
        if tid == 192:
            cute.arch.cp_async_bulk_wait_group(0, read=False)
        cute.arch.sync_threads()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        allocator.free(ptr)
