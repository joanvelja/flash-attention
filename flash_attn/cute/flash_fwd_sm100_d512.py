# Copyright (c) 2026, Colfax International.
"""D512 attention with independent K and V on datacenter Blackwell.

The M64 weight-stationary MMA and Layout E accumulators are shared with the
MLA forward. Unlike absorbed MLA, the score and value operands are independent:
the 128 KiB operand ring receives K[0:256], K[256:512], V[0:256], V[256:512]
for each key block. UMMA completion releases each K slot before V overwrites it.

Per CTA: 384/512 TMEM columns (two 64-column S slots and two 128-column O
slots), 64 KiB stationary Q, 128 KiB K/V ring, 16 KiB P, plus synchronization
and softmax state. Q uses TMA or packed-row cp.async gather; K/V use
zero-filling predicated cp.async so a packed sequence's tail never reads the
next sequence's values. Paged K/V are gathered directly. O is predicated.
The shared-K/V case follows the same schedule and has exactly the same math.
"""

import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05

from flash_attn.cute import copy_utils
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute.flash_fwd_mla_1cta_sm100 import FlashAttentionMLAForward1CtaSm100
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100_MLA2CTA
from flash_attn.cute.pack_gqa import pack_gqa_layout
from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.softmax import SoftmaxSm100
from flash_attn.cute.tile_scheduler import TileSchedulerArguments, TileSchedulerProtocol


class FlashAttentionForwardSm100D512(FlashAttentionMLAForward1CtaSm100):
    """Native 16-bit D512 forward, standard FA4 Q/K/V/O/LSE layouts."""

    def __init__(
        self,
        qhead_per_kvhead: int,
        nheads_kv: int,
        is_causal: bool = True,
        has_cu_seqlens_q: bool = True,
        has_seqused_q: bool = False,
        pack_gqa: bool = True,
        window_size_left: Optional[int] = None,
        window_size_right: Optional[int] = None,
        use_clc_scheduler: bool = False,
        rescale_threshold: float = 0.0,
    ):
        assert qhead_per_kvhead > 0 and nheads_kv > 0
        # The MLA base limits its sparse packed-head tile to 128 heads. Dense
        # attention has no such limit: large ratios use ordinary Q-head tiles.
        base_ratio = qhead_per_kvhead if qhead_per_kvhead <= 128 else 1
        pack_gqa = pack_gqa and qhead_per_kvhead <= 128
        super().__init__(
            is_causal=is_causal,
            qhead_per_kvhead=base_ratio,
            nheads_kv=nheads_kv,
            hdim=64,  # unused optional MLA RoPE operand; Q itself is the D512 operand
            hdimv=512,
            has_qk=False,
            pack_gqa=pack_gqa,
            has_cu_seqlens_q=has_cu_seqlens_q,
            has_seqused_q=has_seqused_q,
            use_cpasync_load_KV=True,
            use_clc_scheduler=use_clc_scheduler,
            rescale_threshold=rescale_threshold,
        )
        # Predicated stores also avoid a storage-lifetime dependency between the
        # operand ring and the previous work tile's epilogue under CLC.
        self.qhead_per_kvhead = qhead_per_kvhead
        self.qhead_per_kvhead_valid = qhead_per_kvhead
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.is_local = window_size_left is not None or window_size_right is not None
        self.use_tma_O = False
        self.has_learnable_sink = False
        self.cpasync_barrier = None
        assert not self.s_ahead

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        softmax_scale: Float32,
        cu_seqlens_q: Optional[cute.Tensor] = None,
        cu_seqlens_k: Optional[cute.Tensor] = None,
        seqused_q: Optional[cute.Tensor] = None,
        seqused_k: Optional[cute.Tensor] = None,
        page_table: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        assert q.element_type in (cutlass.BFloat16, cutlass.Float16)
        assert all(t.element_type == q.element_type for t in (k, v, o)), (
            "SM100 D512 requires matching fp16 or bf16 Q, K, V and O"
        )
        # interface._validate_sm100_d512_inputs checks the runtime head widths.
        # to_cute_tensor marks extents dynamic; all MMA sizes come from the
        # constructor rather than asking Python to coerce a staged comparison.
        assert lse.element_type == Float32
        assert page_table is None or cu_seqlens_k is None
        assert (cu_seqlens_q is not None) == self.has_cu_seqlens_q
        assert (seqused_q is not None) == self.has_seqused_q
        self.dtype_Q = self.dtype_K = self.dtype_Qv = self.dtype_V = q.element_type
        self.dtype_P = self.dtype_O = q.element_type

        # A singleton's outer stride is never used for addressing. Canonicalize
        # it before asserting alignment, and retain static broadcast strides.
        q, k, v, o = [
            assume_tensor_aligned(t, canonicalize_singletons=True) for t in (q, k, v, o)
        ]
        qo_modes = [1, 3, 2, 0] if const_expr(cu_seqlens_q is None) else [0, 2, 1]
        kv_modes = [1, 3, 2, 0] if const_expr(cu_seqlens_k is None) else [0, 2, 1]
        q, o = [cute.make_tensor(t.iterator, cute.select(t.layout, mode=qo_modes)) for t in (q, o)]
        k, v = [cute.make_tensor(t.iterator, cute.select(t.layout, mode=kv_modes)) for t in (k, v)]
        # FA4's public LSE is (heads,total) / (batch,heads,seqlen), in natural log units.
        lse_modes = [2, 1, 0] if const_expr(cu_seqlens_q is None) else [1, 0]
        lse = cute.make_tensor(lse.iterator, cute.select(lse.layout, mode=lse_modes))
        if const_expr(self.pack_gqa):
            q, o = [
                pack_gqa_layout(t, self.qhead_per_kvhead, self.nheads_kv, head_idx=2)
                for t in (q, o)
            ]
            lse = pack_gqa_layout(lse, self.qhead_per_kvhead, self.nheads_kv, head_idx=1)
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(o)

        mma_qk, mma_q, mma_score, mma_out = [
            sm100_utils.make_trivial_tiled_mma(
                self.dtype_Qv,
                self.dtype_Qv,
                major_a,
                major_b,
                Float32,
                self.cta_group,
                tiler[:2],
                self.operand_source_A,
            )
            for major_a, major_b, tiler in [
                (self.major_mode_Q, self.major_mode_K, self.mma_tiler_QK),
                (self.major_mode_Q, self.major_mode_K, self.mma_tiler_QK),
                (self.major_mode_Qvi, self.major_mode_Vi, self.mma_tiler_QvV),
                (self.major_mode_P, self.major_mode_Vti, self.mma_tiler_PVt),
            ]
        ]
        for attr, make_fn, mma, tiler, stages in [
            ("sQ_layout", sm100_utils.make_smem_layout_a, mma_q, self.mma_tiler_QK, 1),
            ("sK_layout", sm100_utils.make_smem_layout_b, mma_qk, self.mma_tiler_QK, 1),
            ("sQv_layout", sm100_utils.make_smem_layout_a, mma_score, self.mma_tiler_QvV, 2),
            ("sV_layout", sm100_utils.make_smem_layout_b, mma_score, self.mma_tiler_QvV, 2),
            ("sVt_layout", sm100_utils.make_smem_layout_b, mma_out, self.mma_tiler_PVt, 2),
            ("sP_layout", sm100_utils.make_smem_layout_a, mma_out, self.mma_tiler_PVt, 1),
        ]:
            dtype_arg = "a_dtype" if make_fn is sm100_utils.make_smem_layout_a else "b_dtype"
            layout = make_fn(
                tiled_mma=mma, mma_tiler_mnk=tiler, num_stages=stages, **{dtype_arg: self.dtype_Qv}
            )
            setattr(self, attr + "_staged", layout)
            setattr(self, attr, cute.select(layout, mode=[0, 1, 2]))
        self.sQst_layout = None
        assert cute.cosize(self.sV_layout_staged) == cute.cosize(self.sVt_layout_staged)
        self.sStats_layout = cute.make_layout((64, 2))
        self.sScale_layout = cute.make_layout((64, 2))
        self.sBitmask_layout = cute.make_layout((4, 2))
        self.tma_copy_bytes_Q = self.tma_copy_bytes_K = 0
        self.tma_copy_bytes_Qvi = cute.size_in_bytes(self.dtype_Qv, self.sQv_layout)
        self.tma_copy_bytes_Vi = cute.size_in_bytes(self.dtype_V, self.sV_layout)
        cta_layout = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (mma_score.thr_id.shape,)
        )
        tma_q, q_tma, copy_q = None, q, None
        if const_expr(self.use_tma_QQv):
            tma_q, q_tma = cute.nvgpu.make_tiled_tma_atom_A(
                cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
                q,
                self.sQv_layout,
                self.mma_tiler_QvV,
                mma_score,
                cta_layout.shape,
            )
        else:
            copy_q = copy_utils.tiled_copy_2d(
                self.dtype_Qv,
                self._packed_threads_per_row(256) * 8,
                self.num_load_threads,
                is_async=True,
            )
        sO_layout = sm100_utils.make_smem_layout_epi(self.dtype_O, self.o_layout, self.epi_tile, 2)
        self.overlap_sO_sV = True
        copy_o = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype_O, num_bits_per_copy=128),
            cute.make_layout((64, 2), stride=(1, 64)),
            cute.make_layout((1, 128)),
        )
        storage = self._get_shared_storage_cls()
        self.smem_bytes = storage.size_in_bytes()
        assert self.smem_bytes <= 227 * 1024, (
            f"D512 forward needs {self.smem_bytes} shared-memory bytes"
        )
        scheduler = self.TileScheduler
        batch_count = (
            cu_seqlens_q.shape[0] - 1 if const_expr(cu_seqlens_q is not None) else q.shape[3]
        )
        total_q = cute.size(q.shape[0]) * (
            1 if const_expr(cu_seqlens_q is not None) else q.shape[3]
        )
        args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(q.shape[0]), 64),
            num_head=cute.size(q.shape[2]),
            num_batch=batch_count,
            num_splits=Int32(1),
            seqlen_k=cute.size(v.shape[0])
            * (page_table.shape[1] if const_expr(page_table is not None) else 1),
            headdim=512,
            headdim_v=512,
            total_q=total_q,
            tile_shape_mn=(64, 128),
            mCuSeqlensQ=cu_seqlens_q,
            mSeqUsedQ=seqused_q,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=2,
            is_persistent=self.is_persistent,
            lpt=False,
            is_split_kv=False,
            cluster_shape_mn=self.cluster_shape_mn,
            use_cluster_idx=True,
        )
        params = scheduler.to_underlying_arguments(args, scheduling_mode=self.scheduling_mode)
        self.tile_scheduler_cls = scheduler
        for name, barrier_id, threads in [
            ("softmax_barrier", NamedBarrierFwdSm100_MLA2CTA.Softmax, 128),
            ("epi_barrier", NamedBarrierFwdSm100_MLA2CTA.Epilogue, 128),
            ("sm_stats_barrier_full", NamedBarrierFwdSm100_MLA2CTA.SoftmaxStatsFull, 256),
            ("sm_stats_barrier_empty", NamedBarrierFwdSm100_MLA2CTA.SoftmaxStatsEmpty, 256),
        ]:
            setattr(
                self, name, pipeline.NamedBarrier(barrier_id=int(barrier_id), num_threads=threads)
            )

        self.kernel(
            None,
            q_tma,
            k,
            v,
            o,
            lse,
            None,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            None,
            None,
            None,
            None,
            tma_q,
            None,
            None,
            None,
            copy_o,
            None,
            copy_q,
            self.sQ_layout_staged,
            None,
            self.sK_layout_staged,
            self.sQv_layout_staged,
            self.sV_layout_staged,
            self.sVt_layout_staged,
            self.sP_layout_staged,
            self.sStats_layout,
            self.sScale_layout,
            self.sBitmask_layout,
            sO_layout,
            mma_qk,
            mma_q,
            mma_score,
            mma_out,
            softmax_scale * math.log2(math.e),
            Int32(1),
            params,
            storage,
        ).launch(
            grid=scheduler.get_grid_shape(params),
            block=(self.num_threads, 1, 1),
            cluster=self.cluster_shape_mnk,
            smem=self.smem_bytes,
            stream=stream,
        )

    @cute.jit
    def load_cpasync(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPageTable,
        mIndexTopk,
        sK,
        sV: cute.Tensor,
        sBitmask,
        pipeline_K,
        pipeline_V: pipeline.PipelineAsyncUmma,
        pipeline_bitmask,
        sO_empty_mbar_ptr,
        staging_mbar_ptr,
        sK_free_mbar_ptr,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        num_splits: Int32,
        tile_scheduler: TileSchedulerProtocol,
    ):
        producer = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, stages=2)
        tidx = cute.arch.thread_idx()[0] % 128
        tiled_copy = copy_utils.tiled_copy_2d(self.dtype_V, 256, 128, is_async=True)
        thr_copy = tiled_copy.get_slice(tidx)
        if const_expr(mPageTable is not None):
            page_size_divmod = cute.FastDivmodDivisorV2(cute.size(mV.shape[0]))
        tile = tile_scheduler.initial_work_tile_info()
        while tile.is_valid_tile:
            m_block, head, batch, split = self._tile_coords(
                tile.tile_idx, SeqlenInfoCls.keywords["mCuSeqlensQ"]
            )
            head_kv = head if const_expr(self.pack_gqa) else head // self.qhead_per_kvhead
            seqlen = SeqlenInfoCls(batch)
            lo, hi = self._n_block_range(block_info, seqlen, m_block, split)
            if block_info.has_kv_work(lo, hi):
                if const_expr(mPageTable is None):
                    gK = seqlen.offset_batch_K(mK, batch, dim=3)[None, None, head_kv]
                    gV = seqlen.offset_batch_K(mV, batch, dim=3)[None, None, head_kv]
                    for i in cutlass.range(cutlass.max(hi - lo, 1), unroll=1):
                        n_block = hi - 1 - i if hi > lo else 0
                        for part in cutlass.range_constexpr(2):
                            producer = self.load_operand(
                                gK,
                                sV,
                                pipeline_V,
                                producer,
                                thr_copy,
                                n_block,
                                part,
                                seqlen.seqlen_k,
                            )
                        for part in cutlass.range_constexpr(2):
                            producer = self.load_operand(
                                gV,
                                sV,
                                pipeline_V,
                                producer,
                                thr_copy,
                                n_block,
                                part,
                                seqlen.seqlen_k,
                            )
                else:
                    make_manager = partial(
                        PagedKVManager.create,
                        mPageTable,
                        page_size_divmod=page_size_divmod,
                        bidb=batch,
                        bidh=head_kv,
                        thread_idx=tidx,
                        seqlen_k=seqlen.seqlen_k,
                        leftpad_k=0,
                        n_block_size=self.tile_n,
                        head_dim_padded=256,
                        head_dim_v_padded=256,
                        num_threads=self.num_cpasync_load_threads,
                        dtype=self.dtype_V,
                    )
                    # Both use the manager's non-transposed K address mode. V is
                    # transposed only through the UMMA SMEM descriptor, not gmem.
                    paged_k = make_manager(mK_paged=mK, mV_paged=mK)
                    paged_v = make_manager(mK_paged=mV, mV_paged=mV)
                    for i in cutlass.range(cutlass.max(hi - lo, 1), unroll=1):
                        n_block = hi - 1 - i if hi > lo else 0
                        # Page-table register writes stay in this loop's SSA scope.
                        paged_k.load_page_table(n_block)
                        for part in cutlass.range_constexpr(2):
                            producer = self.cpasync_load_KV(
                                paged_k,
                                pipeline_V,
                                sV,
                                256,
                                n_block,
                                producer,
                                d_offset=part * 256,
                            )
                        paged_v.load_page_table(n_block)
                        for part in cutlass.range_constexpr(2):
                            producer = self.cpasync_load_KV(
                                paged_v,
                                pipeline_V,
                                sV,
                                256,
                                n_block,
                                producer,
                                d_offset=part * 256,
                            )
            tile = tile_scheduler.advance_to_next_work()
        pipeline_V.producer_tail(producer)

    @cute.jit
    def load_operand(
        self,
        gX: cute.Tensor,
        sX: cute.Tensor,
        pipe: pipeline.PipelineAsyncUmma,
        state: pipeline.PipelineState,
        thr_copy: cute.ThrCopy,
        n_block: Int32,
        part: cutlass.Constexpr[int],
        seqlen_k: Int32,
    ):
        stage = state.index
        pipe.producer_acquire(state)
        shared = cute.composition(
            sX[None, None, None, stage], cute.make_ordered_layout((128, 256), order=(0, 1))
        )
        global_tile = cute.local_tile(gX, (128, 256), (n_block, part))
        src = thr_copy.partition_S(global_tile)
        dst = thr_copy.partition_D(shared)
        coords = thr_copy.partition_S(cute.make_identity_tensor((128, 256)))
        for row in cutlass.range_constexpr(cute.size(dst, mode=[1])):
            valid = n_block * 128 + coords[0, row, 0][0] < seqlen_k
            for col in cutlass.range_constexpr(cute.size(dst, mode=[2])):
                # One predicate per 16-byte transaction, not per bf16 element.
                pred = cute.make_fragment_like(dst[(0, None), row, col], Boolean)
                pred.fill(valid)
                cute.copy(thr_copy, src[None, row, col], dst[None, row, col], pred=pred)
        cute.arch.cp_async_commit_group()
        pipe.sync_object_full.arrive_cp_async_mbarrier(stage)
        state.advance()
        return state

    @cute.jit
    def mma_S_step(
        self,
        gemm_QvV,
        gemm_QK,
        pipeline_S,
        pipeline_V,
        pipeline_K,
        tSrQ,
        tSrK,
        sK,
        tSrQv,
        sQv,
        tSrV,
        sV,
        producer_state_S,
        consumer_state_V_wait,
        consumer_state_K,
        stage: cutlass.Constexpr[Int32],
    ):
        pipeline_S.producer_acquire(producer_state_S)
        for part in cutlass.range_constexpr(2):
            slot = consumer_state_V_wait.index
            pipeline_V.consumer_wait(consumer_state_V_wait)
            gemm_QvV[stage](
                tCrA=tSrQv[None, None, None, part],
                tCrB=tSrV[None, None, None, slot],
                sA=sQv[None, None, None, part],
                sB=sV[None, None, None, slot],
                zero_init=part == 0,
            )
            # This tcgen05 completion-backed release orders V overwriting K.
            pipeline_V.consumer_release(consumer_state_V_wait)
            consumer_state_V_wait.advance()
        pipeline_S.producer_commit(producer_state_S)
        producer_state_S.advance()
        # The next two transactions are V, consumed by the PV step.
        for _ in cutlass.range_constexpr(2):
            consumer_state_V_wait.advance()
        return producer_state_S, consumer_state_V_wait, consumer_state_K

    @cute.jit
    def mma_PVt_step(
        self,
        gemm_PVt,
        pipelines_O,
        pipeline_P,
        pipeline_V,
        tOrP,
        sP,
        tOrVt,
        sVt,
        producer_state_O0,
        producer_state_O1,
        consumer_state_P,
        consumer_state_V_release,
        O_should_accumulate,
    ):
        # Skip this block's K transactions; S has already consumed/released them.
        for _ in cutlass.range_constexpr(2):
            consumer_state_V_release.advance()
        pipeline_P.consumer_wait(consumer_state_P)
        states = [producer_state_O0, producer_state_O1]
        for part in cutlass.range_constexpr(2):
            state = states[part]
            pipelines_O[part].producer_acquire(state)
            pipeline_V.consumer_wait(consumer_state_V_release)
            slot = consumer_state_V_release.index
            gemm_PVt[part](
                tCrA=tOrP[None, None, None, consumer_state_P.index],
                tCrB=tOrVt[None, None, None, slot],
                sA=sP[None, None, None, consumer_state_P.index],
                sB=sVt[None, None, None, slot],
                zero_init=~O_should_accumulate,
            )
            pipelines_O[part].producer_commit(state)
            state.advance()
            states[part] = state
            pipeline_V.consumer_release(consumer_state_V_release)
            consumer_state_V_release.advance()
        pipeline_P.consumer_release(consumer_state_P)
        consumer_state_P.advance()
        return states[0], states[1], consumer_state_P, consumer_state_V_release, Boolean(True)

    @cute.jit
    def _n_block_range(self, block_info, seqlen, m_block: Int32, split_idx: Int32):
        # All warp roles use this same range, including empty/fully-masked tiles.
        info = BlockInfo(
            self.cta_tile_m,
            self.tile_n,
            is_causal=self.is_causal,
            is_local=self.is_local,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        lo, hi = info.get_n_block_min_max(seqlen, m_block)
        hi = cutlass.max(hi, 0)
        return cutlass.min(lo, hi), hi

    @cute.jit
    def apply_mask(
        self,
        tSrS: cute.Tensor,
        tScS_t2r: cute.Tensor,
        seqlen,
        m_block: Int32,
        n_block: Int32,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
    ):
        row = tScS_t2r[0][0] + m_block * self.cta_tile_m
        if const_expr(self.pack_gqa):
            row = row // self.qhead_per_kvhead
        center = row + seqlen.seqlen_k - seqlen.seqlen_q
        right = seqlen.seqlen_k
        left = Int32(0)
        if const_expr(self.is_causal):
            right = cutlass.min(right, center + 1)
        if const_expr(self.window_size_right is not None):
            right = cutlass.min(right, center + self.window_size_right + 1)
        if const_expr(self.window_size_left is not None):
            left = cutlass.max(left, center - self.window_size_left)
        for i in cutlass.range_constexpr(cute.size(tSrS)):
            col = tScS_t2r[i][1] + n_block * self.tile_n
            if col < left or col >= right or row >= seqlen.seqlen_q:
                tSrS[i] = -Float32.inf

    @cute.jit
    def softmax_loop(
        self,
        softmax_scale_log2: Float32,
        mLSE: Optional[cute.Tensor],
        sRowMax: cute.Tensor,
        sRowSum: cute.Tensor,
        sScale: cute.Tensor,
        sBitmask: Optional[cute.Tensor],
        sP: cute.Tensor,
        tmem_ptr: cute.Pointer,
        pipeline_S: pipeline.PipelineAsync,
        pipeline_P: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        pipeline_bitmask: Optional[pipeline.PipelineAsync],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        num_splits: Int32,
        descale_tensors,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # ==== softmax warpgroup ====
        # Description: computes softmax on S and writes the result to P
        # Produces: P, softmax stats
        # Consumes: S

        tidx = cute.arch.thread_idx()[0] % self.num_softmax_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % (
            self.num_softmax_threads // 32
        )

        # Hand-built Layout E packed S accumulators (one per stage).
        acc_layout_S = self._acc_tmem_layout(self.tile_n)
        tSAcc_staged = [
            cute.make_tensor(tmem_ptr + self.tmem_offset_S[stage], acc_layout_S)
            for stage in range(self.num_stages_S)
        ]
        tSAcc = tSAcc_staged[0]

        # S tmem -> rmem copy objects
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.dtype_acc,
        )
        tmem_load_tiled = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc)
        tmem_load_thr = tmem_load_tiled.get_slice(tidx)
        tStS_t2r_staged = [
            tmem_load_thr.partition_S(tSAcc_staged[stage]) for stage in range(self.num_stages_S)
        ]
        # logical (row, col) coordinates matching the packed layout
        tScS_t2r = tmem_load_thr.partition_D(self._acc_coord_view(self.tile_n))
        tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, self.dtype_acc)

        # P rmem -> smem copy objects
        universal_copy_bits = 128
        smem_store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype_P,
            num_bits_per_copy=universal_copy_bits,
        )
        smem_store_tiled = cute.make_tiled_copy_D(smem_store_atom, tmem_load_tiled)
        smem_store_thr = smem_store_tiled.get_slice(tidx)
        # P rmem -> smem copy operands
        sP_mnp_layout = cute.make_ordered_layout(
            self.tile_P + (self.num_stages_P,), order=(0, 1, 2)
        )
        sP_mnp = cute.composition(sP, sP_mnp_layout)
        sP_smem_view = smem_store_thr.partition_D(sP_mnp)

        Consumer, Producer = pipeline.PipelineUserType.Consumer, pipeline.PipelineUserType.Producer
        consumer_state_S = pipeline.make_pipeline_state(Consumer, stages=self.num_stages_S)
        producer_state_P = pipeline.make_pipeline_state(Producer, stages=self.num_stages_P)
        producer_state_sm_stats = pipeline.make_pipeline_state(
            Producer, stages=self.num_stages_sm_stats
        )
        # Threaded through every softmax step; only waited on / advanced when sparse.
        consumer_state_bitmask = pipeline.make_pipeline_state(
            Consumer, stages=self.num_stages_bitmask
        )

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = self._tile_coords(
                work_tile.tile_idx, SeqlenInfoCls.keywords["mCuSeqlensQ"]
            )
            seqlen = SeqlenInfoCls(batch_idx)
            # fp8: fold q_descale * v_descale into the softmax scale (see
            # _effective_descales); log2 folding is valid because the descale multiplies
            # S, and softmax_scale_log2 == softmax_scale * log2(e).
            qk_descale, _ = self._effective_descales(descale_tensors, batch_idx, head_idx)
            softmax_scale_log2_eff = softmax_scale_log2 * qk_descale
            n_block_min, n_block_max = self._n_block_range(block_info, seqlen, m_block, split_idx)
            num_n_blocks = n_block_max - n_block_min
            # Split-kv: this split's slice of the n_block range can be empty (the
            # per-split count is rounded up, so empties are the trailing splits).
            # Every warp role skips the whole tile body together, which keeps all
            # pipeline phases and raw-mbar phase counters consistent; the epilogue
            # still writes LSE = -inf so the combine kernel can ignore this split.
            has_work = block_info.has_kv_work(n_block_min, n_block_max)
            if has_work:
                even_n_blocks = num_n_blocks % 2 == 0 and num_n_blocks > 0
                num_n_block_groups = cute.ceil_div(num_n_blocks, self.num_stages_S)

                mask_fn = partial(
                    self.apply_mask,
                    tScS_t2r=tScS_t2r,
                    seqlen=seqlen,
                    m_block=m_block,
                    mask_causal=self.is_causal,
                )

                softmax = SoftmaxSm100.create(
                    softmax_scale_log2_eff,
                    rescale_threshold=self.rescale_threshold,
                    max_offset=self.max_offset,
                )
                softmax.reset()

                softmax_step_fn = partial(
                    self.softmax_step,
                    softmax,
                    sRowMax,
                    sScale,
                    tStS_t2r_staged,
                    tSrS_t2r,
                    sP_smem_view,
                    tmem_load_thr,
                    smem_store_thr,
                    pipeline_S,
                    pipeline_P,
                    pipeline_sm_stats,
                    tidx,
                    warp_idx,
                    sBitmask=sBitmask,
                    pipeline_bitmask=pipeline_bitmask,
                    tScS_t2r=tScS_t2r,
                )

                ### first iteration ###
                n_block = n_block_max - 1
                (
                    consumer_state_S,
                    producer_state_P,
                    producer_state_sm_stats,
                    consumer_state_bitmask,
                ) = softmax_step_fn(
                    consumer_state_S,
                    producer_state_P,
                    producer_state_sm_stats,
                    0,
                    n_block,
                    mask_fn=partial(mask_fn, mask_seqlen=True),
                    is_first=True,
                    consumer_state_bitmask=consumer_state_bitmask,
                )
                n_block -= 1

                ### Mainloop ###
                for _ in cutlass.range(num_n_block_groups - 1, unroll=1):
                    for stage in cutlass.range_constexpr(self.num_stages_S):
                        (
                            consumer_state_S,
                            producer_state_P,
                            producer_state_sm_stats,
                            consumer_state_bitmask,
                        ) = softmax_step_fn(
                            consumer_state_S,
                            producer_state_P,
                            producer_state_sm_stats,
                            1 - stage,
                            n_block,
                            mask_fn=partial(mask_fn, mask_seqlen=True),
                            consumer_state_bitmask=consumer_state_bitmask,
                        )
                        n_block -= 1

                ### last iteration if even ###
                # always mask to simplify logic
                if even_n_blocks:
                    (
                        consumer_state_S,
                        producer_state_P,
                        producer_state_sm_stats,
                        consumer_state_bitmask,
                    ) = softmax_step_fn(
                        consumer_state_S,
                        producer_state_P,
                        producer_state_sm_stats,
                        1,
                        n_block,
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                        consumer_state_bitmask=consumer_state_bitmask,
                    )
                    n_block -= 1

                # write row max and sum to smem
                sRowSum[tidx % self.cta_tile_m, warp_idx // self.num_acc_halves] = softmax.row_sum[
                    0
                ]
                if const_expr(mLSE is not None or self.has_learnable_sink):
                    if tidx < self.cta_tile_m:
                        sRowMax[tidx, 0] = softmax.row_max[0]
                self.sm_stats_barrier_full.arrive()

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
            if has_work:
                # Paired with the epilogue's arrive inside ITS has_work branch: an empty
                # split must skip both halves of this 256-thread barrier.
                self.sm_stats_barrier_empty.arrive_and_wait()

        pipeline_P.producer_tail(producer_state_P)
        pipeline_sm_stats.producer_tail(producer_state_sm_stats)
