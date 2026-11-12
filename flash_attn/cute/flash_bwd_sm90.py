import math
from typing import Callable, Optional, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute import FastDivmodDivisorV2
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils
from quack.sm90_utils import gemm_zero_init, gemm_w_idx

from cutlass.utils import TensorMapManager, TensorMapUpdateMode
from flash_attn.cute.config import Sm90BwdRegisters
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import pipeline
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTBwdScheduler,
    SingleTileVarlenScheduler,
)
from flash_attn.cute import barrier
from flash_attn.cute.named_barrier import NamedBarrierBwd
from flash_attn.cute.softmax import apply_score_mod_inner, apply_score_mod_bwd_inner
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.utils import AuxData
from flash_attn.cute.block_sparse_utils import (
    get_total_q_block_count_bwd,
    produce_block_sparse_q_loads_bwd_sm90,
    consume_block_sparse_mma_bwd_sm90,
    dQaccum_store_block_sparse_bwd_sm90,
)


class _AlternatingOperandState:
    """One token per semantic operand; both index and phase flip each tile.

    Both operands share ONE two-slot pipeline. A(t+1) occupies B(t)'s retired
    slot, and B(t+1) occupies A(t)'s retired slot. Flipping the index alone would
    wait on the previous full generation and permit an early overwrite.
    """

    def __init__(self, phase_index: Int32):
        self._phase_index = phase_index

    @property
    def index(self) -> Int32:
        return self._phase_index & 1

    @property
    def phase(self) -> Int32:
        return (self._phase_index >> 1) & 1

    def advance(self):
        self._phase_index ^= 3

    def __extract_mlir_values__(self):
        return [self._phase_index.ir_value()]

    def __new_from_mlir_values__(self, values):
        return _AlternatingOperandState(Int32(values[0]))


class FlashAttentionBackwardSm90:
    arch = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        deterministic: bool = False,
        tile_m: int = 64,
        tile_n: int = 128,
        Q_stage: int = 2,
        dO_stage: int = 2,
        PdS_stage: int = 2,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 2,
        AtomLayoutMdQ: int = 1,
        num_threads: int = 384,
        V_in_regs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        q_subtile_factor: cutlass.Constexpr[int] = 1,
        dQ_single_wg: bool = False,
        dQ_accum_lane_contiguous: bool = False,
        split_mode: Optional[str] = None,
        *,
        spt: bool,
        registers: Sm90BwdRegisters,
    ):
        self.dtype = dtype
        # Swapped dK/dV accumulators put head_dim on WGMMA's 64-row M axis.
        hdim_multiple_of = 64 if dKV_swapAB else 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.deterministic = deterministic
        # BwdConfig.spt: deterministic dQ semaphores in shortest-processing-time order.
        self.spt = spt
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.Q_stage = Q_stage
        self.dO_stage = dO_stage
        self.PdS_stage = PdS_stage
        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB
        self.AtomLayoutMSdP = AtomLayoutMSdP
        self.AtomLayoutNdKV = AtomLayoutNdKV
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.num_wg_mma = (self.num_threads // 128) - 1
        assert split_mode in (None, "dq", "dkv")
        # split_mode "dq": M-stationary kernel that owns the dQ tile in registers and
        # writes final dQ once per (m_block, head) — no global reduction stream.
        # split_mode "dkv": the fused kernel with the dQ GEMM and atomic loop compiled out.
        self.alias_stream_slots = split_mode is not None
        self.dq_owner = split_mode == "dq"
        self.fused_dq = split_mode is None
        self.compute_dkv = split_mode != "dq"
        if split_mode is not None:
            assert (head_dim, head_dim_v) == (512, 512)
            assert qhead_per_kvhead in (4, 8, 16)
            assert is_causal and not is_local and not deterministic
            assert score_mod is None and score_mod_bwd is None and mask_mod is None
            assert not has_aux_tensors
            assert not V_in_regs
        if self.dq_owner:
            assert not dQ_single_wg and not dQ_swapAB and not SdP_swapAB
            assert (tile_m, tile_n) == (64, 32)
            assert (Q_stage, dO_stage) == (1, 1)
            assert num_threads == 384
            # Double-buffer sdS so one PdS rendezvous per n-iteration suffices: the
            # WAR on buffer (n % 2) is covered by the barrier of iteration n-1 plus the
            # wait_group(0) that retires the dQ GEMM of n-2 before its buffer is reused.
            self.PdS_stage = 2
        assert self.dO_stage in [1, self.Q_stage]
        assert self.PdS_stage in [1, self.Q_stage] or self.dq_owner
        self.mma_dkv_is_rs = (
            AtomLayoutMSdP == 1
            and AtomLayoutNdKV == self.num_wg_mma
            and SdP_swapAB
            and not dKV_swapAB
        )
        self.V_in_regs = V_in_regs
        # May be overridden in __call__ for varlen inputs.
        if qhead_per_kvhead > 1:
            assert self.same_hdim_kv, "GQA backward requires head_dim == head_dim_v"
            assert self.num_wg_mma == 2, "GQA backward assumes 2 warp groups"
        # These are tuned for speed
        # Do we keep the LSE and dPsum in each thread, or split them across 8 threads that share
        # them and then shuffle to get the value whenever we need? This can reduce register
        # pressure when SdP_swapAB, where each thread needs to keep statistics for (kBlockM / 4)
        # rows. If !SdP_swapAB, each thread only needs to keep statistics for 2 rows.
        self.shuffle_LSE = self.SdP_swapAB and self.tile_hdim <= 64
        self.shuffle_dPsum = self.SdP_swapAB and self.tile_hdim <= 64

        self.buffer_align_bytes = 1024

        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd
        self.mask_mod = mask_mod
        self.has_aux_tensors = has_aux_tensors
        self.q_subtile_factor = q_subtile_factor
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 4
        self.qk_acc_dtype = Float32
        # dQ_single_wg: WG0 computes the full dQ GEMM, WG1 skips it.
        # Only valid for 2 MMA warp groups.
        # Credit: Ben Spector
        if dQ_single_wg:
            assert self.num_wg_mma == 2, "dQ_single_wg only supports 2 warp groups"
        self.num_wg_dQ = 1 if dQ_single_wg else self.num_wg_mma
        # BwdConfig.registers carries the split; config.validate_bwd_registers checks it.
        self.num_mma_regs_wg0 = registers.mma
        self.num_mma_regs_wg1 = registers.mma_no_dq
        self.num_producer_regs = registers.producer
        self.dQ_direct_atomic = dQ_accum_lane_contiguous and self.fused_dq
        if self.dQ_direct_atomic:
            assert (head_dim, head_dim_v) == (512, 512)
            assert qhead_per_kvhead in (4, 8, 16) and is_causal and not is_local
            assert not deterministic and not dQ_swapAB
            assert (tile_m, tile_n) == (64, 32)
            assert (Q_stage, dO_stage, PdS_stage) == (1, 1, 1)
            assert not SdP_swapAB and dKV_swapAB
            assert (AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ) == (1, 1, 1)
            assert num_threads == 384
            assert self.num_wg_dQ == self.num_wg_mma
            assert (self.tile_m * self.tile_hdim) % (128 * self.num_wg_dQ) == 0

        # One WG publishes complete P/dS tiles; both retain their gradient halves.
        # These are the nominal stages, not B2's physical operand slots or dQ's
        # two dS buffers. Keep register budgets from the explicit typed config.
        self.single_wg_sdp = (
            self.num_wg_mma == 2
            and not SdP_swapAB
            and not dQ_swapAB
            and not dQ_single_wg
            and (AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ) == (1, 1, 1)
            and (Q_stage, dO_stage, PdS_stage) == (1, 1, 1)
            and not deterministic
            and not V_in_regs
            and score_mod is None
            and score_mod_bwd is None
            and mask_mod is None
            and not has_aux_tensors
            and split_mode in ("dq", "dkv")
        )
        # The reviewed RS fragment budget assumes the canonical feature split.
        # Other explicit configs retain their existing shared-memory Q operand.
        self.Q_in_regs = split_mode == "dkv" and self.single_wg_sdp
        self.num_wg_mma_SdP = 1 if self.single_wg_sdp else self.num_wg_mma
        assert self.num_wg_mma_SdP % self.AtomLayoutMSdP == 0
        score_n_wg = self.num_wg_mma_SdP // self.AtomLayoutMSdP
        assert self.tile_n % score_n_wg == 0
        assert self.tile_m % self.AtomLayoutMSdP == 0
        score_wgmma_n = (
            self.tile_m // self.AtomLayoutMSdP if self.SdP_swapAB else self.tile_n // score_n_wg
        )
        assert score_wgmma_n % 16 == 0

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        Q_stage,
        num_threads,
        V_in_regs=False,
    ) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if (tile_m * 2) % num_threads != 0:
            return False
        return True

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mdO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric],
        mdPsum_type: Type[cutlass.Numeric],
        mdQaccum_type: Type[cutlass.Numeric],
        mdK_type: Type[cutlass.Numeric],
        mdV_type: Type[cutlass.Numeric],
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mdPsum_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if const_expr(self.dq_owner):
            if const_expr(mdQaccum_type != mQ_type):
                raise TypeError("The dQ owner writes dQ in the input data type")
        elif const_expr(self.fused_dq):
            if const_expr(mdQaccum_type not in [Float32]):
                raise TypeError("dQaccum tensor must be Float32")
        else:
            assert mdQaccum_type is None, "The dKV-only kernel does not consume dQ"
        if const_expr(self.qhead_per_kvhead == 1):
            if const_expr(not (mdK_type == mdV_type == mQ_type)):
                raise TypeError("mdK and mdV tensors must have the same data type as mQ")
        elif const_expr(self.compute_dkv):
            if const_expr(not (mdK_type == mdV_type == Float32)):
                raise TypeError("mdKaccum and mdVaccum tensors must have the data type Float32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        # We need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
        # Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
        # The M dimension (tile_m) doesn't matter for the layout, only the K dimension
        wg_d_dKV = self.num_wg_mma // self.AtomLayoutNdKV
        self.sQ_layout, self.sdO_layout = [
            # Need to set major_mode_size (mms) to accommodate Q and Q.T
            sm90_utils.make_smem_layout(
                self.dtype,
                LayoutEnum.ROW_MAJOR,
                shape,
                stage,
                major_mode_size=mms,
            )
            for shape, stage, mms in [
                (
                    (self.tile_m, self.tile_hdim),
                    2 if self.alias_stream_slots and not self.dq_owner else self.Q_stage,
                    self.tile_hdim // wg_d_dKV,
                ),
                (
                    (self.tile_m, self.tile_hdimv),
                    2 if self.alias_stream_slots and not self.dq_owner else self.dO_stage,
                    self.tile_hdim // wg_d_dKV,
                ),
            ]
        ]
        wg_d_dQ = self.num_wg_dQ // self.AtomLayoutMdQ
        # Accomodate both K and K.T
        self.sK_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdim),
            stage=2 if self.alias_stream_slots and self.dq_owner else 1 if self.dq_owner else None,
            major_mode_size=self.tile_hdim // wg_d_dQ,
        )
        # There's only V, no V.T, so layout is normal
        self.sV_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdimv),
            2 if self.alias_stream_slots and self.dq_owner else 1 if self.dq_owner else None,
        )
        # Accomodate both S and S.T
        wg_n_SdP = self.num_wg_mma_SdP // self.AtomLayoutMSdP
        wg_n_dKV = self.AtomLayoutNdKV
        self.sPdS_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n),
            stage=self.PdS_stage,
            major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV),
        )
        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ)
        )
        # dQ accumulator partitioning
        if const_expr(self.dQ_direct_atomic):
            # Map each warp's scalar atomics to adjacent FP32 addresses.
            num_thr_dQ = self.num_threads_per_warp_group * self.num_wg_dQ
            self.r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=32),
                cute.make_layout((num_thr_dQ, 1)),
                cute.make_layout((1, 128 // Float32.width)),
            )
        else:
            self.r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
                cute.make_layout((self.num_threads_per_warp_group, self.num_wg_dQ)),
                cute.make_layout(128 // Float32.width),
            )
        # dKVaccum for GQA epilogue - reuses sV+sK memory recast as f32
        # TODO: assert that sVaccum and sKaccum don't overflow smem

    def make_operand_state(self, user, stages, *, second=False):
        if self.alias_stream_slots:
            producer_phase = 1 if user is cutlass.pipeline.PipelineUserType.Producer else 0
            return _AlternatingOperandState(Int32(2 * producer_phase + int(second)))
        return cutlass.pipeline.make_pipeline_state(user, stages)

    def resident_producer_copy_fn(self, copy, pipe):
        if not self.alias_stream_slots:
            return copy_utils.tma_producer_copy_fn(copy, pipe)

        def copy_fn(src_idx, producer_state, **kwargs):
            copy(
                src_idx=src_idx,
                dst_idx=0,
                tma_bar_ptr=pipe.producer_get_barrier(producer_state),
                **kwargs,
            )

        return copy_fn

    def _get_tiled_mma(self):
        maybe_swap_mn = lambda shape, swap: (shape[1], shape[0], *shape[2:]) if swap else shape
        # S = Q @ K.T, dP = dO @ V.T
        atom_layout_SdP = (self.AtomLayoutMSdP, self.num_wg_mma_SdP // self.AtomLayoutMSdP, 1)
        tiler_mn_SdP = (self.tile_m // atom_layout_SdP[0], self.tile_n // atom_layout_SdP[1])
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=maybe_swap_mn(atom_layout_SdP, self.SdP_swapAB),
            tiler_mn=(64, tiler_mn_SdP[1] if not self.SdP_swapAB else tiler_mn_SdP[0]),
        )
        # dV = P.T @ dO, dK = dS.T @ Q
        atom_layout_dKV = (self.AtomLayoutNdKV, self.num_wg_mma // self.AtomLayoutNdKV, 1)
        tiler_mn_dK = (self.tile_n // atom_layout_dKV[0], self.tile_hdim // atom_layout_dKV[1])
        tiler_mn_dV = (self.tile_n // atom_layout_dKV[0], self.tile_hdimv // atom_layout_dKV[1])
        tiled_mma_dK, tiled_mma_dV = [
            sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.MN
                if not self.mma_dkv_is_rs
                else warpgroup.OperandMajorMode.K,
                warpgroup.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=maybe_swap_mn(atom_layout_dKV, self.dKV_swapAB),
                tiler_mn=(64, tiler_mn_d[1] if not self.dKV_swapAB else tiler_mn_d[0]),
                a_source=warpgroup.OperandSource.RMEM
                if self.mma_dkv_is_rs
                else warpgroup.OperandSource.SMEM,
            )
            for tiler_mn_d in (tiler_mn_dK, tiler_mn_dV)
        ]
        if self.Q_in_regs:
            assert self.dKV_swapAB and not self.mma_dkv_is_rs
            tiled_mma_dK = sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.K,
                warpgroup.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=maybe_swap_mn(atom_layout_dKV, True),
                tiler_mn=(64, tiler_mn_dK[0]),
                a_source=warpgroup.OperandSource.RMEM,
            )
        # dQ = dS @ K
        assert self.num_wg_dQ % self.AtomLayoutMdQ == 0
        atom_layout_dQ = (self.AtomLayoutMdQ, self.num_wg_dQ // self.AtomLayoutMdQ, 1)
        tiler_mn_dQ = (self.tile_m // atom_layout_dQ[0], self.tile_hdim // atom_layout_dQ[1])
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K if not self.dQ_swapAB else warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN if not self.dQ_swapAB else warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=maybe_swap_mn(atom_layout_dQ, self.dQ_swapAB),
            tiler_mn=(64, tiler_mn_dQ[1] if not self.dQ_swapAB else tiler_mn_dQ[0]),
        )
        return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct = [
            cute.struct.Align[cute.struct.MemRange[t, cute.cosize(layout)], self.buffer_align_bytes]
            for (layout, t) in [
                (self.sQ_layout, self.dtype),
                (self.sK_layout, self.dtype),
                (self.sV_layout, self.dtype),
                (self.sdO_layout, self.dtype),
            ]
        ]
        cosize_sdQaccum = (
            0
            if const_expr(not self.fused_dq or self.dQ_direct_atomic)
            else cute.cosize(self.sdQaccum_layout)
        )
        sdQaccum_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cosize_sdQaccum],
            self.buffer_align_bytes,
        ]

        cosize_sdS = cute.cosize(self.sPdS_layout)
        cosize_sP = (
            cute.cosize(self.sPdS_layout)
            if const_expr(not self.mma_dkv_is_rs and self.compute_dkv)
            else 0
        )
        sLSE_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.Q_stage], 128
        ]
        sdPsum_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.round_up(self.tile_m, 64) * self.dO_stage], 128
        ]

        if self.alias_stream_slots:
            # Allocate each physical operand exactly once. Q/dO are adjacent,
            # as are V/K (V first preserves the GQA FP32 epilogue alias).
            sQ_one = cute.struct.Align[
                cute.struct.MemRange[self.dtype, self.tile_m * self.tile_hdim], 1024
            ]
            sKV_one = cute.struct.Align[
                cute.struct.MemRange[self.dtype, self.tile_n * self.tile_hdim], 1024
            ]

            @cute.struct
            class SharedStorageAliasedOperands:
                mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, 4]
                sLSE: sLSE_struct
                sdPsum: sdPsum_struct
                sQ: sQ_one
                sdO: sQ_one
                sV: sKV_one
                sK: sKV_one
                sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
                sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
                sdQaccum: sdQaccum_struct

            return SharedStorageAliasedOperands

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
            sdQaccum: sdQaccum_struct

        return SharedStorageQKV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: Optional[cute.Tensor],
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        tensormap_workspace: Optional[
            cute.Tensor
        ] = None,  # (num_ctas, 2, 16) Int64: K, V descriptors per CTA
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        if const_expr(not self.fused_dq):
            assert mSeqUsedQ is None and mSeqUsedK is None
            assert window_size_left is None and window_size_right is None
            assert mdQ_semaphore is None and mdK_semaphore is None and mdV_semaphore is None
            assert aux_data.tensors is None or len(aux_data.tensors) == 0
            assert aux_data.scalars is None or len(aux_data.scalars) == 0
            assert blocksparse_tensors is None
        # For GQA (qhead_per_kvhead > 1), multiple Q heads accumulate into the same dK/dV,
        # so we need the float32 accum path + postprocess.
        # For varlen_k with qhead_per_kvhead == 1, we use ragged TMA tensors.
        self.varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None

        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
            )
        )

        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        ]

        # Non-varlen inputs are (b, s, n, h), varlen inputs are (s, n, h).
        # We convert both to a seqlen-major view with head-dim second.
        # Each tensor may have different rank when Q is padded (seqused_q) but K/V are unpadded (cu_seqlens_k).
        def _qkv_transpose(t):
            return layout_utils.select(t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1])

        mQ, mK, mV, mdO = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO)]
        if const_expr(self.qhead_per_kvhead == 1):
            mdK, mdV = [_qkv_transpose(t) for t in (mdK, mdV)]
        else:
            # Accum tensors are (b, n, s*h) for non-varlen and (n, s*h) for varlen.
            accum_transpose = [2, 1, 0] if cute.rank(mdK.shape) == 3 else [1, 0]
            mdK, mdV = [layout_utils.select(t, accum_transpose) for t in (mdK, mdV)]
        # Non-varlen stats are (b, n, s), varlen stats are (n, s).
        stats_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        mLSE, mdPsum = [layout_utils.select(t, stats_transpose) for t in (mLSE, mdPsum)]
        if const_expr(self.dq_owner):
            # The split wrapper passes final dQ in this argument slot, not FP32 scratch.
            mdQaccum = _qkv_transpose(mdQaccum)
        elif const_expr(self.fused_dq):
            mdQaccum = layout_utils.select(mdQaccum, stats_transpose)

        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()
        # (batch, num_head, num_m_blocks, cluster_size) -> (num_m_blocks, cluster_size, num_head, batch)
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=[2, 3, 1, 0])
        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                layout_utils.select(t, mode=[2, 3, 1, 0]) for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self.num_mma_threads = max(
            tiled_mma_SdP.size, tiled_mma_dK.size, tiled_mma_dV.size, tiled_mma_dQ.size
        )
        assert self.num_mma_threads + 128 == self.num_threads

        self.num_threads_per_warp_group = 128
        self.num_producer_threads = 32

        self._setup_attributes()
        # Bound the K/V loads by the logical seqlen_k (see below); decided here because the
        # shared storage needs descriptor slots for it.
        # The split dQ loader retains the pin's original K/V load path. Only the
        # fused / split dKV loader below uses upstream's bounded descriptors.
        self.ragged_kv_tma = not self.dq_owner and (
            mCuSeqlensK is not None or mSeqUsedK is not None
        )
        if const_expr(self.ragged_kv_tma):
            assert tensormap_workspace is not None, (
                "ragged K/V TMA loads need a tensormap workspace"
            )
        SharedStorage = self._get_shared_storage_cls()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = (
            self.tile_m * self.tile_hdim * Float32.width // 8 // self.num_wg_dQ
        )
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.tile_hdim * Float32.width // 8
        self.tma_copy_bytes["dVacc"] = self.tile_n * self.tile_hdimv * Float32.width // 8

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        # Bound the K and V TMA loads by the logical seqlen_k (cu_seqlens_k and/or seqused_k) so
        # rows in [seqlen_k, tile_n) arrive zero-filled instead of as whatever the buffer holds.
        # The masked P / dS are exactly zero there, but 0 * NaN = NaN in dP = dO @ V.T and
        # dQ = dS @ K. The descriptors use the load-safe form of flash_attn.cute.copy_utils
        # (quack's ptr_shift form is store-only, and its wraparound form needs rank <= 3, i.e.
        # only the packed layout); it covers the packed (t, d, h) and the dense (s, d, h, b)
        # layout alike.
        # The load warp rewrites its CTA's copies of the K and V descriptors per work tile with
        # the sequence's row extent (tensormap.replace); a descriptor with a per-launch extent
        # cannot do this (TMA prefetches past the footprint it is given, #3068).
        mK_tma, mV_tma = mK, mV
        mK_gmem, mV_gmem = (mK, mV) if const_expr(self.ragged_kv_tma) else (None, None)
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK_tma,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV_tma,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdimv),
        )
        tma_atom_dQ = None
        if const_expr(self.dq_owner):
            mdQ_tma = (
                copy_utils.create_ragged_tensor_for_tma(mdQaccum, ragged_dim=0, ptr_shift=True)
                if self.is_varlen_q
                else mdQaccum
            )
            tma_atom_dQ, mdQaccum = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdQ_tma,
                cute.select(self.sQ_layout, mode=[0, 1]),
                (self.tile_m, self.tile_hdim),
            )
        if const_expr(self.qhead_per_kvhead == 1):
            mdK_tma = (
                copy_utils.create_ragged_tensor_for_tma(mdK, ragged_dim=0, ptr_shift=True)
                if self.varlen_k
                else mdK
            )
            mdV_tma = (
                copy_utils.create_ragged_tensor_for_tma(mdV, ragged_dim=0, ptr_shift=True)
                if self.varlen_k
                else mdV
            )
            tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdK_tma,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
            )
            tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdV_tma,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
            )
        else:
            tma_atom_dK = tma_atom_dV = tma_tensor_dK = tma_tensor_dV = None

        if const_expr(self.dq_owner):
            # M-stationary grid: one CTA per (m_block, q_head, batch), like the
            # preprocess/postprocess kernels. Schedule follows Q's cu_seqlens.
            if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
                TileScheduler = SingleTileVarlenScheduler
            else:
                TileScheduler = SingleTileScheduler
            tile_sched_args = TileSchedulerArguments(
                cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
                cute.size(mQ.shape[2]),
                cute.size(mQ.shape[3])
                if const_expr(mCuSeqlensQ is None)
                else cute.size(mCuSeqlensQ.shape[0] - 1),  # num_batch
                1,  # num_splits
                cute.size(mK.shape[0]),  # seqlen_k (informational)
                mQ.shape[1],  # headdim
                mV.shape[1],  # headdim_v
                total_q=cute.size(mQ.shape[0])
                if const_expr(mCuSeqlensQ is not None)
                else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
                tile_shape_mn=(self.tile_m, self.tile_n),
                mCuSeqlensQ=mCuSeqlensQ,
                mSeqUsedQ=mSeqUsedQ,
                qhead_per_kvhead_packgqa=1,
                element_size=self.dtype.width // 8,
                is_persistent=False,
                lpt=False,
                head_swizzle=False,
                cu_total_m_blocks_ptr=mCuTotalMBlocks,
            )
        else:
            if const_expr(mCuSeqlensK is not None or mSeqUsedK is not None):
                TileScheduler = SingleTileVarlenScheduler
            elif const_expr(self.deterministic):
                TileScheduler = SingleTileLPTBwdScheduler
            else:
                TileScheduler = SingleTileScheduler
            tile_sched_args = TileSchedulerArguments(
                cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
                cute.size(mQ.shape[2]),
                cute.size(mK.shape[3])
                if const_expr(mCuSeqlensK is None)
                else cute.size(mCuSeqlensK.shape[0] - 1),  # num_batch
                1,  # num_splits
                cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
                mQ.shape[1],  # headdim
                mV.shape[1],  # headdim_v
                total_q=cute.size(mK.shape[0])
                if const_expr(mCuSeqlensK is not None)
                else cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
                tile_shape_mn=(self.tile_n, self.tile_m),  # Swapping the role of Q & K
                mCuSeqlensQ=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedK,
                qhead_per_kvhead_packgqa=1,
                element_size=self.dtype.width // 8,
                is_persistent=False,
                lpt=self.spt,
                head_swizzle=self.deterministic,
                cu_total_m_blocks_ptr=mCuTotalMBlocks,
            )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
        else:
            softmax_scale_log2 = LOG2_E

        fastdiv_mods = None
        if const_expr(aux_data.tensors is not None):
            seqlen_q = cute.size(mQ.shape[0])
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmodDivisorV2(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisorV2(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        qhead_per_kvhead_divmod = None
        if const_expr(self.qhead_per_kvhead > 1):
            qhead_per_kvhead_divmod = FastDivmodDivisorV2(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            mK_gmem,
            mV_gmem,
            tensormap_workspace if const_expr(self.ragged_kv_tma) else None,
            tma_tensor_dO,
            tma_tensor_dK if const_expr(self.qhead_per_kvhead == 1) else mdK,
            tma_tensor_dV if const_expr(self.qhead_per_kvhead == 1) else mdV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            tma_atom_dQ,
            mLSE,
            mdPsum,
            mdQaccum,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,
            self.r2s_tiled_copy_dQaccum,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            aux_data,
            fastdiv_mods,
            blocksparse_tensors,
            qhead_per_kvhead_divmod,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            window_size_left,
            window_size_right,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mK_gmem: Optional[cute.Tensor],  # gmem K/V behind tma_atom_K/V (ragged K/V only)
        mV_gmem: Optional[cute.Tensor],
        tensormap_workspace: Optional[cute.Tensor],  # (num_ctas, 2, 16) Int64 (ragged K/V only)
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tma_atom_dQ: Optional[cute.CopyAtom],
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # prefetch TMA descriptors
        if warp_idx == 0:
            for atom in [
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_dK,
                tma_atom_dV,
                tma_atom_dQ,
            ]:
                if const_expr(atom is not None):
                    cpasync.prefetch_descriptor(atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
        )
        pipeline_Q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=2 if const_expr(self.alias_stream_slots) else self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["K"]
            if const_expr(self.dq_owner)
            else self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"],
            defer_sync=not self.alias_stream_slots,
        )
        if const_expr(self.alias_stream_slots):
            pipeline_dO = pipeline_Q
        else:
            pipeline_dO = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_dO.data_ptr(),
                num_stages=self.dO_stage,
                producer_group=pipeline_producer_group,
                consumer_group=pipeline_consumer_group,
                tx_count=self.tma_copy_bytes["V"]
                if const_expr(self.dq_owner)
                else self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"],
                defer_sync=False,
            )

        if const_expr(self.alias_stream_slots):
            # The stage axis is a PHYSICAL slot. Both semantic views share its
            # base; alternating state tokens select the right slot per tile.
            if const_expr(self.dq_owner):
                sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
                sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
                sK = storage.sV.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
                sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
            else:
                sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
                sdO = storage.sQ.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
                sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
                sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
            sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
            sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = None
        if const_expr(not self.mma_dkv_is_rs and self.compute_dkv):
            sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.Q_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.dO_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdQaccum = (
            storage.sdQaccum.get_tensor(sdQaccum_layout)
            if const_expr(self.fused_dq and not self.dQ_direct_atomic)
            else None
        )

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            # With ragged K/V TMA tensors mode 0 is a sentinel extent, but cu_seqlens_k or
            # seqused_k is then always present and the static length is unused.
            seqlen_k_static=mK.shape[0] if const_expr(not self.ragged_kv_tma) else Int32(0),
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            swap_AB=self.SdP_swapAB,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mK_gmem,
                    mV_gmem,
                    tensormap_workspace,
                    mdO,
                    mLSE,
                    mdPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    pipeline_Q,
                    pipeline_dO,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    blocksparse_tensors,
                    qhead_per_kvhead_divmod,
                )
            if const_expr(self.fused_dq and not self.dQ_direct_atomic):
                if warp_idx == 1:
                    self.dQaccum_store(
                        mdQaccum,
                        sdQaccum,
                        block_info,
                        TileSchedulerCls,
                        SeqlenInfoCls,
                        blocksparse_tensors,
                        mdQ_semaphore,
                    )
        else:
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            mma_args = (
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                mdK_semaphore,
                mdV_semaphore,
                mdQaccum,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sdQaccum,
                pipeline_Q,
                pipeline_dO,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                tma_atom_dQ,
                r2s_tiled_copy_dQaccum,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                aux_data,
                fastdiv_mods,
                blocksparse_tensors,
                qhead_per_kvhead_divmod,
            )
            if const_expr(self.single_wg_sdp):
                warp_idx_in_mma = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4
                if warp_idx_in_mma < 4:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                    self.mma(*mma_args, is_dQ_wg=self.dq_owner, is_sdp_wg=True)
                else:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
                    self.mma(*mma_args, is_dQ_wg=self.dq_owner, is_sdp_wg=False)
            elif const_expr(not self.fused_dq and not self.dq_owner):
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                self.mma(*mma_args, is_dQ_wg=False)
            elif const_expr(self.num_wg_dQ == self.num_wg_mma):
                # Both WGs compute dQ
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                self.mma(*mma_args, is_dQ_wg=True)
            else:
                # WG0 computes dQ, WG1 skips it
                warp_idx_in_mma = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4
                if warp_idx_in_mma < 4:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                    self.mma(*mma_args, is_dQ_wg=True)
                else:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
                    self.mma(*mma_args, is_dQ_wg=False)

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mK_gmem: Optional[cute.Tensor],
        mV_gmem: Optional[cute.Tensor],
        tensormap_workspace: Optional[cute.Tensor],
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        if const_expr(self.dq_owner):
            if warp_idx_in_wg == 0:
                self.load_dq_owner(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    pipeline_Q,
                    pipeline_dO,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    qhead_per_kvhead_divmod,
                )
            return

        if warp_idx_in_wg == 0:
            producer_state_Q = self.make_operand_state(
                cutlass.pipeline.PipelineUserType.Producer, self.Q_stage, second=False
            )
            producer_state_dO = self.make_operand_state(
                cutlass.pipeline.PipelineUserType.Producer, self.dO_stage, second=True
            )
            tensormap_manager = None
            tensormap_gmem = ()
            desc_ptrs = (None, None)
            if const_expr(self.ragged_kv_tma):
                tensormap_manager = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
                bidx, bidy, bidz = cute.arch.block_idx()
                gdx, gdy, _ = cute.arch.grid_dim()
                cta_linear_idx = bidx + bidy * gdx + bidz * gdx * gdy
                if cta_linear_idx >= cute.size(tensormap_workspace, mode=[0]):
                    # An undersized workspace would silently corrupt memory: fail loudly instead.
                    cute.arch.inline_ptx("trap;", read_only_args=[])
                tensormap_gmem = tuple(
                    tensormap_manager.get_tensormap_ptr(
                        tensormap_workspace[(cta_linear_idx, i, None)].iterator
                    )
                    for i in range(2)
                )
                desc_ptrs = tuple(
                    tensormap_manager.get_tensormap_ptr(p, cute.AddressSpace.generic)
                    for p in tensormap_gmem
                )
                for atom, gmem_ptr in zip((tma_atom_K, tma_atom_V), tensormap_gmem):
                    tensormap_manager.init_tensormap_from_atom(atom, gmem_ptr, 0)
                tensormap_manager.fence_tensormap_initialization()
            # The K/V extent depends on the batch only: rewrite the descriptors when it changes.
            kv_desc_batch_idx = Int32(-1)
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                n_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                if const_expr(self.ragged_kv_tma) and batch_idx != kv_desc_batch_idx:
                    # Shrink the rows extent (mode 0) of K and V to this sequence's end: packed
                    # layouts count absolute rows, dense layouts per-batch rows. Extents >= 1.
                    kv_desc_batch_idx = batch_idx
                    kv_rows_end = (
                        seqlen.offset_k + seqlen.seqlen_k
                        if const_expr(seqlen.has_cu_seqlens_k)
                        else seqlen.seqlen_k
                    )
                    kv_rows_end = cutlass.max(kv_rows_end, 1)
                    bounded = []
                    for t in (mK_gmem, mV_gmem):
                        shape = list(t.shape)
                        shape[0] = kv_rows_end
                        bounded.append(
                            cute.make_tensor(
                                t.iterator, cute.make_layout(tuple(shape), stride=t.stride)
                            )
                        )
                    tensormap_manager.update_tensormap(
                        tuple(bounded), (tma_atom_K, tma_atom_V), tensormap_gmem, 0, (None, None)
                    )
                    for p in tensormap_gmem:
                        tensormap_manager.fence_tensormap_update(p)
                        cute.arch.inline_ptx(
                            "prefetch.tensormap [{$r0}];", read_only_args=[p.toint()]
                        )
                head_idx_kv = (
                    head_idx
                    if const_expr(self.qhead_per_kvhead == 1)
                    else head_idx // qhead_per_kvhead_divmod
                )
                mK_cur, mV_cur = [
                    seqlen.offset_batch_K(t, batch_idx, dim=3)[None, None, head_idx_kv]
                    for t in (mK, mV)
                ]
                kv_block = n_block
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (kv_block, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (kv_block, 0))

                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[
                    None, head_idx
                ]
                mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]
                mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[
                    None, head_idx
                ]
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
                gdO = cute.local_tile(mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0))
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))

                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    gK,
                    sK,
                    single_stage=True,
                    **({"tma_desc_ptr": desc_ptrs[0]} if const_expr(self.ragged_kv_tma) else {}),
                )
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V,
                    0,
                    cute.make_layout(1),
                    gV,
                    sV,
                    single_stage=True,
                    **({"tma_desc_ptr": desc_ptrs[1]} if const_expr(self.ragged_kv_tma) else {}),
                )
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ
                )
                load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
                load_dO, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO, 0, cute.make_layout(1), gdO, sdO
                )
                load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
                load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
                load_LSE = self.resident_producer_copy_fn(load_LSE, pipeline_Q)
                load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)
                load_dPsum = self.resident_producer_copy_fn(load_dPsum, pipeline_dO)

                m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

                if const_expr(not self.use_block_sparsity):
                    total_m_block_cnt = m_block_max - m_block_min
                    process_tile = (
                        const_expr(not self.is_local and not self.is_varlen_q)
                        or m_block_min < m_block_max
                    )
                else:
                    total_m_block_cnt = get_total_q_block_count_bwd(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                    )
                    process_tile = total_m_block_cnt > Int32(0)

                if process_tile:
                    if const_expr(not self.use_block_sparsity):
                        first_m_block = m_block_min
                        pipeline_Q.producer_acquire(
                            producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"]
                        )
                        load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q))
                        load_Q(first_m_block, producer_state=producer_state_Q)
                        # Wait for bwd preprocess to finish writing LSE and dPsum
                        cute.arch.griddepcontrol_wait()
                        load_LSE(first_m_block, producer_state=producer_state_Q)
                        producer_state_dO_cur = (
                            producer_state_dO
                            if const_expr(self.alias_stream_slots or self.Q_stage != self.dO_stage)
                            else producer_state_Q
                        )
                        pipeline_dO.producer_acquire(
                            producer_state_dO_cur, extra_tx_count=self.tma_copy_bytes["V"]
                        )
                        load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_cur))
                        load_dO(first_m_block, producer_state=producer_state_dO_cur)
                        load_dPsum(first_m_block, producer_state=producer_state_dO_cur)
                        producer_state_Q.advance()
                        producer_state_dO.advance()

                        for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                            pipeline_Q.producer_acquire(producer_state_Q)
                            load_Q(m_block, producer_state=producer_state_Q)
                            load_LSE(m_block, producer_state=producer_state_Q)
                            producer_state_dO_cur = (
                                producer_state_dO
                                if const_expr(
                                    self.alias_stream_slots or self.Q_stage != self.dO_stage
                                )
                                else producer_state_Q
                            )
                            pipeline_dO.producer_acquire(producer_state_dO_cur)
                            load_dO(m_block, producer_state=producer_state_dO_cur)
                            load_dPsum(m_block, producer_state=producer_state_dO_cur)
                            producer_state_Q.advance()
                            producer_state_dO.advance()
                    else:
                        # The wait for bwd preprocess that the dense arm above does inline sits
                        # in _load_q_do_block_sm90, at the same point in the load order.
                        producer_state_Q, producer_state_dO = produce_block_sparse_q_loads_bwd_sm90(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            n_block,
                            producer_state_Q,
                            producer_state_dO,
                            pipeline_Q,
                            pipeline_dO,
                            load_K,
                            load_V,
                            load_Q,
                            load_dO,
                            load_LSE,
                            load_dPsum,
                            self.tma_copy_bytes["K"],
                            self.tma_copy_bytes["V"],
                            Q_stage_eq_dO_stage=(self.Q_stage == self.dO_stage),
                            q_subtile_factor=self.q_subtile_factor,
                            m_block_max=m_block_max,
                        )

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def load_dq_owner(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        qhead_per_kvhead_divmod: FastDivmodDivisorV2,
    ):
        # M-stationary producer: Q/dO/LSE/dPsum load once per work tile, riding the
        # first K/V stage barriers via extra_tx_count; K/V n-tiles stream through the
        # single-stage pipelines (pipeline_Q carries K, pipeline_dO carries V).
        producer_state_K = self.make_operand_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage, second=False
        )
        producer_state_V = self.make_operand_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage, second=True
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            head_idx_kv = head_idx // qhead_per_kvhead_divmod
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            if n_block_min < n_block_max:
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
                load_K = copy_utils.tma_producer_copy_fn(
                    copy_utils.tma_get_copy_fn(tma_atom_K, 0, cute.make_layout(1), gK, sK)[0],
                    pipeline_Q,
                )
                load_V = copy_utils.tma_producer_copy_fn(
                    copy_utils.tma_get_copy_fn(tma_atom_V, 0, cute.make_layout(1), gV, sV)[0],
                    pipeline_dO,
                )

                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[
                    None, head_idx
                ]
                mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[
                    None, head_idx
                ]
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
                gdO = cute.local_tile(mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0))
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
                load_Q = self.resident_producer_copy_fn(
                    copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ)[0],
                    pipeline_Q,
                )
                load_dO = self.resident_producer_copy_fn(
                    copy_utils.tma_get_copy_fn(tma_atom_dO, 0, cute.make_layout(1), gdO, sdO)[0],
                    pipeline_dO,
                )
                load_LSE = self.resident_producer_copy_fn(
                    copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE), pipeline_Q
                )
                load_dPsum = self.resident_producer_copy_fn(
                    copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum), pipeline_dO
                )

                cute.arch.griddepcontrol_wait()
                pipeline_Q.producer_acquire(
                    producer_state_K,
                    extra_tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"],
                )
                load_K(n_block_min, producer_state=producer_state_K)
                load_Q(m_block, producer_state=producer_state_K)
                load_LSE(m_block, producer_state=producer_state_K)
                producer_state_V_cur = (
                    producer_state_V
                    if const_expr(self.alias_stream_slots or self.Q_stage != self.dO_stage)
                    else producer_state_K
                )
                pipeline_dO.producer_acquire(
                    producer_state_V_cur,
                    extra_tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"],
                )
                load_V(n_block_min, producer_state=producer_state_V_cur)
                load_dO(m_block, producer_state=producer_state_V_cur)
                load_dPsum(m_block, producer_state=producer_state_V_cur)
                producer_state_K.advance()
                producer_state_V.advance()

                for n_block in cutlass.range(n_block_min + 1, n_block_max, unroll=1):
                    pipeline_Q.producer_acquire(producer_state_K)
                    load_K(n_block, producer_state=producer_state_K)
                    producer_state_V_cur = (
                        producer_state_V
                        if const_expr(self.alias_stream_slots or self.Q_stage != self.dO_stage)
                        else producer_state_K
                    )
                    pipeline_dO.producer_acquire(producer_state_V_cur)
                    load_V(n_block, producer_state=producer_state_V_cur)
                    producer_state_K.advance()
                    producer_state_V.advance()

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def apply_score_mod(
        self,
        acc_S: cute.Tensor,
        thr_mma_SdP: cute.ThrMma,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ):
        # [NOTE] SdP_swapAB: swapAB transposes the tile, so use (n, m) indexing
        cS = cute.make_identity_tensor(
            (self.tile_n, self.tile_m) if self.SdP_swapAB else (self.tile_m, self.tile_n)
        )
        cS = cute.domain_offset(
            (n_block * self.tile_n, m_block * self.tile_m)
            if self.SdP_swapAB
            else (m_block * self.tile_m, n_block * self.tile_n),
            cS,
        )
        tScS = thr_mma_SdP.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead,
            transpose_indices=self.SdP_swapAB,
        )

    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad_tensor: cute.Tensor,
        score_tensor: cute.Tensor,
        thr_mma_SdP: cute.ThrMma,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ):
        cS = cute.make_identity_tensor(
            (self.tile_n, self.tile_m) if self.SdP_swapAB else (self.tile_m, self.tile_n)
        )
        cS = cute.domain_offset(
            (n_block * self.tile_n, m_block * self.tile_m)
            if self.SdP_swapAB
            else (m_block * self.tile_m, n_block * self.tile_n),
            cS,
        )
        tScS = thr_mma_SdP.partition_C(cS)

        apply_score_mod_bwd_inner(
            grad_tensor,
            score_tensor,
            tScS,
            self.score_mod_bwd,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead,
            transpose_indices=self.SdP_swapAB,
        )

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tma_atom_dQ: Optional[cute.CopyAtom],
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
        is_sdp_wg: cutlass.Constexpr[bool] = True,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        num_wg_mma_SdP = tiled_mma_SdP.size // self.num_threads_per_warp_group
        sdp_tidx = tidx % tiled_mma_SdP.size
        thr_mma_SdP = tiled_mma_SdP.get_slice(sdp_tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(
            (warp_group_idx % num_wg_mma_SdP) * self.num_threads_per_warp_group
        )
        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dQ = None
        if const_expr(is_dQ_wg):
            wg_idx_dQ = warp_group_idx if const_expr(self.num_wg_dQ > 1) else 0
            wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(wg_idx_dQ))
        # S = Q @ K.T
        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_S, sQ, sK, swap_AB=self.SdP_swapAB
        )
        mma_qk_fn = partial(
            gemm_zero_init, tiled_mma_SdP, shape_mnk_S[:2], tSrQ, tSrK, swap_AB=self.SdP_swapAB
        )
        # dP = dO @ V.T
        shape_mnk_dP = (self.tile_m, self.tile_n, self.tile_hdimv)
        _, tdPrdO, tdPrV = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_dP, sdO, sV, swap_AB=self.SdP_swapAB
        )
        mma_dov_fn = partial(
            gemm_zero_init, tiled_mma_SdP, shape_mnk_dP[:2], tdPrdO, tdPrV, swap_AB=self.SdP_swapAB
        )
        # dV += P.T @ dO
        sPt = layout_utils.transpose_view(sP) if sP is not None else None
        sdOt = layout_utils.transpose_view(sdO)
        acc_dV = None
        acc_dK = None
        mma_pdo_fn = None
        mma_dsq_fn = None
        copy_Q_s2r = None
        sdSt = layout_utils.transpose_view(sdS)
        if const_expr(self.compute_dkv):
            shape_mnk_dV = (self.tile_n, self.tile_hdimv, self.tile_m)
            acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(
                wg_mma_dV, shape_mnk_dV, sPt, sdOt, swap_AB=self.dKV_swapAB
            )
            if const_expr(not self.mma_dkv_is_rs):
                mma_pdo_fn = partial(
                    gemm_w_idx, tiled_mma_dV, acc_dV, tdVrPt, tdVrdOt, swap_AB=self.dKV_swapAB
                )
            else:
                mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)
            # dK += dS.T @ Q
            sQt = layout_utils.transpose_view(sQ)
            shape_mnk_dK = (self.tile_n, self.tile_hdim, self.tile_m)
            acc_dK, tdKrdSt, tdKrQt = sm90_utils.partition_fragment_ABC(
                wg_mma_dK, shape_mnk_dK, sdSt, sQt, swap_AB=self.dKV_swapAB
            )
            if const_expr(self.Q_in_regs):
                # Derive feature ownership and register order from the dK MMA.
                # Logical Q.T is MN-major in shared memory, physical A in registers.
                q_copy = cute.make_tiled_copy_A(
                    copy_utils.get_smem_load_atom(self.dtype, transpose=True), tiled_mma_dK
                )
                q_thr_copy = q_copy.get_slice(tidx)
                q_src = q_thr_copy.partition_S(sQt)
                q_dst = q_thr_copy.retile(tdKrQt)

                def copy_Q_s2r(stage):
                    cute.copy(q_copy, q_src[None, None, None, stage], q_dst)

                mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tdKrQt, tdKrdSt)
            elif const_expr(not self.mma_dkv_is_rs):
                mma_dsq_fn = partial(
                    gemm_w_idx, tiled_mma_dK, acc_dK, tdKrdSt, tdKrQt, swap_AB=self.dKV_swapAB
                )
            else:
                mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)
        # dQ = dS @ K
        sKt = layout_utils.transpose_view(sK)
        shape_mnk_dQ = (self.tile_m, self.tile_hdim, self.tile_n)
        mma_dsk_fn = None
        mma_dq_acc_fn = None
        acc_dQ = None
        if const_expr(is_dQ_wg and self.fused_dq):
            _, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(
                wg_mma_dQ, shape_mnk_dQ, sdS, sKt, swap_AB=self.dQ_swapAB
            )
            mma_dsk_fn = partial(
                gemm_zero_init,
                tiled_mma_dQ,
                shape_mnk_dQ[:2],
                tdQrdS,
                tdQrKt,
                swap_AB=self.dQ_swapAB,
            )
        if const_expr(self.dq_owner):
            # Persistent per-WG dQ accumulator (m64 x 256 FP32), accumulated over the
            # n sweep via zero_init on the first iteration — the dK/dV idiom.
            acc_dQ, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(
                wg_mma_dQ, shape_mnk_dQ, sdS, sKt, swap_AB=self.dQ_swapAB
            )
            mma_dq_acc_fn = partial(
                gemm_w_idx, tiled_mma_dQ, acc_dQ, tdQrdS, tdQrKt, swap_AB=self.dQ_swapAB
            )

        # Smem copy atom tiling for P/dS R2S
        copy_P_r2s = None
        mms_PdS = self.tile_n // (num_wg_mma_SdP // self.AtomLayoutMSdP)
        if const_expr(sP is not None):
            sP_cpy = sP if const_expr(not self.SdP_swapAB) else sPt
            copy_P_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_SdP,
                sP_cpy,
                sdp_tidx,
                transpose=self.SdP_swapAB,
                position_independent=True,
                major_mode_size=mms_PdS,
            )
        sdS_cpy = sdS if const_expr(not self.SdP_swapAB) else sdSt
        copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            sdS_cpy,
            sdp_tidx,
            transpose=self.SdP_swapAB,
            position_independent=True,
            major_mode_size=mms_PdS,
        )

        tLSEsLSE = layout_utils.mma_partition_C_vec(
            sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        tLSEsdPsum = layout_utils.mma_partition_C_vec(
            sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        # When shuffle=True, rows are distributed across 8 quads (4 threads each) within a warp.
        # Each thread loads only ceil(num_rows/8) values;
        shfl_copy = copy_utils.tiled_copy_1d(sLSE.element_type, num_threads=8, num_copy_elems=2)
        if const_expr(self.shuffle_LSE):
            tLSEsLSE = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(tLSEsLSE)
            # ((2, 1), 1, 2) -> (((2, 1), 1), 2)
            tLSEsLSE = cute.group_modes(tLSEsLSE, 0, 2)
        if const_expr(self.shuffle_dPsum):
            tLSEsdPsum = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(tLSEsdPsum)
            tLSEsdPsum = cute.group_modes(tLSEsdPsum, 0, 2)

        tdQsdQaccum = None
        dQaccum_thr_copy = None
        if const_expr(is_dQ_wg):
            dQaccum_thr_copy = r2s_tiled_copy_dQaccum.get_slice(tidx)
            if const_expr(self.fused_dq and not self.dQ_direct_atomic):
                tdQsdQaccum = dQaccum_thr_copy.partition_D(sdQaccum)

        PdS_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.PdS), num_threads=self.num_mma_threads
        )

        mma_one_m_block_all = partial(
            self.mma_one_m_block,
            warp_group_idx=warp_group_idx,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            copy_Q_s2r=copy_Q_s2r,
            mma_dsk_fn=mma_dsk_fn,
            copy_P_r2s=copy_P_r2s,
            copy_dS_r2s=copy_dS_r2s,
            pipeline_Q=pipeline_Q,
            pipeline_dO=pipeline_dO,
            tLSEsLSE=tLSEsLSE,
            tLSEsdPsum=tLSEsdPsum,
            tdQsdQaccum=tdQsdQaccum,
            dQaccum_thr_copy=dQaccum_thr_copy,
            softmax_scale_log2=softmax_scale_log2,
            PdS_barrier=PdS_barrier,
            # acc_dV=acc_dV,
            # acc_dK=acc_dK,
            is_dQ_wg=is_dQ_wg,
            is_sdp_wg=is_sdp_wg,
        )

        consumer_state_Q = self.make_operand_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage, second=False
        )
        consumer_state_dO = self.make_operand_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage, second=True
        )
        if const_expr(self.dq_owner):
            self.mma_dq_owner(
                acc_dQ,
                mma_qk_fn,
                mma_dov_fn,
                mma_dq_acc_fn,
                copy_dS_r2s,
                pipeline_Q,
                pipeline_dO,
                tLSEsLSE,
                tLSEsdPsum,
                mdQaccum,
                sQ[None, None, 0],
                tma_atom_dQ,
                tiled_mma_dQ,
                tidx,
                softmax_scale,
                softmax_scale_log2,
                PdS_barrier,
                thr_mma_SdP,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                qhead_per_kvhead_divmod,
                consumer_state_Q,
                consumer_state_dO,
                is_sdp_wg=is_sdp_wg,
            )
            return

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            recompute_fastdiv_mods_q = const_expr(
                aux_data.tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = const_expr(
                aux_data.tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )

            if const_expr(fastdiv_mods is not None and fastdiv_mods[0] is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisorV2(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisorV2(seqlen.seqlen_k),
                )

            mask = AttentionMaskCls(seqlen)
            score_mod_fn = partial(
                self.apply_score_mod,
                thr_mma_SdP=thr_mma_SdP,
                softmax_scale=softmax_scale,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_bwd_fn = partial(
                self.apply_score_mod_bwd,
                thr_mma_SdP=thr_mma_SdP,
                softmax_scale=softmax_scale,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_fn_cur = partial(
                score_mod_fn,
                batch_idx=batch_idx,
                head_idx=head_idx,
                n_block=n_block,
                seqlen_info=seqlen,
            )
            score_mod_bwd_fn_cur = partial(
                score_mod_bwd_fn,
                batch_idx=batch_idx,
                head_idx=head_idx,
                n_block=n_block,
                seqlen_info=seqlen,
            )
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            if const_expr(not self.use_block_sparsity):
                process_tile = (
                    const_expr(not self.is_local and not self.is_varlen_q)
                    or m_block_min < m_block_max
                )
            else:
                total_m_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_m_block_cnt > Int32(0)

            if process_tile:
                mdQaccum_cur = None
                if const_expr(is_dQ_wg and self.dQ_direct_atomic):
                    if const_expr(not seqlen.has_cu_seqlens_q):
                        mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
                    else:
                        mdQaccum_cur = cute.domain_offset(
                            (seqlen.padded_offset_q * self.tile_hdim,),
                            mdQaccum[None, head_idx],
                        )
                if const_expr(not self.use_block_sparsity):
                    mask_fn = partial(
                        mask.apply_mask,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        n_block=n_block,
                        thr_mma=thr_mma_SdP,
                        mask_seqlen=True,
                        mask_causal=self.is_causal,
                        mask_local=self.is_local,
                        mask_mod=self.mask_mod,
                        aux_data=aux_data,
                        fastdiv_mods=fastdiv_mods,
                    )
                    dKV_accumulate = False
                    for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                        consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                            m_block,
                            consumer_state_Q,
                            consumer_state_dO,
                            mask_fn=mask_fn,
                            score_mod_fn=score_mod_fn_cur,
                            score_mod_bwd_fn=score_mod_bwd_fn_cur,
                            dKV_accumulate=dKV_accumulate,
                            mdQaccum_cur=mdQaccum_cur,
                        )
                        dKV_accumulate = True
                else:
                    consumer_state_Q, consumer_state_dO = consume_block_sparse_mma_bwd_sm90(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        consumer_state_Q,
                        consumer_state_dO,
                        mma_one_m_block_all,
                        mask,
                        self.mask_mod,
                        is_causal=self.is_causal,
                        is_local=self.is_local,
                        thr_mma_SdP=thr_mma_SdP,
                        score_mod_fn=score_mod_fn_cur,
                        score_mod_bwd_fn=score_mod_bwd_fn_cur,
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                        aux_data=aux_data,
                        fastdiv_mods=fastdiv_mods,
                    )

                if const_expr(self.compute_dkv):
                    if const_expr(self.qhead_per_kvhead == 1):
                        acc_dK.store(acc_dK.load() * softmax_scale)
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                        mdK_semaphore,
                        mdV_semaphore,
                    )
            else:
                # KV tile with zero Q blocks produces no dK/dV; write zeros.
                if const_expr(
                    self.compute_dkv
                    and (self.use_block_sparsity or self.is_local or self.is_varlen_q)
                ):
                    acc_dK.fill(0.0)
                    acc_dV.fill(0.0)
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                        mdK_semaphore,
                        mdV_semaphore,
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def mma_dq_owner(
        self,
        acc_dQ: cute.Tensor,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_dq_acc_fn: Callable,
        copy_dS_r2s: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        mdQ: cute.Tensor,
        sdQ: cute.Tensor,
        tma_atom_dQ: cute.CopyAtom,
        tiled_mma_dQ: cute.TiledMma,
        tidx: Int32,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        thr_mma_SdP: cute.ThrMma,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        qhead_per_kvhead_divmod: FastDivmodDivisorV2,
        consumer_state_K: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_V: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        is_sdp_wg: cutlass.Constexpr[bool] = True,
    ):
        # M-stationary consumer: each warp group owns dQ[:, wg * 256:(wg + 1) * 256] in
        # registers across the n sweep and writes final dQ exactly once.
        # This alias is safe only with the nonpersistent, single-work-tile scheduler:
        # the producer loads sQ once, and both consumer WGs retire all Q reads below.
        copy_dQ_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_dQ, sdQ, tidx, position_independent=True
        )
        epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.Epilogue), num_threads=self.num_mma_threads
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mask = AttentionMaskCls(seqlen)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            if n_block_min < n_block_max:
                mask_fn = partial(
                    mask.apply_mask,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    m_block=m_block,
                    thr_mma=thr_mma_SdP,
                    mask_seqlen=True,
                    mask_causal=self.is_causal,
                    mask_local=self.is_local,
                    mask_mod=self.mask_mod,
                    aux_data=AuxData(),
                    fastdiv_mods=(None, None),
                )
                for n_block in cutlass.range(n_block_min, n_block_max, unroll=1):
                    consumer_state_K, consumer_state_V = self.mma_one_n_block(
                        n_block,
                        n_block_min,
                        consumer_state_K,
                        consumer_state_V,
                        mma_qk_fn=mma_qk_fn,
                        mma_dov_fn=mma_dov_fn,
                        mma_dq_acc_fn=mma_dq_acc_fn,
                        copy_dS_r2s=copy_dS_r2s,
                        pipeline_Q=pipeline_Q,
                        pipeline_dO=pipeline_dO,
                        tLSEsLSE=tLSEsLSE,
                        tLSEsdPsum=tLSEsdPsum,
                        softmax_scale_log2=softmax_scale_log2,
                        PdS_barrier=PdS_barrier,
                        mask_fn=mask_fn,
                        is_sdp_wg=is_sdp_wg,
                    )
                # Retire the final dQ GEMM, free the last K/V stages.
                warpgroup.wait_group(0)
                pipeline_Q.consumer_release(consumer_state_K)
                consumer_state_K.advance()
            else:
                # No K tile attends to these Q rows. Preprocess no longer zero-fills
                # a scratch buffer, so this CTA must explicitly write its zero output.
                acc_dQ.fill(0.0)

            # A peer WG may still be reading Q for S. Retiring only our own WGMMA
            # is not enough before either WG reuses sQ as the final dQ store buffer.
            epi_barrier.arrive_and_wait()
            acc_dQ.store(acc_dQ.load() * softmax_scale)
            copy_dQ_r2s(acc_dQ, dst_idx=None)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()

            mdQ_cur = seqlen.offset_batch_Q(mdQ, batch_idx, dim=3, ragged=self.is_varlen_q)[
                None, None, head_idx
            ]
            gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
            store_dQ, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dQ, 0, cute.make_layout(1), sdQ, gdQ, single_stage=True
            )
            if warp_idx == 4:
                store_dQ()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma_one_n_block(
        self,
        n_block: Int32,
        n_block_min: Int32,
        consumer_state_K: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_V: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_dq_acc_fn: Callable,
        copy_dS_r2s: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        softmax_scale_log2: Float32,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        mask_fn: Optional[Callable] = None,
        is_sdp_wg: cutlass.Constexpr[bool] = True,
    ):
        is_first = n_block == n_block_min
        smem_idx_PdS = n_block % self.PdS_stage
        if not is_first:
            # Retire dQ GEMM(n-1) so its K stage and sdS buffer are reusable, then
            # free K(n-1) in the alternating physical-slot pipeline.
            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_K)
            consumer_state_K.advance()
        # (1) [GEMM 1] S = Q @ K^T (Q resident at stage 0; carries Q/LSE on iter 0)
        pipeline_Q.consumer_wait(consumer_state_K, pipeline_Q.consumer_try_wait(consumer_state_K))
        if const_expr(is_sdp_wg):
            acc_S = mma_qk_fn(A_idx=0, B_idx=consumer_state_K.index, wg_wait=-1)
            tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, 0])
        # (2) [GEMM 2] dP = dO @ V^T
        pipeline_dO.consumer_wait(consumer_state_V, pipeline_dO.consumer_try_wait(consumer_state_V))
        if const_expr(is_sdp_wg):
            acc_dP = mma_dov_fn(A_idx=0, B_idx=consumer_state_V.index, wg_wait=1)

            # (3) [Pointwise 1] P = exp2(S * scale_log2 - LSE)
            if const_expr(mask_fn is not None):
                mask_fn(acc_S, n_block=n_block)
            acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
            lane_idx = cute.arch.lane_idx()
            for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
                lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_LSE)
                for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                    acc_S_mn[r, c] = cute.math.exp2(
                        acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True
                    )
            tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, 0])

        # (4) [Pointwise 2] dS = P * (dP - dPsum)
        warpgroup.wait_group(0)
        pipeline_dO.consumer_release(consumer_state_V)
        consumer_state_V.advance()
        if const_expr(is_sdp_wg):
            acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
            for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
                dpsum_val = self._get_stat(tLSErdPsum, r, lane_idx, shuffle=self.shuffle_dPsum)
                for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                    acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)
            tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)

        # R2S for dS: WAR on buffer (n % 2) is safe — its previous reader dQ(n-2) was
        # retired by the wait_group(0) at the top of iteration n-1, and the barrier of
        # n-1 orders both warp groups past that point.
        if const_expr(is_sdp_wg):
            copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()

        # (5) [GEMM 3] dQ[:, wg half] += dS @ K[:, wg half]
        mma_dq_acc_fn(
            A_idx=smem_idx_PdS, B_idx=consumer_state_K.index, zero_init=is_first, wg_wait=-1
        )
        return consumer_state_K, consumer_state_V

    @staticmethod
    @cute.jit
    def _get_stat(tSrS: cute.Tensor, row: Int32, lane: Int32, shuffle: bool) -> Float32:
        """Retrieve the statistic for a given accumulator row.

        When shuffle=False, direct register indexing.
        When shuffle=True, warp shuffle from the thread group that holds the value.
        """
        if const_expr(not shuffle):
            return tSrS[row]
        # tSrS: (((2, 1), 1), 1)), distributed across 8 threads in the warp
        vecsize = cute.size(tSrS, mode=[0, 0])  # 2
        idx0, off, idx1 = cute.idx2crd(row, (vecsize, 8, cute.shape(tSrS, mode=[0, 1])))
        # register index: 0, 1, 0, 1, ..., 2, 3, 2, 3, ...
        return utils.shuffle_sync(tSrS[idx0 + idx1 * vecsize], offset=off * 4 + (lane % 4))

    @cute.jit
    def mma_one_m_block(
        self,
        m_block: Int32,
        consumer_state_Q: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_dO: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        copy_Q_s2r: Optional[Callable],
        mma_dsk_fn: Callable,
        copy_P_r2s: Optional[Callable],
        copy_dS_r2s: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tdQsdQaccum: Optional[cute.Tensor],
        dQaccum_thr_copy: Optional[cute.TiledCopy],
        softmax_scale_log2: Float32,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
        is_sdp_wg: cutlass.Constexpr[bool] = True,
        mask_fn: Optional[Callable] = None,
        score_mod_fn: Optional[Callable] = None,
        score_mod_bwd_fn: Optional[Callable] = None,
        dKV_accumulate: Boolean = True,
        mdQaccum_cur: Optional[cute.Tensor] = None,
    ):
        consumer_state_dO_cur = (
            consumer_state_Q
            if const_expr(not self.alias_stream_slots and self.Q_stage == self.dO_stage)
            else consumer_state_dO
        )
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = (
            consumer_state_dO_cur.index
            if const_expr(self.alias_stream_slots or self.dO_stage > 1)
            else 0
        )
        smem_idx_PdS = smem_idx_Q if const_expr(self.PdS_stage > 1) else 0
        # (1) [GEMM 1] S = Q @ K^T
        pipeline_Q.consumer_wait(consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q))
        if const_expr(is_sdp_wg):
            acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
            # If shuffle_LSE, OOB reads are OK since sLSE is already padded
            tLSErLSE = copy_utils.load_s2r(
                tLSEsLSE[None, 0 if const_expr(self.alias_stream_slots) else smem_idx_Q]
            )
        if const_expr(self.Q_in_regs):
            copy_Q_s2r(smem_idx_Q)
        # (2) [GEMM 2] dP = dO @ V.T
        pipeline_dO.consumer_wait(
            consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur)
        )
        if const_expr(is_sdp_wg):
            acc_dP = mma_dov_fn(
                A_idx=smem_idx_dO if const_expr(self.alias_stream_slots) else smem_idx_Q, wg_wait=1
            )

            if const_expr(self.score_mod_bwd is not None):
                acc_S_pre = cute.make_fragment_like(acc_S)
                cute.autovec_copy(acc_S, acc_S_pre)

            if const_expr(self.score_mod is not None):
                score_mod_fn(acc_S, m_block=m_block)

            # (3) [Pointwise 1] P = exp(S - LSE)
            if cutlass.const_expr(mask_fn is not None):
                mask_fn(acc_S, m_block=m_block)
            acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
            lane_idx = cute.arch.lane_idx()
            for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
                lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_LSE)
                for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                    acc_S_mn[r, c] = cute.math.exp2(
                        acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True
                    )
            tLSErdPsum = copy_utils.load_s2r(
                tLSEsdPsum[None, 0 if const_expr(self.alias_stream_slots) else smem_idx_dO]
            )

            # Convert P from f32 -> f16
            tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
        # R2S for P
        if const_expr(self.compute_dkv and not self.mma_dkv_is_rs):
            # sync to ensure P has already been used in the previous iteration before overwriting
            if const_expr(self.PdS_stage == 1):
                PdS_barrier.arrive_and_wait()
            if const_expr(is_sdp_wg):
                copy_P_r2s(tdVrP, dst_idx=smem_idx_PdS)

        # (4) [Pointwise 2] dS = P*(dP-dPsum)
        warpgroup.wait_group(0)
        if const_expr(self.Q_in_regs):
            # Both Q snapshots and the owner's S/LSE reads are retired. B2 may
            # reuse this physical slot for the next dO, never the current dK.
            pipeline_Q.consumer_release(consumer_state_Q)
        if const_expr(is_sdp_wg):
            acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
            for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
                dpsum_val = self._get_stat(tLSErdPsum, r, lane_idx, shuffle=self.shuffle_dPsum)
                for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                    acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)

            if const_expr(self.score_mod_bwd is not None):
                score_mod_bwd_fn(acc_dP, acc_S_pre, m_block=m_block)

            # Convert dS from f32 -> f16
            tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)

        # If there's double buffering on dS, we don't need to sync here.
        # Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
        # But because both WGs have to sync at the end of the loop and double buffering,
        # this race condition is not possible.
        # This sync is to ensure (1) P is written in case of !mma_dkv_is_rs and
        # (2) dS is already read by the Mma in the previous iteration in case of mma_dkv_is_rs.
        if const_expr(not self.mma_dkv_is_rs or (self.PdS_stage == 1 and self.mma_dkv_is_rs)):
            cute.arch.fence_view_async_shared()
            PdS_barrier.arrive_and_wait()

        # R2S for dS
        if const_expr(is_sdp_wg):
            copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)

        # (5) [GEMM 3] dV += P.T @ dO
        if const_expr(self.compute_dkv and not self.mma_dkv_is_rs):
            mma_pdo_fn(
                A_idx=smem_idx_PdS, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1
            )
        elif const_expr(self.compute_dkv):
            mma_pdo_fn(tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1)

        # smem fence to make sure sdS is written before it's read by WGMMA
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()

        if const_expr(is_dQ_wg):
            # (6) [GEMM 4] dQ = dS @ K
            acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)
            pipeline_dO.consumer_release(consumer_state_dO_cur)  # release dO as dV mma is done

            # (7) [GEMM 5] dK += dS.T @ Q
            if const_expr(self.compute_dkv and not self.mma_dkv_is_rs):
                mma_dsq_fn(
                    A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
                )
            elif const_expr(self.compute_dkv):
                mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)

            if const_expr(self.dQ_direct_atomic):
                mdQaccum_tile = cute.domain_offset(
                    (m_block * self.tile_m * self.tile_hdim,), mdQaccum_cur
                )
                num_thr_dQ = self.num_threads_per_warp_group * self.num_wg_dQ
                gdQaccum = cute.make_tensor(
                    mdQaccum_tile.iterator,
                    cute.make_layout(
                        (
                            num_thr_dQ,
                            self.tile_m * self.tile_hdim // num_thr_dQ,
                        )
                    ),
                )
                tdQgdQaccum_cur = dQaccum_thr_copy.partition_D(gdQaccum)
                acc_dQ_atomic = cute.make_tensor(
                    acc_dQ.iterator, cute.make_layout(tdQgdQaccum_cur.shape)
                )
                assert cute.size(acc_dQ_atomic) == cute.size(tdQgdQaccum_cur)
                for i in cutlass.range(cute.size(acc_dQ_atomic), unroll_full=True):
                    utils.atomic_add_fp32(acc_dQ_atomic[i], utils.elem_pointer(tdQgdQaccum_cur, i))
            else:
                # dQ R2S: wait for dQaccum_store to free the smem buffer, then write dQ to smem
                # When dQ_single_wg, only WG0 enters here so warp_group_idx == 0
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                    number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
                )
                tdQrdQaccum_flat = cute.make_tensor(
                    acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape)
                )
                cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                    number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
                )

            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)
        else:
            # dQ_single_wg or dKV-only: dV wait + dK, no dQ GEMM.
            # (7) [GEMM 5] dK += dS.T @ Q
            if const_expr(self.Q_in_regs):
                mma_dsq_fn(B_idx=smem_idx_PdS, zero_init=not dKV_accumulate, wg_wait=1)
            elif const_expr(self.compute_dkv and not self.mma_dkv_is_rs):
                mma_dsq_fn(
                    A_idx=smem_idx_PdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
                )
            elif const_expr(self.compute_dkv):
                mma_dsq_fn(tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1)
            pipeline_dO.consumer_release(consumer_state_dO_cur)
            warpgroup.wait_group(0)
            if const_expr(not self.Q_in_regs):
                pipeline_Q.consumer_release(consumer_state_Q)

        consumer_state_Q.advance()
        consumer_state_dO.advance()
        return consumer_state_Q, consumer_state_dO

    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
    ):
        epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.Epilogue), num_threads=self.num_mma_threads
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if const_expr(self.qhead_per_kvhead == 1):
            mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3, ragged=self.varlen_k)[
                None, None, head_idx
            ]
            mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3, ragged=self.varlen_k)[
                None, None, head_idx
            ]
            gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
            gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
            store_dK, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dK, 0, cute.make_layout(1), sK, gdK, single_stage=True
            )
            store_dV, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dV, 0, cute.make_layout(1), sV, gdV, single_stage=True
            )
            sdV = sV if const_expr(not self.dKV_swapAB) else layout_utils.transpose_view(sV)
            sdK = sK if const_expr(not self.dKV_swapAB) else layout_utils.transpose_view(sK)
            copy_dV_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_dV,
                sdV,
                tidx,
                transpose=self.dKV_swapAB,
                position_independent=True,
            )
            copy_dK_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_dK,
                sdK,
                tidx,
                transpose=self.dKV_swapAB,
                position_independent=True,
            )
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            epi_barrier.arrive_and_wait()
            copy_dV_r2s(acc_dV, dst_idx=None)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dV()
                cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            epi_barrier.arrive_and_wait()
            copy_dK_r2s(acc_dK, dst_idx=None)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dK()
                cute.arch.cp_async_bulk_commit_group()
        else:
            deterministic_KV = self.deterministic and self.qhead_per_kvhead > 1
            sdKaccum_shape0 = self.tile_n * self.tile_hdim // self.num_wg_mma
            sdVaccum_shape0 = self.tile_n * self.tile_hdimv // self.num_wg_mma
            sdKaccum_layout = cute.make_layout((sdKaccum_shape0, self.num_wg_mma))
            sdVaccum_layout = cute.make_layout((sdVaccum_shape0, self.num_wg_mma))
            head_idx_kv = head_idx // qhead_per_kvhead_divmod
            if const_expr(deterministic_KV):
                assert mdK_semaphore is not None
                assert mdV_semaphore is not None
                mdK_semaphore_cur = mdK_semaphore[n_block, None, head_idx_kv, batch_idx]
                mdV_semaphore_cur = mdV_semaphore[n_block, None, head_idx_kv, batch_idx]
                lock_value = head_idx % self.qhead_per_kvhead
            mdKaccum_cur = seqlen.offset_batch_K(
                mdK, batch_idx, dim=2, padded=True, multiple=self.tile_hdim
            )[None, head_idx_kv]
            mdVaccum_cur = seqlen.offset_batch_K(
                mdV, batch_idx, dim=2, padded=True, multiple=self.tile_hdimv
            )[None, head_idx_kv]
            gdKaccum_ = cute.local_tile(mdKaccum_cur, (self.tile_n * self.tile_hdim,), (n_block,))
            gdKaccum = cute.flat_divide(gdKaccum_, (sdKaccum_shape0,))
            gdVaccum_ = cute.local_tile(mdVaccum_cur, (self.tile_n * self.tile_hdimv,), (n_block,))
            gdVaccum = cute.flat_divide(gdVaccum_, (sdVaccum_shape0,))
            # These two overlap each other
            sVaccum_ptr = cute.recast_ptr(sV.iterator, dtype=Float32)
            sdKaccum = cute.make_tensor(sVaccum_ptr, sdKaccum_layout)
            sdVaccum = cute.make_tensor(sVaccum_ptr, sdVaccum_layout)
            tiled_copy_dKVaccum_r2s = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
                cute.make_layout((self.num_threads_per_warp_group, self.num_wg_mma)),
                cute.make_layout(128 // Float32.width),
            )
            thr_copy_dKVaccum_r2s = tiled_copy_dKVaccum_r2s.get_slice(tidx)
            tdKsdKaccum = thr_copy_dKVaccum_r2s.partition_D(sdKaccum)
            tdVsdVaccum = thr_copy_dKVaccum_r2s.partition_D(sdVaccum)

            read_flag = const_expr(not deterministic_KV)
            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            if const_expr(deterministic_KV):
                barrier.wait_eq(mdK_semaphore_cur.iterator, tidx, 0, lock_value)
            epi_barrier.arrive_and_wait()
            tdKrdKaccum_flat = cute.make_tensor(acc_dK.iterator, tdKsdKaccum.shape)
            cute.autovec_copy(tdKrdKaccum_flat, tdKsdKaccum)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_wg_mma):
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKaccum[None, wg_idx].iterator,
                            gdKaccum[None, wg_idx].iterator,
                            self.tma_copy_bytes["dKacc"] // self.num_wg_mma,
                        )
                cute.arch.cp_async_bulk_commit_group()

            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            if const_expr(deterministic_KV):
                barrier.arrive_inc(mdK_semaphore_cur.iterator, tidx, 0, 1)
                barrier.wait_eq(mdV_semaphore_cur.iterator, tidx, 0, lock_value)
            epi_barrier.arrive_and_wait()
            tdVrdVaccum_flat = cute.make_tensor(acc_dV.iterator, tdVsdVaccum.shape)
            cute.autovec_copy(tdVrdVaccum_flat, tdVsdVaccum)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_wg_mma):
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdVaccum[None, wg_idx].iterator,
                            gdVaccum[None, wg_idx].iterator,
                            self.tma_copy_bytes["dVacc"] // self.num_wg_mma,
                        )
                cute.arch.cp_async_bulk_commit_group()
            if const_expr(deterministic_KV):
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                barrier.arrive_inc(mdV_semaphore_cur.iterator, tidx, 0, 1)

    @cute.jit
    def dQaccum_store(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        block_info: BlockInfo,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        # warp-local thread index (dQaccum_store runs on warp 1, global tidx 32-63)
        warp_local_tidx = tidx % cute.arch.WARP_SIZE
        read_flag = const_expr(not self.deterministic)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            else:
                mdQaccum_cur = cute.domain_offset(
                    (seqlen.padded_offset_q * self.tile_hdim,), mdQaccum[None, head_idx]
                )
            # ((M * K / num_wg_dQ, num_wg_dQ), num_m_blocks)
            gdQaccum = cute.local_tile(
                mdQaccum_cur,
                (
                    cute.make_layout(
                        (self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ)
                    ),
                ),
                (None,),
            )

            if const_expr(mdQ_semaphore is not None):
                # mdQ_semaphore is (num_m_blocks, cluster_size, num_head, batch) after transpose
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]

            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            if const_expr(not self.use_block_sparsity):
                process_tile = (
                    const_expr(not self.is_local and not self.is_varlen_q)
                    or m_block_min < m_block_max
                )
                loop_count = m_block_max - m_block_min
            else:
                total_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_block_cnt > Int32(0)

            if process_tile:
                if const_expr(not self.use_block_sparsity):
                    for iter_idx in cutlass.range(loop_count, unroll=1):
                        m_block = m_block_min + iter_idx
                        m_block_safe = m_block

                        num_dQ_chunks = self.num_wg_dQ
                        for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                            if const_expr(not self.deterministic):
                                # If deterministic, we already waited at the end of the prev iter
                                cute.arch.cp_async_bulk_wait_group(
                                    num_dQ_chunks - 1 - warp_group_idx, read=read_flag
                                )
                            cute.arch.barrier_arrive(
                                barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                                number_of_threads=self.num_threads_per_warp_group
                                + cute.arch.WARP_SIZE,
                            )

                        # Semaphore acquire: wait for prior n_blocks to finish writing this m_block
                        if const_expr(self.deterministic):
                            if const_expr(self.spt):
                                _, n_block_max_for_m_block = block_info.get_n_block_min_max(
                                    seqlen, m_block_safe
                                )
                                lock_value = n_block_max_for_m_block - 1 - n_block
                            else:
                                lock_value = n_block
                            barrier.wait_eq(
                                mdQ_semaphore_cur[(m_block_safe, None)].iterator,
                                warp_local_tidx,
                                0,  # flag_offset
                                lock_value,
                            )

                        for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                            cute.arch.barrier(
                                barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                                number_of_threads=self.num_threads_per_warp_group
                                + cute.arch.WARP_SIZE,
                            )
                            with cute.arch.elect_one():
                                copy_utils.cpasync_reduce_bulk_add_f32(
                                    sdQaccum[None, warp_group_idx].iterator,
                                    gdQaccum[(None, warp_group_idx), m_block_safe].iterator,
                                    self.tma_copy_bytes["dQ"],
                                )
                            cute.arch.cp_async_bulk_commit_group()

                        # Semaphore release: signal that this n_block is done with this m_block
                        if const_expr(self.deterministic):
                            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                            barrier.arrive_inc(
                                mdQ_semaphore_cur[(m_block_safe, None)].iterator,
                                warp_local_tidx,
                                0,  # flag_offset
                                1,
                            )
                else:
                    assert not self.deterministic, (
                        "Deterministic not implemented for block-sparse backward"
                    )
                    dQaccum_store_block_sparse_bwd_sm90(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        sdQaccum,
                        gdQaccum,
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                        num_dQ_warp_groups=self.num_wg_dQ,
                        num_threads_per_warp_group=self.num_threads_per_warp_group,
                        tma_copy_bytes_dQ=self.tma_copy_bytes["dQ"],
                    )

            # For local masking + deterministic (non-spt): signal remaining m_blocks
            # that this n_block won't visit, so they don't deadlock waiting.
            if const_expr(
                self.deterministic and not self.spt and block_info.window_size_left is not None
            ):
                m_block_global_max = cute.ceil_div(seqlen.seqlen_q, self.tile_m)
                for m_block in cutlass.range(m_block_max, m_block_global_max, unroll=1):
                    barrier.arrive_inc(
                        mdQ_semaphore_cur[(m_block, None)].iterator,
                        warp_local_tidx,
                        0,  # flag_offset
                        1,
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.deterministic):
            cute.arch.cp_async_bulk_wait_group(0, read=True)


class FlashAttentionBackwardSm90Split:
    """One cacheable CuTe callable for the SM90 D512 dQ and dKV kernels."""

    def __init__(
        self,
        dq_kernel: FlashAttentionBackwardSm90,
        dkv_kernel: FlashAttentionBackwardSm90,
    ):
        assert dq_kernel.dq_owner and not dq_kernel.compute_dkv
        assert not dkv_kernel.dq_owner and not dkv_kernel.fused_dq
        assert dkv_kernel.compute_dkv
        self.dq_kernel = dq_kernel
        self.dkv_kernel = dkv_kernel

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQ: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mCuTotalMBlocksQ: Optional[cute.Tensor] = None,
        mCuTotalMBlocksK: Optional[cute.Tensor] = None,
        tensormap_workspace: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        self.dq_kernel(
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            mdQ,
            mdK,
            mdV,
            softmax_scale,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            window_size_left,
            window_size_right,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            aux_data,
            blocksparse_tensors,
            mCuTotalMBlocksQ,
            None,  # split dQ retains the original unbounded K/V loader
            stream,
        )
        self.dkv_kernel(
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            None,  # dKV-only kernel has no dQ output or accumulator
            mdK,
            mdV,
            softmax_scale,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            window_size_left,
            window_size_right,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            aux_data,
            blocksparse_tensors,
            mCuTotalMBlocksK,
            tensormap_workspace,
            stream,
        )
