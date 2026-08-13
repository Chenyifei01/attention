import argparse
import tilelang
from tilelang import DataType, language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout

import torch

torch.set_default_device("npu")
torch.manual_seed(0)
tilelang.disable_cache()

NUM_CORES = 24

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

@tilelang.jit(out_idx=[], workspace_idx=[], pass_configs=pass_configs)
def sparse_flash_mla_grad(
        batch,
        seq_len_q,
        seq_len_kv,
        heads_q,
        heads_kv,
        dim,
        num_stages=8,
        cross_interval=2,
):
    block_M = 128
    block_N = 128
    D_tile = 128  # D维度的分块大小

    assert dim % D_tile == 0, f"dim must be divisible by {D_tile}"
    n_d_tiles = dim // D_tile  # D维度的分块数量，dim=512时为4
    assert seq_len_q % block_M == 0, f"seq_len_q must be divisible by {block_M}"
    assert seq_len_kv % block_N == 0, f"seq_len_kv must be divisible by {block_N}"
    assert num_stages % 2 == 0, "num_stages must be even for double buffering"

    dtype = "float16"
    accum_dtype = "float"
    sm_scale = (1.0 / dim) ** 0.5

    # B, S, N, D layout
    shape_q = [batch, seq_len_q, heads_q, dim]
    shape_kv = [batch, seq_len_kv, heads_kv, dim]
    shape_o = [batch, seq_len_q, heads_q, dim]

    num_seq_blocks = seq_len_q // block_M
    num_kv_blocks = seq_len_kv // block_N
    block_num = num_seq_blocks * heads_q * batch
    num_iters = T.ceildiv(seq_len_kv, block_N)
    num_outer = T.ceildiv(num_iters, num_stages)

    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    SEM_S_C2V = 0
    SEM_S_V2C = 1
    SEM_DP_C2V = 2
    SEM_DP_V2C = 3
    SEM_P_V2C = 4
    SEM_P_C2V = 5
    SEM_DS_V2C = 6
    SEM_DS_C2V = 7
    SEM_T_V2C = 8
    SEM_T_C2V = 9

    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_Q_L1 = 3
    SIG_DO_L1 = 4

    SIG_L0A = 3
    SIG_L0B = 5
    SIG_L0C = 0

    SIG_IO_UB = 0
    SIG_S_HALF = 1
    SIG_MTE3_MTE2 = 0
    SIG_MASK_UB = 2

    half_M = block_M // 2

    def task_range(cid_val):
        start = cid_val * q_tasks + T.if_then_else(cid_val < r_tasks, cid_val, r_tasks)
        count = q_tasks + T.if_then_else(cid_val < r_tasks, 1, 0)
        return start, count

    @T.prim_func
    def main(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            O: T.Tensor(shape_o, dtype),
            dO: T.Tensor(shape_o, dtype),
            softmax_lse: T.Tensor([batch, seq_len_q, heads_q, 1], accum_dtype),
            ws_s: T.Tensor([NUM_CORES, num_iters, block_M, block_N], accum_dtype),
            ws_dp: T.Tensor([NUM_CORES, num_iters, block_M, block_N], accum_dtype),
            ws_p: T.Tensor([NUM_CORES, num_iters, block_M, block_N], dtype),
            ws_ds: T.Tensor([NUM_CORES, num_iters, block_M, block_N], dtype),
            ws_dq: T.Tensor([batch, seq_len_q, heads_q, dim], accum_dtype),
            ws_dk: T.Tensor([batch, seq_len_kv, heads_kv, dim], accum_dtype),
            ws_dv: T.Tensor([batch, seq_len_kv, heads_kv, dim], accum_dtype),
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # L1 buffers - 使用D_tile而非dim，按D维度分块加载
            q_l1 = T.alloc_L1([block_M, D_tile], dtype)
            k_l1 = T.alloc_L1([block_N, D_tile], dtype)
            v_l1 = T.alloc_L1([block_N, D_tile], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)
            do_l1 = T.alloc_L1([block_M, D_tile], dtype)

            T.annotate_layout({
                q_l1: make_zn_layout(q_l1),
                k_l1: make_nz_layout(k_l1),
                p_l1: make_zn_layout(p_l1),
                v_l1: make_zn_layout(v_l1),
                do_l1: make_zn_layout(do_l1),
            })

            # L0 buffers - 使用D_tile而非dim
            l0a = T.alloc_L0A([2, block_M, D_tile], dtype)
            l0b = T.alloc_L0B([2, D_tile, block_N], dtype)
            l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            d_val = T.alloc_ub([half_M, 1], accum_dtype)
            lse_ub = T.alloc_ub([half_M, 1], accum_dtype)
            d_val_accum = T.alloc_ub([half_M, 1], accum_dtype)

            io_buf_f32 = T.alloc_ub([half_M, block_N], accum_dtype)
            io_buf_f16 = T.alloc_ub([half_M, D_tile], dtype)
            work_ub = T.alloc_ub([half_M, D_tile], accum_dtype)
            buf_2d = T.alloc_ub([half_M, D_tile], accum_dtype)

            my_start, my_count = task_range(cid)

            with T.Scope("C"):
                T.set_cross_flag("MTE2", SEM_T_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                T.set_flag("MTE1", "MTE2", SIG_DO_L1)
                T.set_flag("M", "MTE1", SIG_L0A)
                T.set_flag("M", "MTE1", SIG_L0A + 1)
                T.set_flag("M", "MTE1", SIG_L0B)
                T.set_flag("M", "MTE1", SIG_L0B + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_seq_blocks
                    by = (task_id // num_seq_blocks) % heads_q
                    bz = task_id // (num_seq_blocks * heads_q)
                    kv_by = by // (heads_q // heads_kv)

                    T.barrier_all()

                    T.wait_cross_flag(SEM_T_V2C)

                    for k_outer in T.serial(num_outer):
                        _remaining = num_iters - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)
                        buf_offset = k_outer * num_stages

                        # ============================================================
                        # Phase 1: S = Q @ K^T (reduction over D, 需要D-tile循环累加)
                        # ============================================================
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k_outer * num_stages + i

                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            for d_idx in T.serial(n_d_tiles):
                                # 加载K的D-tile分片到L1: K[bz, idx*N:(idx+1)*N, kv_by, d*D:(d+1)*D]
                                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                T.copy(K[bz, idx * block_N: (idx + 1) * block_N, kv_by, d_idx * D_tile: (d_idx + 1) * D_tile], k_l1)
                                T.set_flag("MTE2", "MTE1", SIG_K_L1)

                                # Q 加载: Q[bz, bx*M:(bx+1)*M, by, d*D:(d+1)*D]
                                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.copy(Q[bz, bx * block_M: (bx + 1) * block_M, by, d_idx * D_tile: (d_idx + 1) * D_tile], q_l1)
                                T.set_flag("MTE2", "MTE1", SIG_Q_L1)

                                T.wait_flag("M", "MTE1", SIG_L0A + side)
                                T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.copy(q_l1, l0a[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.set_flag("MTE1", "M", SIG_L0A + side)

                                # 拷贝K到L0B (transpose)
                                T.wait_flag("M", "MTE1", SIG_L0B + side)
                                T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                                T.copy(k_l1, l0b[side, :, :], transpose=True)
                                T.set_flag("MTE1", "MTE2", SIG_K_L1)

                                T.wait_flag("MTE1", "M", SIG_L0A + side)
                                T.set_flag("MTE1", "M", SIG_L0B + side)
                                T.wait_flag("MTE1", "M", SIG_L0B + side)
                                T.pipe_barrier("M")

                                # MMA: 首个D-tile初始化，后续D-tile累加
                                T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=(d_idx == 0))
                                T.set_flag("M", "MTE1", SIG_L0A + side)
                                T.set_flag("M", "MTE1", SIG_L0B + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)
                            # D-tile循环结束，写出累加结果
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], ws_s[cid, buf_offset + i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_S_C2V)

                        # ============================================================
                        # Phase 2: dP = dO @ V^T (reduction over D, 需要D-tile循环累加)
                        # ============================================================

                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k_outer * num_stages + i
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            for d_idx in T.serial(n_d_tiles):
                                # 加载V的D-tile分片到L1: V[bz, idx*N:(idx+1)*N, kv_by, d*D:(d+1)*D]
                                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                                T.copy(V[bz, idx * block_N: (idx + 1) * block_N, kv_by, d_idx * D_tile: (d_idx + 1) * D_tile], v_l1)
                                T.set_flag("MTE2", "MTE1", SIG_V_L1)

                                # 加载dO的D-tile分片到L1: dO[bz, bx*M:(bx+1)*M, by, d*D:(d+1)*D]
                                T.wait_flag("MTE1", "MTE2", SIG_DO_L1)
                                T.copy(dO[bz, bx * block_M: (bx + 1) * block_M, by, d_idx * D_tile: (d_idx + 1) * D_tile], do_l1)
                                T.set_flag("MTE2", "MTE1", SIG_DO_L1)

                                T.wait_flag("M", "MTE1", SIG_L0A + side)
                                T.wait_flag("MTE2", "MTE1", SIG_DO_L1)
                                T.copy(do_l1, l0a[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_DO_L1)
                                T.set_flag("MTE1", "M", SIG_L0A + side)

                                # 拷贝V到L0B (transpose)
                                T.wait_flag("M", "MTE1", SIG_L0B + side)
                                T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                                T.copy(v_l1, l0b[side, :, :], transpose=True)
                                T.set_flag("MTE1", "MTE2", SIG_V_L1)

                                T.wait_flag("MTE1", "M", SIG_L0A + side)
                                T.set_flag("MTE1", "M", SIG_L0B + side)
                                T.wait_flag("MTE1", "M", SIG_L0B + side)
                                T.pipe_barrier("M")

                                T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=(d_idx == 0))
                                T.set_flag("M", "MTE1", SIG_L0A + side)
                                T.set_flag("M", "MTE1", SIG_L0B + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], ws_dp[cid, buf_offset + i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_DP_C2V)

                        T.wait_cross_flag(SEM_S_V2C)
                        T.wait_cross_flag(SEM_DP_V2C)

                        # ============================================================
                        # Phase 3: dV += P^T @ dO (产生D维输出，每个D-tile写不同切片)
                        # P不依赖D-tile，只需加载一次；dO按D-tile分片加载
                        # ============================================================
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k_outer * num_stages + i

                            # 加载P到L1（不依赖D-tile）
                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_P_V2C)
                            T.copy(ws_p[cid, buf_offset + i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            # 拷贝P^T到l0a（不依赖D-tile，只需一次）
                            T.wait_flag("M", "MTE1", SIG_L0A + side)
                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0A + side)

                            T.wait_flag("MTE1", "M", SIG_L0A + side)

                            for d_idx in T.serial(n_d_tiles):
                                # 加载dO的D-tile分片到L0B: dO[bz, bx*M:(bx+1)*M, by, d*D:(d+1)*D]
                                T.wait_flag("M", "MTE1", SIG_L0B + side)
                                T.wait_flag("MTE1", "MTE2", SIG_DO_L1)
                                T.copy(dO[bz, bx * block_M: (bx + 1) * block_M, by, d_idx * D_tile: (d_idx + 1) * D_tile], do_l1)
                                T.set_flag("MTE2", "MTE1", SIG_DO_L1)

                                T.wait_flag("MTE2", "MTE1", SIG_DO_L1)
                                T.copy(do_l1, l0b[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_DO_L1)

                                T.set_flag("MTE1", "M", SIG_L0B + side)
                                T.wait_flag("MTE1", "M", SIG_L0B + side)
                                T.pipe_barrier("M")

                                T.wait_flag("FIX", "M", SIG_L0C + side)
                                T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0B + side)
                                T.set_flag("M", "FIX", SIG_L0C + side)

                                # 写出当前D-tile对应的dV切片: ws_dv[bz, idx*N:(idx+1)*N, kv_by, d*D:(d+1)*D]
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.tile.atomic_add(ws_dv[bz, idx * block_N: (idx + 1) * block_N, kv_by, d_idx * D_tile: (d_idx + 1) * D_tile], l0c[side, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)
                            T.set_flag("M", "MTE1", SIG_L0A + side)
                        # ============================================================
                        # Phase 4: dK += dS^T @ Q  和  dQ += dS @ K
                        # dS不依赖D-tile；Q和K按D-tile分片加载
                        # ============================================================
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k_outer * num_stages + i

                            # --- Part 1: dK += dS^T @ Q ---
                            # 加载dS到L1
                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_DS_V2C)
                            T.copy(ws_ds[cid, buf_offset + i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            # 拷贝dS^T到l0a（不依赖D-tile）
                            T.wait_flag("M", "MTE1", SIG_L0A + side)
                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0A + side)

                            T.wait_flag("MTE1", "M", SIG_L0A + side)

                            for d_idx in T.serial(n_d_tiles):
                                # 加载Q的D-tile分片到L0B: Q[bz, bx*M:(bx+1)*M, by, d*D:(d+1)*D]
                                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                                T.copy(Q[bz, bx * block_M: (bx + 1) * block_M, by, d_idx * D_tile: (d_idx + 1) * D_tile], q_l1)
                                T.set_flag("MTE2", "MTE1", SIG_Q_L1)

                                T.wait_flag("M", "MTE1", SIG_L0B + side)
                                T.wait_flag("MTE2", "MTE1", SIG_Q_L1)
                                T.copy(q_l1, l0b[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_Q_L1)

                                T.set_flag("MTE1", "M", SIG_L0B + side)
                                T.wait_flag("MTE1", "M", SIG_L0B + side)
                                T.pipe_barrier("M")

                                T.wait_flag("FIX", "M", SIG_L0C + side)
                                T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0B + side)
                                T.set_flag("M", "FIX", SIG_L0C + side)

                                # 写出当前D-tile对应的dK切片: ws_dk[bz, idx*N:(idx+1)*N, kv_by, d*D:(d+1)*D]
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.tile.atomic_add(ws_dk[bz, idx * block_N: (idx + 1) * block_N, kv_by, d_idx * D_tile: (d_idx + 1) * D_tile], l0c[side, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)

                            T.set_flag("M", "MTE1", SIG_L0A + side)
                            # --- Part 2: dQ += dS @ K ---
                            # 重新加载dS到L1
                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            T.copy(ws_ds[cid, buf_offset + i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            # 拷贝dS到l0a（不转置，不依赖D-tile）
                            T.wait_flag("M", "MTE1", SIG_L0A + side)
                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0A + side)

                            T.wait_flag("MTE1", "M", SIG_L0A + side)

                            for d_idx in T.serial(n_d_tiles):
                                # 加载K的D-tile分片到L1: K[bz, idx*N:(idx+1)*N, kv_by, d*D:(d+1)*D]
                                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                                T.copy(K[bz, idx * block_N: (idx + 1) * block_N, kv_by, d_idx * D_tile: (d_idx + 1) * D_tile], k_l1)
                                T.set_flag("MTE2", "MTE1", SIG_K_L1)

                                T.wait_flag("M", "MTE1", SIG_L0B + side)
                                T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                                T.copy(k_l1, l0b[side, :, :])
                                T.set_flag("MTE1", "MTE2", SIG_K_L1)

                                T.set_flag("MTE1", "M", SIG_L0B + side)
                                T.wait_flag("MTE1", "M", SIG_L0B + side)
                                T.pipe_barrier("M")

                                T.wait_flag("FIX", "M", SIG_L0C + side)
                                T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                                T.set_flag("M", "MTE1", SIG_L0B + side)
                                T.set_flag("M", "FIX", SIG_L0C + side)

                                # 写出当前D-tile对应的dQ切片: ws_dq[bz, bx*M:(bx+1)*M, by, d*D:(d+1)*D]
                                T.wait_flag("M", "FIX", SIG_L0C + side)
                                T.tile.atomic_add(ws_dq[bz, bx * block_M: (bx + 1) * block_M, by, d_idx * D_tile: (d_idx + 1) * D_tile], l0c[side, :, :])
                                T.set_flag("FIX", "M", SIG_L0C + side)
                            T.set_flag("M", "MTE1", SIG_L0A + side)

                    T.set_cross_flag("MTE2", SEM_T_C2V)
                # 等待所有flag完成
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("MTE1", "MTE2", SIG_Q_L1)
                T.wait_flag("MTE1", "MTE2", SIG_DO_L1)
                T.wait_flag("M", "MTE1", SIG_L0A)
                T.wait_flag("M", "MTE1", SIG_L0A + 1)
                T.wait_flag("M", "MTE1", SIG_L0B)
                T.wait_flag("M", "MTE1", SIG_L0B + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            # V scope
            with T.Scope("V"):
                T.set_cross_flag("MTE2", SEM_T_V2C)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("MTE3", "V", SIG_S_HALF)
                T.set_flag("MTE3", "MTE2", SIG_MTE3_MTE2)
                T.set_flag("V", "MTE2", SIG_MASK_UB)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_seq_blocks
                    by = (task_id // num_seq_blocks) % heads_q
                    bz = task_id // (num_seq_blocks * heads_q)

                    T.tile.fill(d_val, 0.0)
                    T.tile.fill(d_val_accum, 0.0)
                    # d_val = rowsum(dO * O), tiled over dim
                    T.tile.fill(work_ub, 0.0)
                    for d in T.serial(n_d_tiles):
                        # dO[bz, bx*M+vid*half_M:..., by, d*D:(d+1)*D]
                        T.wait_flag("V", "MTE2", SIG_IO_UB)
                        T.copy(dO[bz, bx * block_M + vid * half_M: bx * block_M + vid * half_M + half_M, by, d * D_tile: (d + 1) * D_tile], io_buf_f16)
                        T.set_flag("MTE2", "V", SIG_IO_UB)

                        T.wait_flag("MTE2", "V", SIG_IO_UB)
                        T.tile.cast(work_ub, io_buf_f16, "CAST_NONE", half_M * D_tile)
                        T.set_flag("V", "MTE2", SIG_IO_UB)

                        # O[bz, bx*M+vid*half_M:..., by, d*D:(d+1)*D]
                        T.wait_flag("V", "MTE2", SIG_IO_UB)
                        T.copy(O[bz, bx * block_M + vid * half_M: bx * block_M + vid * half_M + half_M, by, d * D_tile: (d + 1) * D_tile], io_buf_f16)
                        T.set_flag("MTE2", "V", SIG_IO_UB)

                        T.wait_flag("MTE2", "V", SIG_IO_UB)
                        T.tile.cast(buf_2d, io_buf_f16, "CAST_NONE", half_M * D_tile)
                        T.set_flag("V", "MTE2", SIG_IO_UB)
                        T.tile.mul(work_ub, work_ub, buf_2d)
                        T.reduce_sum(work_ub, d_val_accum, dim=-1)  # 对 D_tile 维度求和
                        T.tile.add(d_val, d_val, d_val_accum)  # 累加到 d_val

                    # softmax_lse[bz, bx*M+vid*half_M:..., by, :]
                    T.copy(softmax_lse[bz, bx * block_M + vid * half_M: bx * block_M + vid * half_M + half_M, by, :],
                           lse_ub)
                    T.barrier_all()

                    T.wait_cross_flag(SEM_T_C2V)

                    for k_outer in T.serial(num_outer):
                        _remaining = num_iters - k_outer * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)
                        buf_offset = k_outer * num_stages

                        for i in T.serial(batch_iters):

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_S_C2V)
                            T.copy(ws_s[cid, buf_offset + i, vid * half_M: vid * half_M + half_M, :], io_buf_f32)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf_f32, work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            T.tile.mul(work_ub, work_ub, sm_scale)

                            T.tile.broadcast(buf_2d, lse_ub)
                            T.tile.sub(work_ub, work_ub, buf_2d)
                            T.tile.exp(work_ub, work_ub)

                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.tile.cast(io_buf_f16, work_ub, "CAST_RINT", half_M * block_N)
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(io_buf_f16, ws_p[cid, buf_offset + i, vid * half_M: vid * half_M + half_M, :])
                            T.set_flag("MTE3", "V", SIG_S_HALF)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_P_V2C)

                        T.set_flag("MTE3", "MTE2", SIG_MTE3_MTE2)
                        T.wait_flag("MTE3", "MTE2", SIG_MTE3_MTE2)

                        for i in T.serial(batch_iters):

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_DP_C2V)
                            T.copy(ws_dp[cid, buf_offset + i, vid * half_M: vid * half_M + half_M, :], io_buf_f32)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf_f32, work_ub)
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            T.tile.broadcast(buf_2d, d_val)
                            T.tile.sub(work_ub, work_ub, buf_2d)

                            T.wait_flag("V", "MTE2", SIG_IO_UB)
                            T.copy(ws_s[cid, buf_offset + i, vid * half_M: vid * half_M + half_M, :], io_buf_f32)
                            T.set_flag("MTE2", "V", SIG_IO_UB)

                            T.wait_flag("MTE2", "V", SIG_IO_UB)
                            T.copy(io_buf_f32, buf_2d)
                            T.set_flag("V", "MTE2", SIG_IO_UB)

                            T.tile.mul(buf_2d, buf_2d, sm_scale)
                            T.tile.broadcast(io_buf_f32, lse_ub)
                            T.tile.sub(buf_2d, buf_2d, io_buf_f32)
                            T.tile.exp(buf_2d, buf_2d)

                            T.tile.mul(work_ub, work_ub, buf_2d)
                            T.tile.mul(work_ub, work_ub, sm_scale)

                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.tile.cast(io_buf_f16, work_ub, "CAST_RINT", half_M * block_N)
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(io_buf_f16, ws_ds[cid, buf_offset + i, vid * half_M: vid * half_M + half_M, :])
                            T.set_flag("MTE3", "V", SIG_S_HALF)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_DS_V2C)

                        T.set_cross_flag("MTE2", SEM_S_V2C)
                        T.set_cross_flag("MTE2", SEM_DP_V2C)

                    T.set_cross_flag("MTE2", SEM_T_V2C)

                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("MTE3", "V", SIG_S_HALF)
                T.wait_flag("MTE3", "MTE2", SIG_MTE3_MTE2)
                T.wait_flag("V", "MTE2", SIG_MASK_UB)

    return main


def get_tnd_idx(actual_q_len, t_idx):
    b_idx = 0
    while t_idx >= actual_q_len[b_idx]:
        b_idx += 1
    if b_idx == 0:
        s1_offset = 0
    else:
        s1_offset = actual_q_len[b_idx - 1]
    s1_idx = t_idx - s1_offset
    return b_idx, s1_idx


def tsoftmax(x, sinks):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    N1 = sinks.shape[0]
    x_max = torch.max(x_max, sinks.view(N1, 1))
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    exp_sink = torch.exp(sinks.view(N1, 1) - x_max)
    x_sum += exp_sink
    ans = y.div(x_sum)
    lse = torch.log(x_sum) + x_max
    return ans, lse


def simpleSoftmax(x, lse):
    x_sub = x.sub(lse)
    softmax_res = torch.exp(x_sub)
    return softmax_res


class SparseAttenSharedKv:
    def __init__(self, query, ori_kv, cmp_kv, out, dy, cmp_sparse_indices,
                 sinks, orgDtype, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
                 ori_win_right, ori_win_left, cmp_ratio, scaleValue, cmp_residual):
        self.query = query
        self.ori_kv = ori_kv
        self.cmp_kv = cmp_kv
        self.out = out
        self.dy = dy
        self.cmp_sparse_indices = cmp_sparse_indices
        self.K = cmp_sparse_indices.size(-1) if cmp_sparse_indices is not None else 0
        self.sinks = sinks
        self.dtype = query.dtype
        self.orgDtype = orgDtype
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_ori_k = cu_seqlens_ori_kv
        self.cu_seqlens_cmp_k = cu_seqlens_cmp_kv
        self.next_token = ori_win_right
        self.pre_token = ori_win_left
        self.cmp_ratio = cmp_ratio
        self.scaleValue = scaleValue
        self.cmp_residual = cmp_residual

    def selectKV(self, ori_kv, cmp_kv, s1_index, topk, cur_s1, cur_s2, cur_s3, b_idx):
        diag_offset = cur_s2 - cur_s1
        right = s1_index + self.next_token + 1 + diag_offset
        left = s1_index - self.pre_token + diag_offset
        start = max(0, left)
        end = min(cur_s2, right)
        end = max(end, 0)
        sel_ori_kv = ori_kv[start:end, :]
        if self.cmp_ratio > 1:
            if self.cmp_ratio == 4 and topk is not None:
                s2_sparse = list()
                topk_cpu = topk.cpu() if topk.is_npu else topk
                for sparse_id in topk_cpu:
                    sparse_id = int(sparse_id)
                    if sparse_id == -1:
                        break
                    begin_idx = sparse_id
                    end_idx = begin_idx + 1 if begin_idx + 1 <= cur_s3 else cur_s3
                    s2_sparse.extend(range(begin_idx, end_idx))
                sel_cmp_kv = cmp_kv[s2_sparse, :] if len(s2_sparse) > 0 else cmp_kv[0:0, :]
            else:
                diag_offset_cmp = (cur_s3 * self.cmp_ratio + self.cmp_residual[b_idx]) - cur_s1
                threshold = max(diag_offset_cmp + s1_index + 1, 0) // self.cmp_ratio
                sel_cmp_kv = cmp_kv[:threshold, :]
        else:
            sel_cmp_kv = None
        return sel_ori_kv, sel_cmp_kv

    def scatterKV(self, dori_out, dcmp_out, dkv, s1_index, topk, cur_s1, cur_s2, cur_s3, b_idx):
        diag_offset = cur_s2 - cur_s1
        right = s1_index + self.next_token + 1 + diag_offset
        left = s1_index - self.pre_token + diag_offset
        start = max(0, left)
        end = min(cur_s2, right)
        end = max(end, 0)
        actual_len = end - start
        if actual_len > 0:
            dori_out[start:end, :] += dkv[:actual_len, :]
            dkv = dkv[actual_len:]
        if self.cmp_ratio > 1:
            if self.cmp_ratio == 4 and topk is not None:
                dkv_start = 0
                topk_cpu = topk.cpu() if topk.is_npu else topk
                for sparse_id in topk_cpu:
                    sparse_id = int(sparse_id)
                    if sparse_id == -1:
                        break
                    begin_idx = sparse_id
                    end_idx = begin_idx + 1 if begin_idx + 1 <= cur_s3 else cur_s3
                    dkv_end = dkv_start + (end_idx - begin_idx)
                    dcmp_out[begin_idx:end_idx, :] += dkv[dkv_start:dkv_end, :]
                    dkv_start = dkv_end
            else:
                diag_offset_cmp = (cur_s3 * self.cmp_ratio + self.cmp_residual[b_idx]) - cur_s1
                threshold = max(diag_offset_cmp + s1_index + 1, 0) // self.cmp_ratio
                dcmp_out[:threshold] += dkv
        return dori_out, dcmp_out

    def forward(self):
        query = self.query.float()
        ori_kv = self.ori_kv.float()
        T1, N1, D_qk = query.shape
        T2, N2, D_qk = ori_kv.shape
        G = N1 // N2
        if self.cmp_ratio > 1:
            cmp_kv = self.cmp_kv.float()
            T3, N2, D_qk = cmp_kv.shape
        out_golden = torch.zeros_like(query)
        lse = torch.zeros(T1, N1)
        query = query.reshape(T1, N2, G, D_qk)
        for i in range(T1):
            n2_idx = 0
            topk = self.cmp_sparse_indices[i][n2_idx] if self.cmp_sparse_indices is not None else None
            qi = query[i][n2_idx]
            b_idx, s1_idx = get_tnd_idx(self.cu_seqlens_q[1:], i)
            s2_start = self.cu_seqlens_ori_k[b_idx]
            s2_end = self.cu_seqlens_ori_k[b_idx + 1]
            ori_kvi = ori_kv[s2_start:s2_end, n2_idx, :]
            curS2 = s2_end - s2_start
            curS1 = self.cu_seqlens_q[b_idx + 1] - self.cu_seqlens_q[b_idx]
            curS3 = self.cu_seqlens_cmp_k[b_idx + 1] - self.cu_seqlens_cmp_k[b_idx]
            if self.cmp_ratio > 1:
                s3_start = self.cu_seqlens_cmp_k[b_idx]
                s3_end = self.cu_seqlens_cmp_k[b_idx + 1]
                cmp_kvi = cmp_kv[s3_start:s3_end, n2_idx, :]
            else:
                cmp_kvi = None
            ori_kvi_calc, cmp_kvi_calc = self.selectKV(ori_kvi, cmp_kvi, s1_idx, topk, curS1, curS2, curS3, b_idx)
            kv = torch.cat((ori_kvi_calc, cmp_kvi_calc), dim=-2) if cmp_kvi_calc is not None and cmp_kvi_calc.size() != 0 else ori_kvi_calc
            if kv.shape[0] != 0:
                qk = torch.matmul(qi, kv.permute(1, 0)).mul(self.scaleValue)
                softmax_res, lsei = tsoftmax(qk, self.sinks)
                outi_golden = torch.matmul(softmax_res, kv)
                out_golden[i] = outi_golden
                lsei = lsei.permute(1, 0)
                lse[i] = lsei
        self.out = out_golden.to(self.orgDtype)
        self.lse = lse
        return self.out, self.lse

    def backward(self):
        query = self.query.float()
        ori_kv = self.ori_kv.float()
        dy = self.dy.float()
        T1, N1, D_qk = query.shape
        T2, N2, D_qk = ori_kv.shape
        G = N1 // N2
        dq = torch.zeros_like(self.query)
        dori_kv = torch.zeros_like(self.ori_kv)
        dsinks = torch.zeros_like(self.sinks)
        if self.cmp_ratio > 1:
            cmp_kv = self.cmp_kv.float()
            T3, N2, D_qk = cmp_kv.shape
            dcmp_kv = torch.zeros_like(self.cmp_kv)
        else:
            dcmp_kv = None
        if self.K != 0 and self.cmp_ratio == 4:
            cmp_softmax_l1 = torch.zeros_like(self.cmp_sparse_indices).to(torch.float)
        else:
            cmp_softmax_l1 = None
        query = query.reshape(T1, N2, G, D_qk)
        dy = dy.reshape(T1, N2, G, D_qk)
        out = self.out.reshape(T1, N2, G, D_qk).float()
        for i in range(T1):
            n2_idx = 0
            topk = self.cmp_sparse_indices[i][n2_idx] if self.cmp_sparse_indices is not None else None
            qi = query[i][n2_idx]
            dyi = dy[i][n2_idx]
            outi = out[i][n2_idx]
            b_idx, s1_idx = get_tnd_idx(self.cu_seqlens_q[1:], i)
            s2_start = self.cu_seqlens_ori_k[b_idx]
            s2_end = self.cu_seqlens_ori_k[b_idx + 1]
            ori_kvi = ori_kv[s2_start:s2_end, n2_idx, :]
            curS2 = s2_end - s2_start
            curS1 = self.cu_seqlens_q[b_idx + 1] - self.cu_seqlens_q[b_idx]
            curS3 = self.cu_seqlens_cmp_k[b_idx + 1] - self.cu_seqlens_cmp_k[b_idx]
            if self.cmp_ratio > 1:
                s3_start = self.cu_seqlens_cmp_k[b_idx]
                s3_end = self.cu_seqlens_cmp_k[b_idx + 1]
                cmp_kvi = cmp_kv[s3_start:s3_end, n2_idx, :]
            else:
                cmp_kvi = None
            ori_kvi_calc, cmp_kvi_calc = self.selectKV(ori_kvi, cmp_kvi, s1_idx, topk, curS1, curS2, curS3, b_idx)
            ori_s2 = ori_kvi_calc.shape[0] if ori_kvi_calc is not None and ori_kvi_calc.size() != 0 else 0
            kv = torch.cat((ori_kvi_calc, cmp_kvi_calc), dim=-2) if cmp_kvi_calc is not None and cmp_kvi_calc.size() != 0 else ori_kvi_calc
            if kv.shape[0] != 0:
                _D_tile = 128
                _n_d = D_qk // _D_tile
                _bn = 128
                qk = torch.zeros(qi.shape[0], kv.shape[0], dtype=torch.float32, device=qi.device)
                for _d in range(_n_d):
                    qk += torch.matmul(qi[:, _d*_D_tile:(_d+1)*_D_tile], kv[:, _d*_D_tile:(_d+1)*_D_tile].permute(1, 0))
                qk = qk.mul(self.scaleValue)
                softmax_res = simpleSoftmax(qk, self.lse[i, :].reshape(N1, 1))
                if self.K != 0 and self.cmp_ratio == 4:
                    cmp_softmax = softmax_res[:, ori_s2:]
                    cmp_softmax = cmp_softmax.sum(dim=0, keepdim=True) / G
                    cmp_softmax_l1[i, :, :cmp_softmax.shape[-1]] = cmp_softmax
                dp = torch.zeros(qi.shape[0], kv.shape[0], dtype=torch.float32, device=qi.device)
                for _d in range(_n_d):
                    dp += torch.matmul(dyi[:, _d*_D_tile:(_d+1)*_D_tile], kv[:, _d*_D_tile:(_d+1)*_D_tile].permute(1, 0))
                _dval = torch.zeros(qi.shape[0], 1, dtype=torch.float32, device=qi.device)
                for _d in range(_n_d):
                    _dval += (dyi[:, _d*_D_tile:(_d+1)*_D_tile] * outi[:, _d*_D_tile:(_d+1)*_D_tile]).sum(dim=-1, keepdims=True)
                softmax_grad_res = (dp - _dval) * softmax_res
                softmax_grad_res = softmax_grad_res * self.scaleValue
                dsink_sum = softmax_res * dp
                dsink_sum *= simpleSoftmax(self.sinks.reshape(N1, 1), self.lse[i, :].reshape(N1, 1))
                dsink = -dsink_sum.sum(dim=(-1))
                dsinks += dsink
                dqi = torch.zeros(qi.shape[0], D_qk, dtype=torch.float32, device=qi.device)
                for _j in range(0, kv.shape[0], _bn):
                    dqi += torch.matmul(softmax_grad_res[:, _j:_j+_bn], kv[_j:_j+_bn, :])
                dkv = torch.matmul(softmax_grad_res.permute(1, 0), qi)
                dkv += torch.matmul(softmax_res.permute(1, 0).to(self.orgDtype).to(torch.float32), dyi)
                dq[i:i + 1, :, :] = dqi
                if self.cmp_ratio > 1:
                    dori_kv[s2_start:s2_end, n2_idx, :], dcmp_kv[s3_start:s3_end, n2_idx, :] = self.scatterKV(
                        dori_kv[s2_start:s2_end, n2_idx, :], dcmp_kv[s3_start:s3_end, n2_idx, :], dkv, s1_idx, topk, curS1, curS2, curS3, b_idx)
                else:
                    dori_kv[s2_start:s2_end, n2_idx, :], _ = self.scatterKV(
                        dori_kv[s2_start:s2_end, n2_idx, :], None, dkv, s1_idx, topk, curS1, curS2, curS3, b_idx)
        return dq, dori_kv, dcmp_kv, dsinks, cmp_softmax_l1


def run_flash_mla_grad(func, q, k, v, o, do, lse, ws_dq, ws_dk, ws_dv,
                       num_iters=8, block_M=128, block_N=128, accum_dtype="float32", dtype="float16"):
    """支持 [B, S, N, D] 和 [T, N, D] 两种输入格式。

    [B, S, N, D]: 4D, batch × seq_len × heads × dim
    [T, N, D]:    3D, seq_len × heads × dim (隐式 batch=1)
    """
    squeeze_batch = False
    if q.ndim == 3:
        # T, N, D -> 1, T, N, D
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        o = o.unsqueeze(0)
        do = do.unsqueeze(0)
        lse = lse.unsqueeze(0)
        ws_dq = ws_dq.unsqueeze(0)
        ws_dk = ws_dk.unsqueeze(0)
        ws_dv = ws_dv.unsqueeze(0)
        squeeze_batch = True

    ws_s = torch.zeros((NUM_CORES, num_iters, block_M, block_N), dtype=getattr(torch, accum_dtype))
    ws_dp = torch.zeros((NUM_CORES, num_iters, block_M, block_N), dtype=getattr(torch, accum_dtype))
    ws_p = torch.zeros((NUM_CORES, num_iters, block_M, block_N), dtype=getattr(torch, dtype))
    ws_ds = torch.zeros((NUM_CORES, num_iters, block_M, block_N), dtype=getattr(torch, dtype))

    func(q, k, v, o, do, lse, ws_s, ws_dp, ws_p, ws_ds, ws_dq, ws_dk, ws_dv)

    if squeeze_batch:
        ws_dq.squeeze_(0)
        ws_dk.squeeze_(0)
        ws_dv.squeeze_(0)

    return ws_dq, ws_dk, ws_dv, ws_s, ws_dp, ws_p, ws_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--Sq", type=int, default=256)
    parser.add_argument("--Skv", type=int, default=1024)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--D", type=int, default=512)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--cross-interval", type=int, default=2)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--format", type=str, default="tnd",
                        choices=["bsnd", "tnd"],
                        help="Input format: bsnd=[B,S,N,D], tnd=[T,N,D]")
    parser.add_argument("--ori-win-right", type=int, default=None,
                        help="Local window right size (default: full attention)")
    parser.add_argument("--ori-win-left", type=int, default=None,
                        help="Local window left size (default: full attention)")
    parser.add_argument("--cmp-ratio", type=int, default=1,
                        help="Compression ratio for KV (1=no compression)")
    args = parser.parse_args()

    B, Sq, Skv, H_Q, H_KV, D = args.B, args.Sq, args.Skv, args.H, args.kv_heads, args.D
    use_tnd = (args.format == "tnd")

    # tnd 格式下 batch 固定为 1，内核编译时使用 B=1
    kernel_batch = 1 if use_tnd else B

    func = sparse_flash_mla_grad(
        batch=kernel_batch, seq_len_q=Sq, seq_len_kv=Skv, heads_q=H_Q, heads_kv=H_KV,
        dim=D, num_stages=args.num_stages, cross_interval=args.cross_interval,
    )


    # 稀疏注意力参数
    ori_win_right = args.ori_win_right if args.ori_win_right is not None else Skv
    ori_win_left = args.ori_win_left if args.ori_win_left is not None else Skv
    cmp_ratio = args.cmp_ratio
    scaleValue = (1.0 / D) ** 0.5

    # 创建输入张量（共享 KV：K = V）
    if use_tnd:
        q = torch.randn((Sq, H_Q, D), dtype=torch.float16)
        kv_data = torch.randn((Skv, H_KV, D), dtype=torch.float16)
        k = kv_data.clone()
        v = kv_data.clone()
        do = torch.randn((Sq, H_Q, D), dtype=torch.float16)
    else:
        q = torch.randn((B, Sq, H_Q, D), dtype=torch.float16)
        kv_data = torch.randn((B, Skv, H_KV, D), dtype=torch.float16)
        k = kv_data.clone()
        v = kv_data.clone()
        do = torch.randn((B, Sq, H_Q, D), dtype=torch.float16)

    # 转换为 [T, N, D] 格式
    if use_tnd:
        q_tnd = q
        kv_tnd = kv_data
        do_tnd = do
    else:
        q_tnd = q.reshape(B * Sq, H_Q, D)
        kv_tnd = kv_data.reshape(B * Skv, H_KV, D)
        do_tnd = do.reshape(B * Sq, H_Q, D)

    T1, N1, _ = q_tnd.shape
    T2, N2, _ = kv_tnd.shape
    G = N1 // N2  # 每个 KV head 对应的 query head 数

    # cu_seqlens
    if use_tnd:
        cu_seqlens_q = [0, T1]
        cu_seqlens_ori_kv = [0, T2]
    else:
        cu_seqlens_q = [0] + [(i + 1) * Sq for i in range(B)]
        cu_seqlens_ori_kv = [0] + [(i + 1) * Skv for i in range(B)]
    cu_seqlens_cmp_kv = [0] * len(cu_seqlens_q)
    cmp_residual = [0] * len(cu_seqlens_q)

    # ================================================================
    # 按 KV head 分组调用 SparseAttenSharedKv
    # 因为 SparseAttenSharedKv 内部固定 n2_idx=0，只处理 1 个 KV head
    # 所以每次传入 G 个 query heads + 1 个 KV head，使 N1=G 维度对齐
    # ================================================================
    o_ref_tnd = torch.zeros_like(q_tnd)
    lse_ref_tnd = torch.zeros(T1, N1)
    sparse_attn_instances = []

    for kv_h in range(H_KV):
        # 提取当前 KV head 对应的 G 个 query heads
        q_group = q_tnd[:, kv_h * G:(kv_h + 1) * G, :].contiguous()   # [T1, G, D]
        kv_group = kv_tnd[:, kv_h:kv_h + 1, :].contiguous()           # [T2, 1, D]
        do_group = do_tnd[:, kv_h * G:(kv_h + 1) * G, :].contiguous() # [T1, G, D]

        # sinks 大小必须等于 G（与 qk 的行数一致）
        sinks_group = torch.full((G,), float('-inf'), dtype=torch.float32)

        out_group = torch.zeros_like(q_group)

        sparse_attn = SparseAttenSharedKv(
            query=q_group,
            ori_kv=kv_group,
            cmp_kv=None,
            out=out_group,
            dy=do_group,
            cmp_sparse_indices=None,
            sinks=sinks_group,
            orgDtype=torch.float16,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            ori_win_right=ori_win_right,
            ori_win_left=ori_win_left,
            cmp_ratio=1,
            scaleValue=scaleValue,
            cmp_residual=cmp_residual,
        )

        o_group, lse_group = sparse_attn.forward()
        o_ref_tnd[:, kv_h * G:(kv_h + 1) * G, :] = o_group
        lse_ref_tnd[:, kv_h * G:(kv_h + 1) * G] = lse_group
        sparse_attn_instances.append(sparse_attn)

    torch.npu.synchronize()
    print("Init successful!")

    # lse: [T1, N1] -> [T1, N1, 1]
    lse_ref_tnd = lse_ref_tnd.unsqueeze(-1).to(torch.float32).contiguous()

    # 转回原始格式供内核使用
    if use_tnd:
        o_ref_final = o_ref_tnd.contiguous()
        lse_ref = lse_ref_tnd
    else:
        o_ref_final = o_ref_tnd.reshape(B, Sq, H_Q, D).contiguous()
        lse_ref = lse_ref_tnd.reshape(B, Sq, H_Q, 1).contiguous()

    # workspace 输出
    if use_tnd:
        ws_dq = torch.zeros((Sq, H_Q, D), dtype=torch.float32)
        ws_dk = torch.zeros((Skv, H_KV, D), dtype=torch.float32)
        ws_dv = torch.zeros((Skv, H_KV, D), dtype=torch.float32)
    else:
        ws_dq = torch.zeros((B, Sq, H_Q, D), dtype=torch.float32)
        ws_dk = torch.zeros((B, Skv, H_KV, D), dtype=torch.float32)
        ws_dv = torch.zeros((B, Skv, H_KV, D), dtype=torch.float32)

    num_iters = Skv // 128
    ws_dq, ws_dk, ws_dv, _, _, _, _ = run_flash_mla_grad(
        func, q, k, v, o_ref_final, do, lse_ref, ws_dq, ws_dk, ws_dv,
        num_iters=num_iters, block_M=128, block_N=128,
    )
    torch.npu.synchronize()
    print("func successful!")

    if not args.no_check:
        # 按 KV head 分组调用 backward
        ref_dq_tnd = torch.zeros_like(q_tnd)
        ref_dkv_tnd = torch.zeros_like(kv_tnd)

        for kv_h in range(H_KV):
            sparse_attn = sparse_attn_instances[kv_h]
            ref_dq_group, ref_dori_kv_group, _, _, _ = sparse_attn.backward()

            # 将分组结果写回完整 head 维度
            ref_dq_tnd[:, kv_h * G:(kv_h + 1) * G, :] = ref_dq_group
            ref_dkv_tnd[:, kv_h:kv_h + 1, :] = ref_dori_kv_group

        # 转回原始格式
        if not use_tnd:
            ref_dq = ref_dq_tnd.reshape(B, Sq, H_Q, D)
            ref_dkv = ref_dkv_tnd.reshape(B, Skv, H_KV, D)
        else:
            ref_dq = ref_dq_tnd
            ref_dkv = ref_dkv_tnd

        torch.npu.synchronize()

        mean_tol = 5e-3
        dq_diff = (ws_dq.float() - ref_dq.float()).abs()
        dq_mean_diff = dq_diff.mean().item()
        dq_max_diff = dq_diff.max().item()
        dq_match = dq_mean_diff < mean_tol
        dkv_diff = ((ws_dk + ws_dv).float() - ref_dkv.float()).abs()
        dkv_mean_diff = dkv_diff.mean().item()
        dkv_max_diff = dkv_diff.max().item()
        dkv_match = dkv_mean_diff < mean_tol
        print(f"dQ match: {dq_match}, max diff: {dq_max_diff:.6f}, mean diff: {dq_mean_diff:.6e} (tol: mean < {mean_tol})")
        print(f"dKV match (dK+dV): {dkv_match}, max diff: {dkv_max_diff:.6f}, mean diff: {dkv_mean_diff:.6e} (tol: mean < {mean_tol})")
        exit(0 if (dq_match and dkv_match) else 1)