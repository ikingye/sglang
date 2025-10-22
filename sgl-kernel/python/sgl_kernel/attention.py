"""
注意力计算内核函数模块

这个模块包含了SGLang中用于注意力计算的高性能内核函数，包括：
- Lightning注意力解码
- 注意力状态合并
- CUTLASS多查询注意力支持

这些函数针对大语言模型的注意力计算进行了高度优化。
"""

from typing import Optional, Tuple  # 类型注解

import torch  # PyTorch深度学习框架


def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):
    """
    Lightning注意力解码函数

    这个函数实现了高效的注意力解码计算，使用Lightning Attention算法
    来加速长序列的注意力计算。它支持增量解码，可以复用之前的KV缓存。

    参数
    ----------
    q : torch.Tensor
        查询张量 (Query)
    k : torch.Tensor
        键张量 (Key)
    v : torch.Tensor
        值张量 (Value)
    past_kv : torch.Tensor
        过去的KV缓存，用于增量解码
    slope : torch.Tensor
        注意力斜率参数，用于位置编码
    output : torch.Tensor
        输出张量，存储计算结果
    new_kv : torch.Tensor
        新的KV缓存，用于下次计算
    """
    # 调用自定义的Lightning注意力解码内核
    torch.ops.sgl_kernel.lightning_attention_decode.default(
        q, k, v, past_kv, slope, output, new_kv
    )


def merge_state(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    合并两个注意力状态

    这个函数用于合并两个注意力状态，通常用于处理并行计算的结果
    或者合并不同分支的注意力状态。它实现了高效的张量合并操作。

    参数
    ----------
    v_a : torch.Tensor
        第一个状态的值张量
    s_a : torch.Tensor
        第一个状态的统计张量
    v_b : torch.Tensor
        第二个状态的值张量
    s_b : torch.Tensor
        第二个状态的统计张量
    v_merged : Optional[torch.Tensor]
        合并后的值张量，如果为None则自动创建
    s_merged : Optional[torch.Tensor]
        合并后的统计张量，如果为None则自动创建

    返回
    -------
    Tuple[torch.Tensor, torch.Tensor]
        合并后的值张量和统计张量
    """
    # 将统计张量转换为float32类型以确保计算精度
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)

    # 如果未提供合并后的张量，则创建新的张量
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)

    # 调用自定义的合并状态内核
    torch.ops.sgl_kernel.merge_state.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


def merge_state_v2(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    合并两个注意力状态（v2版本）

    这是merge_state函数的改进版本，提供了更好的性能和功能支持。
    它使用更优化的内核实现来合并注意力状态。

    参数
    ----------
    v_a : torch.Tensor
        第一个状态的值张量
    s_a : torch.Tensor
        第一个状态的统计张量
    v_b : torch.Tensor
        第二个状态的值张量
    s_b : torch.Tensor
        第二个状态的统计张量
    v_merged : Optional[torch.Tensor]
        合并后的值张量，如果为None则自动创建
    s_merged : Optional[torch.Tensor]
        合并后的统计张量，如果为None则自动创建

    返回
    -------
    Tuple[torch.Tensor, torch.Tensor]
        合并后的值张量和统计张量

    注意
    ----
    当前的自定义merge_attn_states内核不支持FP8数据类型和非CUDA设备。
    可能需要回退到使用Triton内核。
    """
    # 将统计张量转换为float32类型以确保计算精度
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)

    # TODO(DefTruth): 当前的自定义merge_attn_states内核
    # 不支持FP8数据类型和非CUDA设备。
    # 可能需要回退到使用Triton内核。

    # 如果未提供合并后的张量，则创建新的张量
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)

    # 调用自定义的合并状态v2内核
    torch.ops.sgl_kernel.merge_state_v2.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


def cutlass_mla_decode(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    workspace: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int = 1,  # Set to 1 to avoid cuda_graph issue by default.
) -> torch.Tensor:
    assert q_nope.ndim == 3, f"q_nope must be a 3D tensor, but got {q_nope.ndim}"
    assert q_pe.ndim == 3, f"q_pe must be a 3D tensor, but got {q_pe.ndim}"
    assert (
        kv_c_and_k_pe_cache.ndim == 3
    ), f"kv_c_and_k_pe_cache must be a 3D tensor, but got {kv_c_and_k_pe_cache.ndim}"

    B_q, H, D_q_nope = q_nope.shape
    B_q_2, H_2, D_q_pe = q_pe.shape
    assert (B_q == B_q_2) and (H == H_2)

    _, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape

    D_latent = 512
    D_rope = 64
    assert D_q_nope == D_latent
    assert D_q_pe == D_rope
    assert D_ckv == D_latent + D_rope

    MAX_HEADS = 128
    assert H <= MAX_HEADS, f"H must be <= {MAX_HEADS}, but got {H}"
    if H < MAX_HEADS:
        q_nope_padded = q_nope.new_empty((B_q, MAX_HEADS, D_q_nope))
        q_nope_padded[:, :H] = q_nope
        q_nope = q_nope_padded

        q_pe_padded = q_pe.new_empty((B_q, MAX_HEADS, D_q_pe))
        q_pe_padded[:, :H] = q_pe
        q_pe = q_pe_padded

    assert len(page_table.shape) == 2
    B_block_table, block_num = page_table.shape
    assert B_block_table == B_q
    assert block_num > 0, f"block num must be greater than 0, got {block_num}"
    assert block_num % (128 / PAGE_SIZE) == 0

    # TODO(kaixih@nvidia): support fp8
    assert q_nope.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"q_nope.dtype needs to be fp16 or bf16 but got {q_nope.dtype}."
    assert q_nope.dtype == q_pe.dtype == kv_c_and_k_pe_cache.dtype
    assert (
        seq_lens.dtype == torch.int32
    ), f"seq_lens.dtype needs to be int32 but got {seq_lens.dtype}."
    assert (
        page_table.dtype == torch.int32
    ), f"page_table.dtype needs to be int32 but got {page_table.dtype}."

    out = q_nope.new_empty((B_q, MAX_HEADS, D_latent))

    torch.ops.sgl_kernel.cutlass_mla_decode.default(
        out,
        q_nope,
        q_pe,
        kv_c_and_k_pe_cache,
        seq_lens,
        page_table,
        workspace,
        sm_scale,
        num_kv_splits,
    )
    return out[:, :H].contiguous()


def cutlass_mla_get_workspace_size(
    max_seq_len: int,
    num_batches: int,
    sm_count: int = 0,
    num_kv_splits: int = 1,  # Set to 1 to avoid cuda_graph issue by default.
) -> int:
    assert max_seq_len > 0, f"max_seq_len must be greater than 0, got {max_seq_len}"
    assert num_batches > 0, f"num_batches must be greater than 0, got {num_batches}"
    return torch.ops.sgl_kernel.cutlass_mla_get_workspace_size.default(
        max_seq_len, num_batches, sm_count, num_kv_splits
    )
