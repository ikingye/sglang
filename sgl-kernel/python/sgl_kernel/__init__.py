"""
SGLang内核模块初始化文件

这个模块是SGLang高性能计算内核的入口点，包含了所有优化的CUDA/ROCm内核函数。
这些内核函数针对大语言模型推理进行了高度优化，提供了：
- 高效的注意力计算
- 优化的矩阵乘法操作
- 快速的内存管理
- 专门的采样算法
- 多专家模型(MoE)支持
"""

import ctypes  # C类型接口
import os  # 操作系统接口
import platform  # 平台信息

import torch  # PyTorch深度学习框架

# 获取系统架构信息
SYSTEM_ARCH = platform.machine()

# 动态加载CUDA运行时库
# 这确保了CUDA函数在需要时可用
cuda_path = f"/usr/local/cuda/targets/{SYSTEM_ARCH}-linux/lib/libcudart.so.12"
if os.path.exists(cuda_path):
    ctypes.CDLL(cuda_path, mode=ctypes.RTLD_GLOBAL)

# 导入通用操作模块
from sgl_kernel import common_ops
# 导入全归约操作，用于分布式计算
from sgl_kernel.allreduce import *
# 导入注意力计算相关函数
from sgl_kernel.attention import (
    cutlass_mla_decode,  # CUTLASS多查询注意力解码
    cutlass_mla_get_workspace_size,  # 获取CUTLASS MLA工作空间大小
    lightning_attention_decode,  # Lightning注意力解码
    merge_state,  # 合并注意力状态
    merge_state_v2,  # 合并注意力状态v2版本
)
# 导入CUTLASS MoE相关函数
from sgl_kernel.cutlass_moe import cutlass_w4a8_moe_mm, get_cutlass_w4a8_moe_mm_data
# 导入逐元素操作函数
from sgl_kernel.elementwise import (
    apply_rope_with_cos_sin_cache_inplace,  # 原地应用RoPE位置编码
    fused_add_rmsnorm,  # 融合加法和RMS归一化
    gelu_and_mul,  # GELU激活和乘法
    gelu_tanh_and_mul,  # GELU Tanh激活和乘法
    gemma_fused_add_rmsnorm,  # Gemma融合加法和RMS归一化
    gemma_rmsnorm,  # Gemma RMS归一化
    rmsnorm,  # RMS归一化
    silu_and_mul,  # SiLU激活和乘法
)
# 导入融合MoE函数
from sgl_kernel.fused_moe import fused_marlin_moe

# 如果是HIP环境（AMD GPU），导入额外的GELU函数
if torch.version.hip is not None:
    from sgl_kernel.elementwise import gelu_quick

# 导入矩阵乘法相关函数
from sgl_kernel.gemm import (
    awq_dequantize,  # AWQ反量化
    bmm_fp8,  # FP8批量矩阵乘法
    cutlass_scaled_fp4_mm,  # CUTLASS缩放FP4矩阵乘法
    dsv3_fused_a_gemm,  # DSV3融合A矩阵乘法
    dsv3_router_gemm,  # DSV3路由器矩阵乘法
    fp8_blockwise_scaled_mm,  # FP8分块缩放矩阵乘法
    fp8_scaled_mm,  # FP8缩放矩阵乘法
    int8_scaled_mm,  # INT8缩放矩阵乘法
    qserve_w4a8_per_chn_gemm,  # QServe W4A8每通道矩阵乘法
    qserve_w4a8_per_group_gemm,  # QServe W4A8每组矩阵乘法
    scaled_fp4_experts_quant,  # 缩放FP4专家量化
    scaled_fp4_quant,  # 缩放FP4量化
    sgl_per_tensor_quant_fp8,  # SGL每张量FP8量化
    sgl_per_token_group_quant_fp8,  # SGL每token组FP8量化
    sgl_per_token_group_quant_int8,  # SGL每token组INT8量化
    sgl_per_token_quant_fp8,  # SGL每token FP8量化
    shuffle_rows,  # 行洗牌操作
)
# 导入语法约束相关函数
from sgl_kernel.grammar import apply_token_bitmask_inplace_cuda
# 导入KV缓存IO相关函数
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer,  # 传输所有层的KV缓存
    transfer_kv_all_layer_mla,  # 传输所有层的MLA KV缓存
    transfer_kv_per_layer,  # 按层传输KV缓存
    transfer_kv_per_layer_mla,  # 按层传输MLA KV缓存
)
# 导入Marlin量化相关函数
from sgl_kernel.marlin import (
    awq_marlin_moe_repack,  # AWQ Marlin MoE重新打包
    awq_marlin_repack,  # AWQ Marlin重新打包
    gptq_marlin_repack,  # GPTQ Marlin重新打包
)
# 导入MoE相关函数
from sgl_kernel.moe import (
    apply_shuffle_mul_sum,  # 应用洗牌乘法和求和
    cutlass_fp4_group_mm,  # CUTLASS FP4分组矩阵乘法
    ep_moe_post_reorder,  # EP MoE后重排序
    ep_moe_pre_reorder,  # EP MoE预重排序
    ep_moe_silu_and_mul,  # EP MoE SiLU和乘法
    fp8_blockwise_scaled_grouped_mm,  # FP8分块缩放分组矩阵乘法
    moe_align_block_size,  # MoE对齐块大小
    moe_fused_gate,  # MoE融合门控
    prepare_moe_input,  # 准备MoE输入
    topk_softmax,  # TopK Softmax
)
# 导入采样相关函数
from sgl_kernel.sampling import (
    min_p_sampling_from_probs,  # 从概率进行最小P采样
    top_k_renorm_prob,  # TopK重归一化概率
    top_k_top_p_sampling_from_probs,  # 从概率进行TopK TopP采样
    top_p_renorm_prob,  # TopP重归一化概率
    top_p_sampling_from_probs,  # 从概率进行TopP采样
)
# 导入空间相关函数
from sgl_kernel.spatial import create_greenctx_stream_by_value, get_sm_available
# 导入推测解码相关函数
from sgl_kernel.speculative import (
    build_tree_kernel_efficient,  # 高效构建树内核
    segment_packbits,  # 段打包位
    tree_speculative_sampling_target_only,  # 树推测采样仅目标
    verify_tree_greedy,  # 验证树贪心
)
# 导入TopK相关函数
from sgl_kernel.top_k import fast_topk
# 导入版本信息
from sgl_kernel.version import __version__

# 向后兼容性：保留旧的函数名
# TODO(ying): 在更新sglang python代码后移除这个
build_tree_kernel = (
    None  # 已弃用的函数，保留用于向后兼容
)
