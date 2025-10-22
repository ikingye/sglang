from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt import operations
from sglang.srt.layers.moe.token_dispatcher import DeepEPConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.operations import Operation

@dataclass
class OperationsStrategy:
    # 有序的算子/YieldOperation 列表，定义层内的执行顺序
    operations: List[Operation]
    # DeepEP 模式下为深度 GEMM 预留的 SM 数量
    deep_gemm_num_sms: Optional[int] = None
    # 两批次重叠调度时使用的阶段偏移量
    tbo_delta_stages: Optional[int] = None

    @classmethod
    def concat(cls, items: List["OperationsStrategy"]) -> "OperationsStrategy":
        # 将多个分层策略拍平成一个，并确保共享参数保持一致
        return OperationsStrategy(
            operations=[x for item in items for x in item.operations],
            deep_gemm_num_sms=_assert_all_same(
                [item.deep_gemm_num_sms for item in items]
            ),
            tbo_delta_stages=_assert_all_same(
                [item.tbo_delta_stages for item in items]
            ),
        )

    @staticmethod
    def init_new_tbo(
        layers: torch.nn.ModuleList,
        forward_mode: ForwardMode,
    ) -> "OperationsStrategy":
        # 根据层类型与前向模式选择合适的调度方案
        layer_name = layers[0].__class__.__name__
        if layer_name == "DeepseekV2DecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_deepseek_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        elif layer_name == "Qwen3MoeDecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_qwen3_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        else:
            raise NotImplementedError


def _assert_all_same(items: List):
    assert all(item == items[0] for item in items)
    return items[0]


# -------------------------------- Strategy for DeepSeek ---------------------------------------


# TODO can refactor to make it more fancy if we have more complex strategies
def _compute_moe_deepseek_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "dense layer TBO not yet implemented"
    if forward_mode == ForwardMode.EXTEND:
        # 预填阶段复用官方博客里的注意力与 MLP 重叠策略
        return _compute_moe_deepseek_blog_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        # 解码阶段通过不同的重叠顺序隐藏注意力和 MoE 路径的延迟
        return _compute_moe_deepseek_blog_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_deepseek_blog_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    # 为专家并行 GEMM 预留 SM，其余交给 DeepEP 的通信任务
    deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            # 在进入稀疏 MoE 之前先完成注意力通信与计算
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            # 阶段 1：收集 token 并挑选专家
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            # 阶段 2：运行专家 GEMM 并重组激活
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_shared_experts,
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_deepseek_blog_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        # 两批次解码互相交错 2 个阶段，保证注意力与 MoE 计算尽量重叠
        tbo_delta_stages=2,
        operations=[
            # 注意力准备提前完成，真正的注意力计算等待重叠窗口
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            # 当上一批的注意力核心在运行时开始 MoE 路由
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            # 拆分派发，使专家与共享专家分阶段推进
            layer.mlp.op_dispatch_a,
            layer.mlp.op_shared_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            operations.YieldOperation(),
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


# -------------------------------- Strategy for Qwen3 ---------------------------------------


# TODO: unstable, current strategy is almost the same as DeepSeek, keep redundant code here for
# convenience to adjust strategy
def _compute_moe_qwen3_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    assert layer.is_layer_sparse, "qwen3 moe only support sparse layers"
    if forward_mode == ForwardMode.EXTEND:
        return _compute_moe_qwen3_prefill(layer)
    elif (
        forward_mode == ForwardMode.DECODE or forward_mode == ForwardMode.TARGET_VERIFY
    ):
        return _compute_moe_qwen3_decode(layer)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_qwen3_prefill(layer):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    # Qwen3 与 DeepSeek 共用 DeepEP 的计算/通信 SM 划分策略
    deep_gemm_num_sms = total_num_sms - DeepEPConfig.get_instance().num_sms

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=0,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            layer.self_attn.op_core,
            layer.op_comm_prepare_mlp,
            # 专家路由与派发沿用 DeepSeek 预填的节奏
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
        ],
    )


def _compute_moe_qwen3_decode(layer):
    return OperationsStrategy(
        deep_gemm_num_sms=None,
        # 下游批次延迟 2 个阶段起跑，与 DeepSeek 策略保持一致
        tbo_delta_stages=2,
        operations=[
            layer.op_comm_prepare_attn,
            layer.self_attn.op_prepare,
            operations.YieldOperation(),
            layer.self_attn.op_core,
            # 解码阶段在最终派发前重叠门控与共享专家
            layer.op_comm_prepare_mlp,
            layer.mlp.op_gate,
            layer.mlp.op_select_experts,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_a,
            operations.YieldOperation(),
            layer.mlp.op_dispatch_b,
            layer.mlp.op_experts,
            layer.mlp.op_combine_a,
            operations.YieldOperation(),
            layer.mlp.op_combine_b,
            layer.mlp.op_output,
            layer.op_comm_postprocess_layer,
            operations.YieldOperation(),
        ],
    )
