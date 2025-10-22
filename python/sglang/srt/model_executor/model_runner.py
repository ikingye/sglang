# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ModelRunner runs the forward passes of the models."""

import datetime
import gc
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.distributed import (
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_size,
    initialize_dp_attention,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend
from sglang.srt.layers.quantization import (
    deep_gemm_wrapper,
    monkey_patch_isinstance_for_vllm_base_layer,
)
from sglang.srt.layers.sampler import Sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.layers.utils import is_sm100_supported
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import (
    GLOBAL_SERVER_ARGS_KEYS,
    global_server_args_dict,
)
from sglang.srt.mem_cache.allocator import (
    AscendPagedTokenToKVPoolAllocator,
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import (
    AscendMLAPagedTokenToKVPool,
    AscendTokenToKVPool,
    DoubleSparseTokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader import get_model
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    MultiprocessingSerializer,
    cpu_has_amx_support,
    dynamic_import,
    enable_show_time_cost,
    get_available_gpu_memory,
    get_bool_env_var,
    get_cpu_ids_by_node,
    init_custom_process_group,
    is_fa3_default_architecture,
    is_flashinfer_available,
    is_hip,
    is_hopper_with_cuda_12_3,
    is_no_spec_infer_or_topk_one,
    is_npu,
    monkey_patch_p2p_access_check,
    monkey_patch_vllm_gguf_config,
    set_cpu_offload_max_bytes,
    set_cuda_arch,
)
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()

# Use a small KV cache pool size for tests in CI
SGLANG_CI_SMALL_KV_SIZE = os.getenv("SGLANG_CI_SMALL_KV_SIZE", None)

# Detect stragger ranks in model loading
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 300

logger = logging.getLogger(__name__)


class RankZeroFilter(logging.Filter):
    """Filter that only allows INFO level logs from rank 0, but allows all other levels from any rank."""

    def __init__(self, is_rank_zero):
        super().__init__()
        self.is_rank_zero = is_rank_zero

    def filter(self, record):
        if record.levelno == logging.INFO:
            return self.is_rank_zero
        return True


class ModelRunner:
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
    ):
        # Parse args
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id

        # Apply the rank zero filter to logger
        if not any(isinstance(f, RankZeroFilter) for f in logger.filters):
            # 仅在单个张量并行分片上打印 INFO，减少日志噪声
            logger.addFilter(RankZeroFilter(tp_rank == 0))
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.dp_size = server_args.dp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid = model_config.is_hybrid
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size

        self.forward_pass_id = 0

        # Model-specific adjustment
        self.model_specific_adjustment()

        if server_args.show_time_cost:
            enable_show_time_cost()

        # Global vars
        # 将常用配置写入模块级字典，方便其他组件读取
        global_server_args_dict.update(
            {k: getattr(server_args, k) for k in GLOBAL_SERVER_ARGS_KEYS}
            | {
                # TODO it is indeed not a "server args"
                "use_mla_backend": self.use_mla_backend,
                "speculative_algorithm": self.spec_algorithm,
            }
            | {
                "moe_a2a_backend": MoeA2ABackend(server_args.moe_a2a_backend),
                "deepep_mode": DeepEPMode(server_args.deepep_mode),
            }
        )

        # CPU offload
        # 控制激活溢写到主内存的上限
        set_cpu_offload_max_bytes(int(server_args.cpu_offload_gb * 1024**3))

        # Init OpenMP threads binding for CPU
        if self.device == "cpu":
            self.init_threads_binding()

        # Get memory before model loading
        # 同时校验张量并行各卡的可用显存是否均衡
        min_per_gpu_memory = self.init_torch_distributed()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            # 针对当前 GPU 自动调优 DeepGEMM 的共享内存与块配置
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # If it is a draft model, tp_group can be different
        # 完成分布式初始化与调参后继续构建运行时
        self.initialize(min_per_gpu_memory)

        # temporary cached values
        # 利用 forward 签名判断模型是否支持接收管道代理张量，以兼容混合 PP 场景
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )
        self._model_update_group = {}

    def initialize(self, min_per_gpu_memory: float):
        server_args = self.server_args

        # 使用辅助器在需要时将张量转存到主机，减少显存压力
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        if not self.is_draft_worker:
            # 初始化专家并行各 rank 共享的专家位置信息
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(server_args, self.model_config)
            )
            if self.tp_rank == 0 and get_bool_env_var(
                "SGLANG_LOG_EXPERT_LOCATION_METADATA"
            ):
                logger.info(
                    f"Initial expert_location_metadata: {get_global_expert_location_metadata()}"
                )

            # 统计动态路由频率，用于日志与自适应均衡策略
            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.tp_rank,
                )
            )

        self.eplb_manager = (
            # EPLB 会在运行时重新分配专家以平衡负载
            EPLBManager(self)
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )
        self.expert_location_updater = ExpertLocationUpdater()

        # Load the model
        # 在内存优化上下文中实例化 Transformer 权重
        # 采样器封装温度/topk 等策略，模型前向完成后可直接复用
        self.sampler = Sampler()
        self.load_model()

        # Check if the model is using hybrid SWA
        if (
            not self.server_args.disable_hybrid_swa_memory
            and self.sliding_window_size is not None
            and self.sliding_window_size > 0
        ):
            # 部分多 token 生成模型需要启用混合 SWA 缓存布局
            architectures = self.model_config.hf_config.architectures
            if architectures and not any("Llama4" in arch for arch in architectures):
                self.is_hybrid = self.model_config.is_hybrid = True

        # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
        # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
        # determine the number of layers.
        model_has_mtp_layers = self.model_config.num_nextn_predict_layers is not None
        model_num_layers = (
            self.model_config.num_nextn_predict_layers
            if self.is_draft_worker and model_has_mtp_layers
            else self.model_config.num_hidden_layers
        )
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", model_num_layers)
        self.num_effective_layers = self.end_layer - self.start_layer
        # 管道并行需要连续层范围，MTP 模型则要求整体分片
        assert (not model_has_mtp_layers) or (
            self.num_effective_layers == model_num_layers
        ), "PP is not compatible with MTP models."

        # Apply torchao quantization
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            # TorchAO 会重写线性层，让其使用量化内核
            apply_torchao_config_to_model(
                self.model, global_server_args_dict["torchao_config"]
            )

        # Apply torch TP if the model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            # 将参数分片切换为 PyTorch 原生的张量并行实现
            self.apply_torch_tp()

        # Init lora
        if server_args.enable_lora:
            # LoRA 管理器负责加载适配器并在请求时合并权重
            self.init_lora_manager()

        # Init memory pool and attention backends
        # 按请求/Token 上限预先分配 KV 缓存池
        self.init_memory_pool(
            min_per_gpu_memory,
            server_args.max_running_requests,
            server_args.max_total_tokens,
        )
        if self.device == "cuda":
            # CUDA 路径需要初始化 cuBLAS 句柄、注意力内核以及 CUDA 图
            # 先跑一遍微型 GEMM 预热 cublas，避免首次真正调用时阻塞或报错
            self.init_cublas()
            self.init_attention_backend()
            self.init_cuda_graphs()
        else:
            self.cuda_graph_runner = None
            self.cuda_graph_mem_usage = 0
            self.init_attention_backend()

        # auxiliary hidden capture mode. TODO: expose this to server args?
        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
            # Eagle3 需要挂载辅助草稿头，对应的配置信息需单独加载
            # load draft config
            draft_model_config = ModelConfig.from_server_args(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                is_draft_model=True,
            )

            try:
                # get the aux layer from draft model config
                eagle_config = getattr(
                    draft_model_config.hf_config, "eagle_config", None
                )
                eagle_aux_hidden_state_layer_ids = eagle_config[
                    "eagle_aux_hidden_state_layer_ids"
                ]
            except:
                # if there is no aux layer, set to None
                eagle_aux_hidden_state_layer_ids = None

            self.model.set_eagle3_layers_to_capture(eagle_aux_hidden_state_layer_ids)

    def model_specific_adjustment(self):
        server_args = self.server_args

        if (
            server_args.attention_backend == "intel_amx"
            and server_args.device == "cpu"
            and not _is_cpu_amx_available
        ):
            logger.info(
                "The current platform does not support Intel AMX, will fallback to torch_native backend."
            )
            server_args.attention_backend = "torch_native"

        if server_args.prefill_attention_backend is not None and (
            server_args.prefill_attention_backend
            == server_args.decode_attention_backend
        ):  # override the default attention backend
            server_args.attention_backend = server_args.prefill_attention_backend

        if server_args.attention_backend is None:
            """
            Auto select the fastest attention backend.

            1. Models with MHA Architecture (e.g: Llama, QWen)
                1.1 We will turn on FA3 on hopper unless user use spec decode with topk > 1 or page_size > 1.
                1.2 In other cases, we will use flashinfer if available, otherwise use triton.
            2. Models with MLA Architecture and using FA3
                2.1 We will use FA3 backend on hopper.
                2.2 We will use Flashinfer backend on blackwell.
                2.3 Otherwise, we will use triton backend.
            """

            if not self.use_mla_backend:
                # MHA architecture
                if (
                    is_hopper_with_cuda_12_3()
                    and is_no_spec_infer_or_topk_one(server_args)
                    and is_fa3_default_architecture(self.model_config.hf_config)
                ):
                    server_args.attention_backend = "fa3"
                elif _is_hip:
                    server_args.attention_backend = "aiter"
                elif _is_npu:
                    server_args.attention_backend = "ascend"
                else:
                    server_args.attention_backend = (
                        "flashinfer" if is_flashinfer_available() else "triton"
                    )
            else:
                # MLA architecture
                if is_hopper_with_cuda_12_3():
                    server_args.attention_backend = "fa3"
                elif is_sm100_supported():
                    server_args.attention_backend = "flashinfer"
                elif _is_hip:
                    head_num = self.model_config.get_num_kv_heads(self.tp_size)
                    # TODO current aiter only support head number 16 or 128 head number
                    if (
                        head_num == 128 or head_num == 16
                    ) and self.spec_algorithm.is_none():
                        server_args.attention_backend = "aiter"
                    else:
                        server_args.attention_backend = "triton"
                elif _is_npu:
                    server_args.attention_backend = "ascend"
                else:
                    server_args.attention_backend = "triton"
            logger.info(
                f"Attention backend not explicitly specified. Use {server_args.attention_backend} backend by default."
            )
        elif self.use_mla_backend:
            if server_args.device != "cpu":
                if server_args.attention_backend in [
                    "aiter",
                    "flashinfer",
                    "fa3",
                    "triton",
                    "flashmla",
                    "cutlass_mla",
                    "trtllm_mla",
                    "ascend",
                ]:
                    logger.info(
                        f"MLA optimization is turned on. Use {server_args.attention_backend} backend."
                    )
                else:
                    raise ValueError(
                        f"Invalid attention backend for MLA: {server_args.attention_backend}"
                    )
            else:
                if server_args.attention_backend != "intel_amx":
                    raise ValueError(
                        "MLA optimization not supported on CPU except for intel_amx backend."
                    )

        if (
            server_args.attention_backend == "fa3"
            and server_args.kv_cache_dtype == "fp8_e5m2"
        ):
            logger.warning(
                "FlashAttention3 only supports fp8_e4m3 if using FP8; "
                "Setting attention backend to triton."
            )
            server_args.attention_backend = "triton"

        if server_args.enable_double_sparsity:
            logger.info(
                "Double sparsity optimization is turned on. Use triton backend without CUDA graph."
            )
            server_args.attention_backend = "triton"
            server_args.disable_cuda_graph = True
            if server_args.ds_heavy_channel_type is None:
                raise ValueError(
                    "Please specify the heavy channel type for double sparsity optimization."
                )
            self.init_double_sparsity_channel_config(server_args.ds_heavy_channel_type)

        if self.is_multimodal:
            if not self.is_multimodal_chunked_prefill_supported:
                server_args.chunked_prefill_size = -1
                logger.info(
                    f"Automatically turn off --chunked-prefill-size as it is not supported for "
                    f"{self.model_config.hf_config.model_type}"
                )

        if not self.use_mla_backend:
            server_args.disable_chunked_prefix_cache = True
        elif self.page_size > 1:
            logger.info("Disable chunked prefix cache when page size > 1.")
            server_args.disable_chunked_prefix_cache = True

        if not server_args.disable_chunked_prefix_cache:
            logger.info("Chunked prefix cache is turned on.")

        if server_args.attention_backend == "aiter":
            if self.model_config.context_len > 8192:
                self.mem_fraction_static *= 0.85

        if (
            server_args.enable_hierarchical_cache
            and server_args.hicache_io_backend == "kernel"
        ):
            # fix for the compatibility issue with FlashAttention3 decoding and HiCache kernel backend
            if server_args.decode_attention_backend is None:
                if not self.use_mla_backend:
                    server_args.decode_attention_backend = (
                        "flashinfer" if is_flashinfer_available() else "triton"
                    )
                else:
                    server_args.decode_attention_backend = (
                        "flashinfer" if is_sm100_supported() else "triton"
                    )
            elif server_args.decode_attention_backend == "fa3":
                server_args.hicache_io_backend = "direct"
                logger.warning(
                    "FlashAttention3 decode backend is not compatible with hierarchical cache. "
                    f"Setting hicache_io_backend to vanilla I/O, which may lead to suboptimal performance with small page sizes."
                )

    def init_torch_distributed(self):
        logger.info("Init torch distributed begin.")

        try:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.tp_rank=} {self.tp_size=}"
            )
            raise

        if self.device == "cuda":
            backend = "nccl"
        elif self.device == "xpu":
            backend = "xccl"
        elif self.device == "hpu":
            backend = "hccl"
        elif self.device == "cpu":
            backend = "gloo"
        elif self.device == "npu":
            backend = "hccl"
        # 根据运行设备挑选通信后端，避免初始化到不支持的 pg 实现

        # 初始化前记录一次显存占用，方便推断通信栈带来的开销
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            # 某些环境下 P2P 检查会误判连接性，这里允许通过参数跳过
            monkey_patch_p2p_access_check()

        # 优先使用用户显式指定的初始化地址，缺省时回退到本地端口
        if self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"
        # 启动 torch.distributed 之前先配置自定义 allreduce，以便复用高效通信核
        # 通过环境变量切换自定义 AllReduce，可在不同通信栈间自由选择
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
        # MSCCl++/自定义 AllReduce 会在初始化后覆盖默认通信算子，降低跨卡同步开销

        if not self.is_draft_worker:
            if self.device == "cpu":
                if _is_cpu_amx_available:
                    # Bind OpenMP threads to CPU cores
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

                    # Set local size to hint SGLang to use shared memory based AllReduce
                    os.environ["LOCAL_SIZE"] = str(self.tp_size)
                    torch.ops.sgl_kernel.initialize(self.tp_size, self.tp_rank)
                else:
                    logger.warning(
                        "init_cpu_threads_env and shared memory based AllReduce is disabled since intel amx backend is not available"
                    )

            # Only initialize the distributed environment on the target model worker.
            # 构造张量并行 rank 对应的 Torch 分布式通信上下文
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
            )
            # 初始化张量/流水线/专家并行组，建立多维度并行拓扑
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                pipeline_model_parallel_size=self.pp_size,
                expert_model_parallel_size=self.moe_ep_size,
                duplicate_tp_group=self.server_args.enable_pdmux,
            )
            # DP attention 在 TP 组内再细分 token 负载，实现注意力层数据并行
            initialize_dp_attention(
                enable_dp_attention=self.server_args.enable_dp_attention,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                dp_size=self.server_args.dp_size,
                moe_dense_tp_size=self.server_args.moe_dense_tp_size,
                pp_size=self.server_args.pp_size,
            )

        min_per_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        # 记录启用通信后集群可见的最小显存，作为后续容量校验的基准值
        self.tp_group = get_tp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1 and not self.is_draft_worker:
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                if get_bool_env_var("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"):
                    logger.warning(
                        "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                        f"{min_per_gpu_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                    )
                else:
                    raise ValueError(
                        "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                        f"{min_per_gpu_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                    )

        logger.info(
            f"Init torch distributed ends. mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return min_per_gpu_memory

    def load_model(self):
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            # 加载权重时关闭算子级并行，避免 I/O 竞争
            torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                # 旧架构缺少 BF16 与高效张量核，只能退回 FP16 内核
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
        )
        # LoadConfig 整合资源位置和动态参数，便于 model loader 统一访问
        if self.device == "cpu":
            # CPU 路径下需要根据张量并行大小对权重切分参数进行再对齐
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.tp_size
            )
        if self.server_args.load_format == "gguf":
            monkey_patch_vllm_gguf_config()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        # 临时复用 vLLM 的辅助方法，以便正确加载量化权重
        monkey_patch_vllm_parallel_state()
        monkey_patch_isinstance_for_vllm_base_layer()

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_WEIGHTS):
            # 该内存保护区会在加载前锁定权重预算，避免与运行时缓存抢占
            # get_model 会真正构建 HuggingFace/sglang 的模型实例
            self.model = get_model(
                model_config=self.model_config,
                load_config=self.load_config,
                device_config=DeviceConfig(self.device),
            )
        # 加载结束后恢复 vLLM 的补丁，避免影响后续流程
        monkey_patch_vllm_parallel_state(reverse=True)
        monkey_patch_isinstance_for_vllm_base_layer(reverse=True)

        if self.server_args.kv_cache_dtype == "fp8_e4m3":
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        self.server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        self.sliding_window_size = None
        if hasattr(self.model, "get_attention_sliding_window_size"):
            # 优先采用模型给出的滑动窗口设置（如 GAU/GQA）
            self.sliding_window_size = self.model.get_attention_sliding_window_size()
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size
            logger.info(
                f"Setting sliding_window_size to be attention_chunk_size: {self.sliding_window_size}"
            )

        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        # 记录权重加载过程实际消耗的显存，以便后续动态判断缓存余量
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        logger.info(
            f"Load weight end. "
            f"type={type(self.model).__name__}, "
            f"dtype={self.dtype}, "
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={self.weight_load_mem_usage:.2f} GB."
        )

        # Handle the case where some ranks do not finish loading.
        try:
            # 等待所有张量并行 rank 完成权重加载，防止部分卡提前进入推理
            dist.monitored_barrier(
                group=get_tp_group().cpu_group,
                timeout=datetime.timedelta(seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S),
                wait_all_ranks=True,
            )
        except RuntimeError:
            # 当本 rank 成功、其他 rank 失败时给出更明确的错误信息
            raise ValueError(
                f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
            ) from None

    def update_expert_location(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        self.expert_location_updater.update(
            self.model.routed_experts_weights_of_layer,
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=self.server_args.nnodes,
            rank=self.tp_rank,
        )

    def update_weights_from_disk(
        self, model_path: str, load_format: str
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
            )
            return iter

        def model_load_weights(model, iter):
            DefaultModelLoader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, f"Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
        load_format: Optional[str] = None,
    ):
        monkey_patch_torch_reductions()
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        infered_device = torch.cuda.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format in self.server_args.custom_weight_loader:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
        )

    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface."""

        logger.info(
            f"LoRA adapter loading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.load_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter loading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    def unload_lora_adapter(self, lora_ref: LoRARef):
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading."""

        logger.info(
            f"LoRA adapter unloading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.unload_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter unloading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    def profile_max_num_token(self, total_gpu_memory: int):
        available_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        if self.is_draft_worker:
            num_layers = getattr(
                self.model_config.hf_config,
                "num_nextn_predict_layers",
                self.num_effective_layers,
            )
        else:
            num_layers = self.num_effective_layers
        if self.use_mla_backend:
            # FIXME: pipeline parallelism is not compatible with mla backend
            assert self.pp_size == 1
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * num_layers
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        else:
            cell_size = (
                self.model_config.get_num_kv_heads(get_attention_tp_size())
                * self.model_config.head_dim
                * num_layers
                * 2
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        # 将静态划定给权重的显存从当前可用显存中扣除，只留下运行时真正能用的空间
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - self.mem_fraction_static
        )
        # 单 token 占用的 KV 空间为 cell_size 字节，除法得到理论可容纳的最大 token 数
        max_num_token = int(rest_memory * (1 << 30) // cell_size)
        return max_num_token

    def set_num_token_hybrid(self):
        if (
            "Llama4ForConditionalGeneration"
            in self.model_config.hf_config.architectures
        ):
            temp_ratio = (
                (1 - self.is_hybrid)
                + self.is_hybrid
                * self.attention_chunk_size
                / self.model_config.context_len
            )
            self.swa_max_total_num_tokens = (
                4 * self.max_total_num_tokens * temp_ratio // (3 * temp_ratio + 1)
            )
            self.full_max_total_num_tokens = (
                4 * self.max_total_num_tokens
                - 12 * self.max_total_num_tokens * temp_ratio // (3 * temp_ratio + 1)
            )
            self.swa_max_total_num_tokens = int(
                self.swa_max_total_num_tokens
                // self.server_args.page_size
                * self.server_args.page_size
            )
            self.full_max_total_num_tokens = int(
                self.full_max_total_num_tokens
                // self.server_args.page_size
                * self.server_args.page_size
            )
            self.max_total_num_tokens = self.full_max_total_num_tokens
        else:
            assert self.sliding_window_size is not None and self.sliding_window_size > 0
            full_attention_layer_ids = []
            swa_attention_layer_ids = []

            try:
                layers = self.model.model.layers
            except:
                try:
                    layers = self.model.language_model.model.layers
                except:
                    try:
                        layers = self.model.language_model.layers
                    except:
                        self.is_hybrid = False
                        return

            for layer in layers:
                if (
                    layer.self_attn.attn.sliding_window_size is None
                    or layer.self_attn.attn.sliding_window_size == -1
                ):
                    full_attention_layer_ids.append(layer.layer_id)
                else:
                    swa_attention_layer_ids.append(layer.layer_id)
            self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
            self.model_config.full_attention_layer_ids = full_attention_layer_ids

            # Algorithm:
            # Existing max_total_num_tokens is per layer and assume all layers have the same number of tokens.
            # - Find total # of tokens available across layers.
            # - Calculate full_max_total_num_tokens and swa_max_total_num_tokens based on the given swa_full_tokens_ratio.
            total_tokens = (
                self.max_total_num_tokens * self.model_config.num_hidden_layers
            )
            full_layers_num = len(full_attention_layer_ids)
            swa_layers_num = len(swa_attention_layer_ids)
            swa_full_tokens_ratio = self.server_args.swa_full_tokens_ratio

            # Solve the equations:
            # 1. swa_max_total_num_tokens * swa_layers_num + full_max_total_num_tokens * full_layers_num == total_tokens
            # 2. full_max_total_num_tokens * swa_full_tokens_ratio == swa_max_total_num_tokens
            denominator = swa_full_tokens_ratio * swa_layers_num + full_layers_num
            self.full_max_total_num_tokens = int(total_tokens / denominator)
            self.swa_max_total_num_tokens = int(
                self.full_max_total_num_tokens * swa_full_tokens_ratio
            )
            self.max_total_num_tokens = self.full_max_total_num_tokens

            logger.info(
                f"Use Sliding window memory pool. full_layer_tokens={self.full_max_total_num_tokens}, swa_layer_tokens={self.swa_max_total_num_tokens}"
            )

    def init_memory_pool(
        self,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = torch.float8_e5m2fnuz
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = torch.float8_e4m3fnuz
            else:
                self.kv_cache_dtype = torch.float8_e4m3fn
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

        # 根据显存上限估算本 worker 能承载的 token 数量，后续调度与缓存均以此为基准
        self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)

        if max_num_reqs is None:
            # 根据上下文长度反推请求上限，既避免稀疏时浪费缓存，也防止批次过多拖垮调度
            max_num_reqs = min(
                max(
                    int(
                        self.max_total_num_tokens / self.model_config.context_len * 512
                    ),
                    2048,
                ),
                4096,
            )

        if SGLANG_CI_SMALL_KV_SIZE:
            self.max_total_num_tokens = int(SGLANG_CI_SMALL_KV_SIZE)

        if not self.spec_algorithm.is_none():
            if self.is_draft_worker:
                # 草稿分支单独限制 KV 大小，避免与主模型互相挤占
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                max_num_reqs = self.server_args.max_num_reqs
            else:
                # We are sharing the `token_to_kv_pool`, and both verify and draft tokens
                # can be concurrently allocated, so we should give a headroom for it.
                # 这里为 draft / verify 预留缓冲数，防止推测解码途中出现 KV 缺页
                self.server_args.draft_runner_cache_size = (
                    self.max_total_num_tokens
                    # draft
                    + max_num_reqs
                    * self.server_args.speculative_num_steps
                    * self.server_args.speculative_eagle_topk
                    # verify
                    + max_num_reqs * self.server_args.speculative_num_draft_tokens
                    # buffer
                    + 100
                )
                # Target worker and draft worker shares the same indices for the
                # token_to_kv_pool, so we should make sure to match max_total_num_tokens.
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                self.server_args.max_num_reqs = max_num_reqs

        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logging.warning(
                    f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                    f"{self.max_total_num_tokens}. "
                    f"Use the profiled value instead."
                )
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)

        self.max_total_num_tokens = (
            self.max_total_num_tokens
            // self.server_args.page_size
            * self.server_args.page_size
        )
        # 将可用 token 数截断到 page_size 的整数倍，方便分页分配与回收
        # create token size for hybrid cache
        if self.is_hybrid:
            self.set_num_token_hybrid()

        if self.max_total_num_tokens <= 0:
            raise RuntimeError(
                "Not enough memory. Please try to increase --mem-fraction-static."
            )

        if self.req_to_token_pool is None:
            if self.server_args.disaggregation_mode == "decode":
                from sglang.srt.disaggregation.decode import DecodeReqToTokenPool

                # subscribe memory for pre-allocated requests
                # if max_num_reqs <= 32, we pre-allocate 2x requests
                # 预先分配额外请求槽位，降低高并发时的申请开销
                pre_alloc_size = max_num_reqs * 2 if max_num_reqs <= 32 else 0
                self.req_to_token_pool = DecodeReqToTokenPool(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len + 4,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    pre_alloc_size=pre_alloc_size,
                )
            else:
                self.req_to_token_pool = ReqToTokenPool(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len + 4,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                )
        else:
            # Draft worker shares req_to_token_pool with the target worker.
            assert self.is_draft_worker

        if self.server_args.attention_backend == "ascend":
            if self.use_mla_backend:
                self.token_to_kv_pool = AscendMLAPagedTokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                self.token_to_kv_pool = AscendTokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.model_config.num_hidden_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                )
        elif self.use_mla_backend:
            self.token_to_kv_pool = MLATokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )
        elif self.server_args.enable_double_sparsity:
            self.token_to_kv_pool = DoubleSparseTokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                heavy_channel_num=self.server_args.ds_heavy_channel_num,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )
        else:
            if self.is_hybrid:
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    enable_kvcache_transpose=False,
                    device=self.device,
                )
            else:
                self.token_to_kv_pool = MHATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )

        need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
        if self.token_to_kv_pool_allocator is None:
            if self.page_size == 1:
                if self.is_hybrid:
                    self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
            else:
                if not _is_npu:
                    self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    self.token_to_kv_pool_allocator = AscendPagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
        else:
            assert self.is_draft_worker

        # 初始化完成后记录剩余显存，便于排查内存碎片或空间不足问题
        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        if self.server_args.enable_two_batch_overlap and not self.is_draft_worker:
            self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
        else:
            self.attn_backend = self._get_attention_backend()

    def _get_attention_backend(self):
        """Init attention kernel backend."""
        self.decode_attention_backend_str = (
            self.server_args.decode_attention_backend
            if self.server_args.decode_attention_backend
            else self.server_args.attention_backend
        )
        self.prefill_attention_backend_str = (
            self.server_args.prefill_attention_backend
            if self.server_args.prefill_attention_backend
            else self.server_args.attention_backend
        )
        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            assert (
                self.server_args.speculative_algorithm is None
            ), "Currently HybridAttentionBackend does not support speculative decoding."
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                f"Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                f"The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend
            )

        global_server_args_dict.update(
            {
                "decode_attention_backend": self.decode_attention_backend_str,
                "prefill_attention_backend": self.prefill_attention_backend_str,
            }
        )
        return attn_backend

    def _get_attention_backend_from_str(self, backend_str: str):
        if backend_str == "flashinfer":
            if not self.use_mla_backend:
                from sglang.srt.layers.attention.flashinfer_backend import (
                    FlashInferAttnBackend,
                )

                # Init streams
                if self.server_args.speculative_algorithm == "EAGLE":
                    if (
                        not hasattr(self, "plan_stream_for_flashinfer")
                        or not self.plan_stream_for_flashinfer
                    ):
                        self.plan_stream_for_flashinfer = torch.cuda.Stream()
                return FlashInferAttnBackend(self)
            else:
                from sglang.srt.layers.attention.flashinfer_mla_backend import (
                    FlashInferMLAAttnBackend,
                )

                return FlashInferMLAAttnBackend(self)
        elif backend_str == "aiter":
            from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

            return AiterAttnBackend(self)
        elif backend_str == "ascend":
            from sglang.srt.layers.attention.ascend_backend import AscendAttnBackend

            return AscendAttnBackend(self)
        elif backend_str == "triton":
            assert not self.model_config.is_encoder_decoder, (
                "Cross attention is not supported in the triton attention backend. "
                "Please use `--attention-backend flashinfer`."
            )
            if self.server_args.enable_double_sparsity:
                from sglang.srt.layers.attention.double_sparsity_backend import (
                    DoubleSparseAttnBackend,
                )

                return DoubleSparseAttnBackend(self)
            else:
                from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

                return TritonAttnBackend(self)
        elif backend_str == "torch_native":
            from sglang.srt.layers.attention.torch_native_backend import (
                TorchNativeAttnBackend,
            )

            return TorchNativeAttnBackend(self)
        elif backend_str == "flashmla":
            from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

            return FlashMLABackend(self)
        elif backend_str == "fa3":
            assert (
                torch.cuda.get_device_capability()[0] == 8 and not self.use_mla_backend
            ) or torch.cuda.get_device_capability()[0] == 9, (
                "FlashAttention v3 Backend requires SM>=80 and SM<=90. "
                "Please use `--attention-backend flashinfer`."
            )
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )

            return FlashAttentionBackend(self)
        elif backend_str == "cutlass_mla":
            from sglang.srt.layers.attention.cutlass_mla_backend import (
                CutlassMLABackend,
            )

            return CutlassMLABackend(self)
        elif backend_str == "trtllm_mla":
            if not self.use_mla_backend:
                raise ValueError("trtllm_mla backend can only be used with MLA models.")
            from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

            return TRTLLMMLABackend(self)
        elif backend_str == "trtllm_mha":
            if self.use_mla_backend:
                raise ValueError(
                    "trtllm_mha backend can only be used with non-MLA models."
                )
            from sglang.srt.layers.attention.trtllm_mha_backend import (
                TRTLLMHAAttnBackend,
            )

            return TRTLLMHAAttnBackend(self)

        elif backend_str == "intel_amx":
            from sglang.srt.layers.attention.intel_amx_backend import (
                IntelAMXAttnBackend,
            )

            logger.info(f"Intel AMX attention backend is enabled.")
            return IntelAMXAttnBackend(self)
        elif self.server_args.attention_backend == "dual_chunk_flash_attn":
            from sglang.srt.layers.attention.dual_chunk_flashattention_backend import (
                DualChunkFlashAttentionBackend,
            )

            return DualChunkFlashAttentionBackend(self)
        else:
            raise ValueError(f"Invalid attention backend: {backend_str}")

    def init_double_sparsity_channel_config(self, selected_channel):
        selected_channel = "." + selected_channel + "_proj"
        self.sorted_channels = []
        # load channel config
        with open(self.server_args.ds_channel_config_path, "r") as f:
            channel_config = json.load(f)

        for i in range(self.start_layer, self.end_layer):
            key = "model.layers." + str(i) + ".self_attn" + selected_channel
            self.sorted_channels.append(
                torch.tensor(channel_config[key])[
                    :, : self.server_args.ds_heavy_channel_num
                ]
                .contiguous()
                .cuda()
            )

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_mem_usage = 0

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.disable_cuda_graph:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        # 捕获阶段会走一遍解码计算图，并将 kernel/参数固化为 CUDA Graph
        self.cuda_graph_runner = CudaGraphRunner(self)
        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.cuda_graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.cuda_graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_threads_binding(self):
        omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
        if omp_cpuids == "all":
            cpu_ids_by_node = get_cpu_ids_by_node()
            n_numa_node = len(cpu_ids_by_node)

            assert self.tp_size <= n_numa_node, (
                f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
                f"tp_size {self.tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
                f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
                f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
                f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
                f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
                f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "

                f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
            )
            if self.tp_size < n_numa_node:
                logger.warning(

                    f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {self.tp_size}, so only {self.tp_size} numa nodes are used."
                )
            self.local_omp_cpuid = cpu_ids_by_node[self.tp_rank]
        else:
            self.local_omp_cpuid = omp_cpuids.split("|")[self.tp_rank]

    def apply_torch_tp(self):
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            self.attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            self.attn_backend.init_forward_metadata(forward_batch)

        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if not self.is_generation:
            kwargs["get_embedding"] = True
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> LogitsProcessorOutput:
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        ret = self.model.forward_split_prefill(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            (forward_batch.split_index, next_split_index),
        )
        forward_batch.split_index = next_split_index
        return ret

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> Tuple[Union[LogitsProcessorOutput, PPProxyTensors], bool]:
        self.forward_pass_id += 1

        # 记录一次前向的信息，便于在专家并行场景统计路由负载情况
        with get_global_expert_distribution_recorder().with_forward_pass(
            self.forward_pass_id,
            forward_batch,
        ):
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        return output

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> Tuple[Union[LogitsProcessorOutput, PPProxyTensors], bool]:
        can_run_cuda_graph = bool(
            forward_batch.forward_mode.is_cuda_graph()
            and self.cuda_graph_runner
            and self.cuda_graph_runner.can_run(forward_batch)
        )
        if can_run_cuda_graph:
            # 能命中 CUDA Graph 时直接复用已捕获的执行图，跳过 Python 调度与 kernel 选择
            ret = self.cuda_graph_runner.replay(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return ret, can_run_cuda_graph

        # For MLP sync
        if forward_batch.global_num_tokens_cpu is not None:
            # MLP 同步模式下需要提前重排 batch，保证各 rank token 对齐
            forward_batch.prepare_mlp_sync_batch(self)

        if forward_batch.forward_mode.is_decode():
            ret = self.forward_decode(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode.is_extend():
            ret = self.forward_extend(
                forward_batch,
                skip_attn_backend_init=skip_attn_backend_init,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        elif forward_batch.forward_mode.is_split_prefill():
            ret = self.forward_split_prefill(
                forward_batch,
                reinit_attn_backend=reinit_attn_backend,
                forward_count=split_forward_count,
            )
        elif forward_batch.forward_mode.is_idle():
            ret = self.forward_idle(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.post_forward_mlp_sync_batch(ret)

        return ret, can_run_cuda_graph

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # Apply logit bias
        if sampling_info.sampling_info_done:
            # Overlap mode: the function update_regex_vocab_mask was executed
            # in process_batch_result of the last batch.
            if sampling_info.grammars:
                # 存在约束语法时等待上一次 forward 中的后台线程完成掩码计算
                sampling_info.sampling_info_done.wait()
        else:
            # Normal mode: Put CPU-heavy tasks here. They will be overlapped with the forward pass.
            # 将正则/JSON Schema 的词表裁剪放在采样前执行，以掩盖 CPU 延迟
            sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        # For duplex models with multiple output streams.
        if isinstance(logits_output, tuple):
            # 双输出流（如多模态、草稿头）需要对每个分支单独采样再拼接
            return torch.stack(
                [self.sample(values, forward_batch) for values in logits_output],
                axis=-1,
            )

        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )
        return next_token_ids

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        rope_scaling = getattr(self.model_config.hf_text_config, "rope_scaling", {})
        if rope_scaling is None:
            return False
        is_mrope_enabled = "mrope_section" in rope_scaling
        return is_mrope_enabled

    def save_remote_model(self, url: str):
        from sglang.srt.model_loader.loader import RemoteModelLoader

        logger.info(f"Saving model to {url}")
        RemoteModelLoader.save_model(self.model, self.model_config.model_path, url)

    def save_sharded_model(
        self, path: str, pattern: Optional[str] = None, max_size: Optional[int] = None
    ):
        from sglang.srt.model_loader.loader import ShardedStateLoader

        logger.info(
            f"Save sharded model to {path} with pattern {pattern} and max_size {max_size}"
        )
        ShardedStateLoader.save_model(self.model, path, pattern, max_size)


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
