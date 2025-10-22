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
"""A tensor parallel worker."""

import dataclasses
import logging
import signal
import threading
from queue import Queue
from typing import Optional, Tuple

import psutil
import torch

from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import DynamicGradMode, get_compiler_backend
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


@torch.compile(dynamic=True, backend=get_compiler_backend())
def resolve_future_token_ids(input_ids, future_token_ids_map):
    # 将占位的负值 token 替换为前序批次推测出的真实 token id
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


class TpModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
    ):
        # Load the model
        self.worker = TpModelWorker(
            server_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, dp_rank, nccl_port
        )
        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device
        self.gpu_id = gpu_id

        # Init future mappings
        self.future_token_ids_ct = 0
        # limit 控制环形缓冲区长度，避免 speculative token 无限增长
        self.future_token_ids_limit = self.max_running_requests * 3
        # future_token_ids_map 保存最近批次生成的 token，供下游请求引用
        self.future_token_ids_map = torch.empty(
            (self.max_running_requests * 5,), dtype=torch.int64, device=self.device
        )
        # 以 ring buffer 方式复用 future token 保存窗口

        # Launch threads
        self.input_queue = Queue()
        # 调度线程异步投递待执行批次，前向线程从该队列取任务
        self.output_queue = Queue()
        self.forward_stream = torch.get_device_module(self.device).Stream()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        # 单独线程在独立 CUDA stream 中执行模型前向，避免 scheduler 被阻塞
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()
        self.scheduler_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.scheduler_stream.synchronize = lambda: None  # No-op for CPU

        self.hicache_layer_transfer_counter = None

    def register_hicache_layer_transfer_counter(self, counter):
        self.hicache_layer_transfer_counter = counter

    def set_hicache_consumer(self, consumer_index):
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_tokens_per_layer_info(self):
        return self.worker.get_tokens_per_layer_info()

    @property
    def sliding_window_size(self) -> Optional[int]:
        return self.worker.sliding_window_size

    @property
    def is_hybrid(self) -> bool:
        return self.worker.is_hybrid

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_tp_group(self):
        return self.worker.get_tp_group()

    def get_attention_tp_group(self):
        return self.worker.get_attention_tp_group()

    def get_attention_tp_cpu_group(self):
        return self.worker.get_attention_tp_cpu_group()

    def get_memory_pool(self):
        return (
            self.worker.model_runner.req_to_token_pool,
            self.worker.model_runner.token_to_kv_pool_allocator,
        )

    def get_kv_cache(self):
        return self.worker.model_runner.token_to_kv_pool

    def forward_thread_func(self):
        try:
            with torch.get_device_module(self.device).stream(self.forward_stream):
                # overlap 模式下使用独立 stream 避免阻塞调度线程
                self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error(f"TpModelWorkerClient hit an exception: {traceback}")
            self.parent_process.send_signal(signal.SIGQUIT)

    @DynamicGradMode()
    def forward_thread_func_(self):
        batch_pt = 0
        batch_lists = [None] * 2

        while True:
            model_worker_batch, future_token_ids_ct, sync_event = self.input_queue.get()
            if not model_worker_batch:
                break

            sync_event.wait()

            # Keep a reference of model_worker_batch by storing it into a list.
            # Otherwise, the tensor members of model_worker_batch will be released
            # by pytorch and cause CUDA illegal memory access errors.
            # 双缓冲记录最近两个 batch，防止 PyTorch 提前释放 Tensor
            batch_lists[batch_pt % 2] = model_worker_batch
            batch_pt += 1

            # Create event
            copy_done = torch.get_device_module(self.device).Event()

            # Resolve future tokens in the input
            input_ids = model_worker_batch.input_ids
            # 将负索引替换为已生成的 token，实现 speculative overlap
            resolve_future_token_ids(input_ids, self.future_token_ids_map)

            # update the consumer index of hicache to the running batch
            self.set_hicache_consumer(model_worker_batch.hicache_consumer_index)
            # Run forward
            logits_output, next_token_ids, can_run_cuda_graph = (
                self.worker.forward_batch_generation(
                    model_worker_batch, model_worker_batch.launch_done
                )
            )

            # Update the future token ids map
            bs = len(model_worker_batch.seq_lens)
            # 预先保留 0 号位置作为对齐空位，后续请求如引用负索引直接写入该窗口
            self.future_token_ids_map[
                future_token_ids_ct + 1 : future_token_ids_ct + bs + 1
            ] = next_token_ids
            # 这样下游请求只需要携带负索引即可引用上一批生成的 token

            # Copy results to the CPU
            if model_worker_batch.return_logprob:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.to("cpu", non_blocking=True)
                )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = (
                        logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                    )
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states.to(
                    "cpu", non_blocking=True
                )
            next_token_ids = next_token_ids.to("cpu", non_blocking=True)
            copy_done.record()

            # 将结果和 event 打包回主线程，等待同步后消费
            self.output_queue.put(
                (copy_done, logits_output, next_token_ids, can_run_cuda_graph)
            )

    def resolve_last_batch_result(self, launch_done: Optional[threading.Event] = None):
        """
        This function is called to resolve the last batch result and
        wait for the current batch to be launched. Used in overlap mode.
        """
        copy_done, logits_output, next_token_ids, can_run_cuda_graph = (
            self.output_queue.get()
        )

        if launch_done is not None:
            # launch_done 确保调度线程已提交下一批 CUDA 工作
            launch_done.wait()
        # 等待 copy_done，保证 GPU->CPU 拷贝已经完成再交还给调度线程
        copy_done.synchronize()

        if logits_output.next_token_logprobs is not None:
            logits_output.next_token_logprobs = (
                logits_output.next_token_logprobs.tolist()
            )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )
        next_token_ids = next_token_ids.tolist()
        return logits_output, next_token_ids, can_run_cuda_graph

    def forward_batch_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> Tuple[None, torch.Tensor, bool]:
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        # 克隆一份采样配置，避免调度线程修改原对象造成竞态
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

        # A cuda stream sync here to avoid the cuda illegal memory access error.
        sync_event = torch.get_device_module(self.device).Event()
        # 记录当前调度 stream 的同步点，确保输入准备完成
        sync_event.record(self.scheduler_stream)

        # Push a new batch to the queue
        self.input_queue.put((model_worker_batch, self.future_token_ids_ct, sync_event))

        # Allocate output future objects
        bs = len(model_worker_batch.seq_lens)
        future_next_token_ids = torch.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        # 负索引区间在下一批 resolve_future_token_ids 时被回填
        self.future_token_ids_ct = (
            self.future_token_ids_ct + bs
        ) % self.future_token_ids_limit
        # 通过环形计数器循环利用映射数组，避免长期增长导致内存浪费
        return None, future_next_token_ids, False

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.worker.update_weights_from_disk(recv_req)
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.worker.init_weights_update_group(recv_req)
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.worker.update_weights_from_distributed(recv_req)
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        success, message = self.worker.update_weights_from_tensor(recv_req)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        return self.worker.get_weights_by_name(recv_req)

    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        return self.worker.load_lora_adapter(recv_req)

    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        return self.worker.unload_lora_adapter(recv_req)

    def can_run_lora_batch(self, lora_ids: list[str]) -> bool:
        return self.worker.can_run_lora_batch(lora_ids)

    def __delete__(self):
        # 通知前向线程退出；copy_queue 仅在旧实现使用，保留兼容逻辑
        self.input_queue.put((None, None))
        self.copy_queue.put((None, None, None))
