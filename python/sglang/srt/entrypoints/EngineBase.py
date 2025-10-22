"""
SGLang引擎基类模块
这个模块定义了SGLang推理引擎的抽象基类，提供了统一的API接口。

EngineBase是所有推理引擎的基础类，定义了引擎必须实现的核心方法，
包括文本生成、权重更新、内存控制等功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch


class EngineBase(ABC):
    """
    引擎接口的抽象基类，支持生成、权重更新和内存控制功能。

    这个基类为基于HTTP的引擎和直接引擎提供了统一的API接口，
    确保所有引擎实现都具有一致的行为和接口。
    """

    @abstractmethod
    def generate(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        image_data: Optional[Union[List[str], str]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: Optional[bool] = None,
        stream: Optional[bool] = None,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        data_parallel_rank: Optional[int] = None,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        基于给定输入生成输出

        这是引擎的核心方法，负责执行文本生成任务。
        支持单次和批量生成，以及流式和非流式输出。

        参数:
        prompt: 输入提示文本，可以是字符串或字符串列表
        sampling_params: 采样参数，控制生成的质量和随机性
        input_ids: 输入token ID，用于直接指定token序列
        image_data: 图像数据，支持多模态输入
        return_logprob: 是否返回对数概率信息
        logprob_start_len: 开始计算对数概率的位置
        top_logprobs_num: 返回的top token概率数量
        token_ids_logprob: 需要计算概率的token ID
        lora_path: LoRA适配器路径
        custom_logit_processor: 自定义logit处理器
        return_hidden_states: 是否返回隐藏状态
        stream: 是否流式输出
        bootstrap_host: 引导主机地址
        bootstrap_port: 引导端口
        bootstrap_room: 引导房间ID
        data_parallel_rank: 数据并行排名

        返回:
        Union[Dict, Iterator[Dict]]: 生成结果或流式结果迭代器
        """
        pass

    @abstractmethod
    def flush_cache(self):
        """
        清空引擎缓存

        这个方法用于清空引擎的KV缓存和其他缓存，
        释放内存资源，通常在需要释放内存时调用。
        """
        pass

    @abstractmethod
    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """
        使用内存中的张量数据更新模型权重

        这个方法允许在运行时动态更新模型权重，
        支持模型微调、权重更新等场景。

        参数:
        named_tensors: 命名张量列表，包含权重名称和张量数据
        load_format: 加载格式，指定权重的存储格式
        flush_cache: 是否在更新后清空缓存
        """
        pass

    def load_lora_adapter(self, lora_name: str, lora_path: str):
        """
        加载新的LoRA适配器，无需重新启动引擎

        这个方法允许在运行时动态加载LoRA适配器，
        实现模型的快速适配和个性化。

        参数:
        lora_name: LoRA适配器名称
        lora_path: LoRA适配器文件路径
        """
        pass

    def unload_lora_adapter(self, lora_name: str):
        """
        卸载LoRA适配器，无需重新启动引擎

        这个方法用于卸载不再需要的LoRA适配器，
        释放相关资源。

        参数:
        lora_name: 要卸载的LoRA适配器名称
        """
        pass

    @abstractmethod
    def release_memory_occupation(self):
        """
        临时释放GPU内存占用

        这个方法用于临时释放GPU内存，
        允许其他进程使用这些内存资源。
        通常在需要为其他任务腾出内存时调用。
        """
        pass

    @abstractmethod
    def resume_memory_occupation(self):
        """
        恢复之前释放的GPU内存占用

        这个方法用于恢复之前通过release_memory_occupation
        释放的GPU内存，重新占用这些资源。
        """
        pass

    @abstractmethod
    def shutdown(self):
        """
        关闭引擎并清理资源

        这个方法用于优雅地关闭引擎，
        清理所有资源，包括GPU内存、文件句柄等。
        应该在程序退出前调用以确保资源正确释放。
        """
        pass
