"""
SGLang后端基类模块
这个模块定义了SGLang后端系统的抽象基类，为所有后端实现提供统一的接口规范。

BaseBackend是所有后端实现的基础类，定义了后端必须实现的核心方法，
包括文本生成、缓存管理、程序执行等功能。
"""

from typing import List, Optional, Union

from sglang.lang.chat_template import get_chat_template
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams

class BaseBackend:
    """
    SGLang后端抽象基类

    这个类定义了所有后端实现必须遵循的接口规范，包括：
    - 文本生成功能
    - 缓存管理功能
    - 程序执行控制
    - 资源管理功能

    所有具体的后端实现都应该继承这个基类并实现其抽象方法。
    """

    def __init__(self) -> None:
        """
        初始化后端基类

        设置默认的配置参数和聊天模板。
        """
        self.support_concate_and_append = False
        self.chat_template = get_chat_template("default")

    def get_model_name(self):
        """
        获取模型名称

        返回当前后端使用的模型名称。
        这是一个抽象方法，必须在子类中实现。

        返回:
        str: 模型名称

        异常:
        NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError()

    def get_chat_template(self):
        """
        获取聊天模板

        返回当前后端使用的聊天模板。

        返回:
        ChatTemplate: 聊天模板对象
        """
        return self.chat_template

    def cache_prefix(self, prefix_str: str):
        """
        缓存前缀字符串

        将指定的前缀字符串缓存到后端，用于后续的快速检索和复用。
        这是一个可选操作，默认实现为空操作。

        参数:
        prefix_str: 要缓存的前缀字符串
        """
        pass

    def uncache_prefix(self, rid: str):
        """
        取消缓存前缀

        从后端缓存中移除指定的前缀字符串。
        这是一个可选操作，默认实现为空操作。

        参数:
        rid: 要取消缓存的请求ID
        """
        pass

    def end_request(self, rid: Union[str, List[str]]):
        """
        结束请求

        通知后端结束指定的请求，释放相关资源。
        这是一个可选操作，默认实现为空操作。

        参数:
        rid: 要结束的请求ID，可以是单个ID或ID列表
        """
        pass

    def begin_program(self, s: StreamExecutor):
        """
        开始程序执行

        在程序开始执行前调用，用于初始化后端状态。
        这是一个可选操作，默认实现为空操作。

        参数:
        s: 流执行器实例
        """
        pass

    def end_program(self, s: Union[StreamExecutor, List[StreamExecutor]]):
        """
        结束程序执行

        在程序执行完成后调用，用于清理后端状态。
        这是一个可选操作，默认实现为空操作。

        参数:
        s: 流执行器实例或实例列表
        """
        pass

    def commit_lazy_operations(self, s: StreamExecutor):
        """
        提交延迟操作

        提交所有延迟执行的操作到后端。
        这是一个可选操作，默认实现为空操作。

        参数:
        s: 流执行器实例
        """
        pass

    def fork_program(
        self,
        src: StreamExecutor,
        dst: List[StreamExecutor],
        position_ids_offset: Optional[List[int]] = None,
    ):
        """
        分叉程序执行

        从源执行器分叉出多个目标执行器，用于并行处理。
        这是一个可选操作，默认实现为空操作。

        参数:
        src: 源流执行器
        dst: 目标流执行器列表
        position_ids_offset: 位置ID偏移量列表，可选
        """
        pass

    def fill_image(self, s: StreamExecutor):
        """
        填充图像数据

        将图像数据填充到流执行器中，用于多模态处理。
        这是一个可选操作，默认实现为空操作。

        参数:
        s: 流执行器实例
        """
        pass

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        """
        生成文本

        基于给定的流执行器和采样参数生成文本。
        这是一个抽象方法，必须在子类中实现。

        参数:
        s: 流执行器实例
        sampling_params: 采样参数

        异常:
        NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError()

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        """
        流式生成文本

        基于给定的流执行器和采样参数进行流式文本生成。
        这是一个抽象方法，必须在子类中实现。

        参数:
        s: 流执行器实例
        sampling_params: 采样参数

        异常:
        NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError()

    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        temperature: float,
        choices_method: Optional[ChoicesSamplingMethod] = None,
    ) -> ChoicesDecision:
        """
        从候选中选择

        基于给定的候选项和温度参数进行选择。
        这是一个抽象方法，必须在子类中实现。

        参数:
        s: 流执行器实例
        choices: 候选项列表
        temperature: 温度参数
        choices_method: 选择方法，可选

        返回:
        ChoicesDecision: 选择决策结果

        异常:
        NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError()

    def concatenate_and_append(self, src_rids: List[str], dst_rid: str):
        """
        连接和追加操作

        将多个源请求的结果连接并追加到目标请求中。
        这是一个抽象方法，必须在子类中实现。

        参数:
        src_rids: 源请求ID列表
        dst_rid: 目标请求ID

        异常:
        NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError()

    def shutdown(self):
        """
        关闭后端

        优雅地关闭后端，清理所有资源。
        这是一个可选操作，默认实现为空操作。
        """
        pass

    def flush_cache(self):
        """
        清空缓存

        清空后端的所有缓存，释放内存资源。
        这是一个可选操作，默认实现为空操作。
        """
        pass

    def get_server_info(self):
        """
        获取服务器信息

        获取后端服务器的状态和配置信息。
        这是一个可选操作，默认实现为空操作。

        返回:
        dict: 服务器信息字典
        """
        pass
