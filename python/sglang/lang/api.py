"""SGLang语言前端公共API接口
这个模块定义了SGLang语言前端的核心API，包括：
- 文本生成函数
- 角色标记函数
- 多模态输入函数
- 后端管理函数
- 运行时管理函数

这些API是用户与SGLang交互的主要接口，提供了简洁易用的编程模型。
"""

import re
from typing import Callable, List, Optional, Union

from sglang.global_config import global_config

from sglang.lang.backend.base_backend import BaseBackend

from sglang.lang.choices import ChoicesSamplingMethod, token_length_normalized

from sglang.lang.ir import (
    SglExpr,
    SglExprList,
    SglFunction,
    SglGen,
    SglImage,
    SglRoleBegin,
    SglRoleEnd,
    SglSelect,
    SglSeparateReasoning,
    SglVideo,
)

def function(
    func: Optional[Callable] = None, num_api_spec_tokens: Optional[int] = None
):
    """
    函数装饰器：将Python函数转换为SGLang函数节点

    这个装饰器用于将普通的Python函数包装成SGLang可以调用的函数节点，
    支持结构化输出和函数调用功能。

    参数
    ----------
    func : Optional[Callable]
        要装饰的Python函数，如果为None则返回装饰器
    num_api_spec_tokens : Optional[int]
        API规范token数量，用于估算函数调用的token消耗

    返回
    -------
    SglFunction 或 decorator
        如果提供了func参数，返回SglFunction对象；否则返回装饰器函数
    """
    if func:
        return SglFunction(func, num_api_spec_tokens=num_api_spec_tokens)

    def decorator(func):
        return SglFunction(func, num_api_spec_tokens=num_api_spec_tokens)

    return decorator

def Runtime(*args, **kwargs):
    """
    创建运行时环境实例

    这个函数用于创建运行时环境，管理模型的生命周期和资源。
    使用延迟导入避免不必要的依赖加载。

    参数
    ----------
    *args, **kwargs
        传递给Runtime构造函数的参数

    返回
    -------
    Runtime
        运行时环境实例
    """

    from sglang.lang.backend.runtime_endpoint import Runtime

    return Runtime(*args, **kwargs)

def Engine(*args, **kwargs):
    """
    创建推理引擎实例

    这个函数用于创建推理引擎，负责实际的模型推理执行。
    使用延迟导入避免不必要的依赖加载。

    参数
    ----------
    *args, **kwargs
        传递给Engine构造函数的参数

    返回
    -------
    Engine
        推理引擎实例
    """

    from sglang.srt.entrypoints.engine import Engine

    return Engine(*args, **kwargs)

def set_default_backend(backend: BaseBackend):
    """
    设置默认后端

    这个函数用于设置全局默认后端，后续的API调用将使用这个后端。

    参数
    ----------
    backend : BaseBackend
        要设置为默认的后端实例
    """
    global_config.default_backend = backend

def flush_cache(backend: Optional[BaseBackend] = None):
    """
    清空后端缓存

    这个函数用于清空指定后端的缓存，释放内存资源。
    如果未指定后端，则使用默认后端。

    参数
    ----------
    backend : Optional[BaseBackend]
        要清空缓存的后端，如果为None则使用默认后端

    返回
    -------
    bool
        清空操作是否成功
    """

    backend = backend or global_config.default_backend
    if backend is None:
        return False

    if hasattr(backend, "endpoint"):
        backend = backend.endpoint
    return backend.flush_cache()

def get_server_info(backend: Optional[BaseBackend] = None):
    """
    获取服务器信息

    这个函数用于获取指定后端的服务器信息，包括模型状态、配置等。
    如果未指定后端，则使用默认后端。

    参数
    ----------
    backend : Optional[BaseBackend]
        要获取信息的后端，如果为None则使用默认后端

    返回
    -------
    dict 或 None
        服务器信息字典，如果后端不存在则返回None
    """

    backend = backend or global_config.default_backend
    if backend is None:
        return None

    if hasattr(backend, "endpoint"):
        backend = backend.endpoint
    return backend.get_server_info()

def gen(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None,
    n: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
    return_logprob: Optional[bool] = None,
    logprob_start_len: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    return_text_in_logprobs: Optional[bool] = None,
    dtype: Optional[Union[type, str]] = None,
    choices: Optional[List[str]] = None,
    choices_method: Optional[ChoicesSamplingMethod] = None,
    regex: Optional[str] = None,
    json_schema: Optional[str] = None,
):
    """
    调用模型进行文本生成

    这是SGLang的核心生成函数，支持多种生成模式和参数配置。
    根据参数的不同组合，可以创建不同类型的生成节点。

    参数
    ----------
    name : Optional[str]
        生成结果的变量名，用于在后续代码中引用
    max_tokens : Optional[int]
        最大生成token数量，控制生成长度的上限
    min_tokens : Optional[int]
        最小生成token数量，确保生成足够长度的内容
    n : Optional[int]
        生成候选数量，用于生成多个不同的结果
    stop : Optional[Union[str, List[str]]]
        停止字符串或字符串列表，遇到这些字符串时停止生成
    stop_token_ids : Optional[List[int]]
        停止token ID列表，遇到这些token时停止生成
    temperature : Optional[float]
        温度参数，控制生成的随机性（0.0-2.0）
    top_p : Optional[float]
        核采样参数，控制候选token的累积概率阈值
    top_k : Optional[int]
        Top-K采样参数，只从概率最高的K个token中选择
    min_p : Optional[float]
        最小概率阈值，过滤掉概率过低的token
    frequency_penalty : Optional[float]
        频率惩罚，减少重复token的出现概率
    presence_penalty : Optional[float]
        存在惩罚，减少已出现token的重复概率
    ignore_eos : Optional[bool]
        是否忽略EOS token，继续生成直到达到其他停止条件
    return_logprob : Optional[bool]
        是否返回每个token的对数概率
    logprob_start_len : Optional[int]
        开始计算对数概率的token位置
    top_logprobs_num : Optional[int]
        返回的top token对数概率数量
    return_text_in_logprobs : Optional[bool]
        是否在概率信息中包含token文本
    dtype : Optional[Union[type, str]]
        返回数据的类型，用于类型转换
    choices : Optional[List[str]]
        候选选项列表，用于约束生成内容
    choices_method : Optional[ChoicesSamplingMethod]
        选择采样方法，控制如何从候选中选择
    regex : Optional[str]
        正则表达式约束，确保生成内容符合指定模式
    json_schema : Optional[str]
        JSON模式约束，确保生成内容符合JSON格式

    返回
    -------
    SglGen 或 SglSelect
        生成节点或选择节点

    注意
    ----
    详细的参数说明请参考 docs/backend/sampling_params.md
    """

    if choices:
        return SglSelect(
            name,
            choices,
            0.0 if temperature is None else temperature,
            token_length_normalized if choices_method is None else choices_method,
        )

    if regex is not None:
        try:
            re.compile(regex)
        except re.error as e:
            raise e

    return SglGen(
        name,
        max_tokens,
        min_tokens,
        n,
        stop,
        stop_token_ids,
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        return_text_in_logprobs,
        dtype,
        regex,
        json_schema,
    )

def gen_int(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    n: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
    return_logprob: Optional[bool] = None,
    logprob_start_len: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    return_text_in_logprobs: Optional[bool] = None,
):
    return SglGen(
        name,
        max_tokens,
        None,
        n,
        stop,
        stop_token_ids,
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        return_text_in_logprobs,
        int,
        None,
    )

def gen_string(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    n: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
    return_logprob: Optional[bool] = None,
    logprob_start_len: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    return_text_in_logprobs: Optional[bool] = None,
):
    return SglGen(
        name,
        max_tokens,
        None,
        n,
        stop,
        stop_token_ids,
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        return_text_in_logprobs,
        str,
        None,
    )

def image(expr: SglExpr):
    return SglImage(expr)

def video(path: str, num_frames: int):
    return SglVideo(path, num_frames)

def select(
    name: Optional[str] = None,
    choices: Optional[List[str]] = None,
    temperature: float = 0.0,
    choices_method: ChoicesSamplingMethod = token_length_normalized,
):
    assert choices is not None
    return SglSelect(name, choices, temperature, choices_method)

def _role_common(name: str, expr: Optional[SglExpr] = None):
    if expr is None:
        return SglExprList([SglRoleBegin(name), SglRoleEnd(name)])
    else:
        return SglExprList([SglRoleBegin(name), expr, SglRoleEnd(name)])

def system(expr: Optional[SglExpr] = None):
    return _role_common("system", expr)

def user(expr: Optional[SglExpr] = None):
    return _role_common("user", expr)

def assistant(expr: Optional[SglExpr] = None):
    return _role_common("assistant", expr)

def system_begin():
    return SglRoleBegin("system")

def system_end():
    return SglRoleEnd("system")

def user_begin():
    return SglRoleBegin("user")

def user_end():
    return SglRoleEnd("user")

def assistant_begin():
    return SglRoleBegin("assistant")

def assistant_end():
    return SglRoleEnd("assistant")

def separate_reasoning(
    expr: Optional[SglExpr] = None, model_type: Optional[str] = None
):
    return SglExprList([expr, SglSeparateReasoning(model_type, expr=expr)])
