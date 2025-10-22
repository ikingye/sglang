"""
SGLang运行时端点模块
这个模块实现了与SGLang运行时服务器通信的后端接口。

RuntimeEndpoint是BaseBackend的具体实现，通过HTTP API与远程的SGLang服务器通信，
支持文本生成、缓存管理、性能分析等功能。
"""

import atexit
import json
import multiprocessing
import warnings
from typing import Dict, List, Optional, Union

import aiohttp
import requests

from sglang.global_config import global_config
from sglang.lang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template, get_chat_template_by_model_path
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import (
    REGEX_BOOL,
    REGEX_FLOAT,
    REGEX_INT,
    REGEX_STR,
    SglSamplingParams,
)
from sglang.utils import http_request

class RuntimeEndpoint(BaseBackend):
    """
    SGLang运行时端点后端

    这个类实现了与远程SGLang服务器通信的后端接口，通过HTTP API调用
    远程服务器的各种功能，包括文本生成、缓存管理、性能分析等。

    主要功能：
    - 与远程SGLang服务器建立连接
    - 执行文本生成任务
    - 管理缓存和会话
    - 支持流式和非流式生成
    - 提供性能分析功能
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        verify: Optional[str] = None,
        chat_template_name: Optional[str] = None,
    ):
        """
        初始化运行时端点后端

        建立与远程SGLang服务器的连接，获取模型信息并设置聊天模板。

        参数:
        base_url: 远程服务器的基础URL
        api_key: API密钥，用于身份验证
        verify: SSL证书验证设置
        chat_template_name: 聊天模板名称，可选
        """
        super().__init__()
        self.support_concate_and_append = True

        self.base_url = base_url
        self.api_key = api_key
        self.verify = verify

        res = http_request(
            self.base_url + "/get_model_info",
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        self.model_info = res.json()

        if chat_template_name:
            self.chat_template = get_chat_template(chat_template_name)
        else:
            self.chat_template = get_chat_template_by_model_path(
                self.model_info["model_path"]
            )

    def get_model_name(self):
        """
        获取模型名称

        返回远程服务器使用的模型路径。

        返回:
        str: 模型路径
        """
        return self.model_info["model_path"]

    def flush_cache(self):
        """
        清空远程服务器缓存

        向远程服务器发送清空缓存的请求，释放服务器端的内存资源。
        """
        res = http_request(
            self.base_url + "/flush_cache",
            api_key=self.api_key,
            verify=self.verify,
            method="POST",
        )
        self._assert_success(res)

    def get_server_info(self):
        """
        获取服务器信息

        向远程服务器请求服务器状态和配置信息。

        返回:
        dict: 服务器信息字典
        """
        res = http_request(
            self.base_url + "/get_server_info",
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        return res.json()

    def get_chat_template(self):
        """
        获取聊天模板

        返回当前使用的聊天模板。

        返回:
        ChatTemplate: 聊天模板对象
        """
        return self.chat_template

    def cache_prefix(self, prefix_str: str):
        """
        缓存前缀字符串

        将指定的前缀字符串发送到远程服务器进行缓存，
        用于后续的快速检索和复用。

        参数:
        prefix_str: 要缓存的前缀字符串
        """
        res = http_request(
            self.base_url + "/generate",
            json={"text": prefix_str, "sampling_params": {"max_new_tokens": 0}},
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def start_profile(self):
        """
        开始性能分析

        向远程服务器发送开始性能分析的请求，
        用于收集和分析服务器性能数据。
        """
        res = http_request(
            self.base_url + "/start_profile",
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def stop_profile(self):
        """
        停止性能分析

        向远程服务器发送停止性能分析的请求，
        结束性能数据收集。
        """
        res = http_request(
            self.base_url + "/stop_profile",
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def commit_lazy_operations(self, s: StreamExecutor):
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def fill_image(self, s: StreamExecutor):
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def _handle_dtype_to_regex(self, sampling_params: SglSamplingParams):
        if sampling_params.dtype is None:
            return

        if sampling_params.stop == ():
            sampling_params.stop = []

        dtype_regex = None
        if sampling_params.dtype in ["int", int]:
            dtype_regex = REGEX_INT
            sampling_params.stop.extend([" ", "\n"])
        elif sampling_params.dtype in ["float", float]:
            dtype_regex = REGEX_FLOAT
            sampling_params.stop.extend([" ", "\n"])
        elif sampling_params.dtype in ["str", str]:
            dtype_regex = REGEX_STR
        elif sampling_params.dtype in ["bool", bool]:
            dtype_regex = REGEX_BOOL
        else:
            raise RuntimeError(f"Invalid dtype: {sampling_params.dtype}")

        if dtype_regex is not None and sampling_params.regex is not None:
            warnings.warn(
                f"Both dtype and regex are set. Only dtype will be used. dtype: {sampling_params.dtype}, regex: {sampling_params.regex}"
            )

        sampling_params.regex = dtype_regex

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        self._handle_dtype_to_regex(sampling_params)
        data = {
            "text": s.text_,
            "sampling_params": {
                "skip_special_tokens": global_config.skip_special_tokens_in_output,
                "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,
                **sampling_params.to_srt_kwargs(),
            },
        }

        for item in [
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "return_text_in_logprobs",
        ]:
            value = getattr(sampling_params, item, None)
            if value is not None:
                data[item] = value

        self._add_images(s, data)

        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

        obj = res.json()
        comp = obj["text"]
        return comp, obj["meta_info"]

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        self._handle_dtype_to_regex(sampling_params)

        data = {
            "text": s.text_,
            "sampling_params": {
                "skip_special_tokens": global_config.skip_special_tokens_in_output,
                "spaces_between_special_tokens": global_config.spaces_between_special_tokens_in_out,
                **sampling_params.to_srt_kwargs(),
            },
        }

        for item in [
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "return_text_in_logprobs",
        ]:
            value = getattr(sampling_params, item, None)
            if value is not None:
                data[item] = value

        data["stream"] = True
        self._add_images(s, data)

        res = http_request(
            self.base_url + "/generate",
            json=data,
            stream=True,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        pos = 0

        for chunk in res.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                data = json.loads(chunk[5:].strip("\n"))
                chunk_text = data["text"][pos:]
                meta_info = data["meta_info"]
                pos += len(chunk_text)
                yield chunk_text, meta_info

    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        temperature: float,
        choices_method: ChoicesSamplingMethod,
    ) -> ChoicesDecision:
        assert temperature <= 1e-5

        # Cache common prefix
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        obj = self._generate_http_request(s, data)
        prompt_len = obj["meta_info"]["prompt_tokens"]
        logprob_start_len = max(prompt_len - 2, 0)  # For token healing

        # Compute logprob
        data = {
            "text": [s.text_ + c for c in choices],
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": 0,
            },
            "return_logprob": True,
            "return_text_in_logprobs": True,
            "logprob_start_len": logprob_start_len,
        }
        obj = self._generate_http_request(s, data)

        input_token_logprobs = [r["meta_info"]["input_token_logprobs"] for r in obj]
        output_token_logprobs = [r["meta_info"]["output_token_logprobs"] for r in obj]
        normalized_prompt_logprobs = [
            compute_normalized_prompt_logprobs(r["meta_info"]["input_token_logprobs"])
            for r in obj
        ]

        # Remove extra token if no token healing occurred
        for i in range(len(input_token_logprobs)):
            healed_token_str = input_token_logprobs[i][0][-1]
            if s.text_.endswith(healed_token_str):
                healed_token_logprob = input_token_logprobs[i][0][0]
                normalized_prompt_logprobs[i] = (
                    normalized_prompt_logprobs[i] * len(input_token_logprobs[i])
                    - healed_token_logprob
                ) / (len(input_token_logprobs[i]) - 1)
                input_token_logprobs[i] = input_token_logprobs[i][1:]

        # Compute unconditional logprobs if required
        if choices_method.requires_unconditional_logprobs:
            input_ids = [[el[1] for el in subl] for subl in input_token_logprobs]
            data = {
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 0},
                "return_logprob": True,
            }
            obj = self._generate_http_request(s, data)
            unconditional_token_logprobs = [
                r["meta_info"]["input_token_logprobs"] for r in obj
            ]
        else:
            unconditional_token_logprobs = None

        return choices_method(
            choices=choices,
            normalized_prompt_logprobs=normalized_prompt_logprobs,
            input_token_logprobs=input_token_logprobs,
            output_token_logprobs=output_token_logprobs,
            unconditional_token_logprobs=unconditional_token_logprobs,
        )

    def concatenate_and_append(self, src_rids: List[str], dst_rid: str):
        res = http_request(
            self.base_url + "/concate_and_append_request",
            json={"src_rids": src_rids, "dst_rid": dst_rid},
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)

    def _generate_http_request(self, s: StreamExecutor, data):
        self._add_images(s, data)
        res = http_request(
            self.base_url + "/generate",
            json=data,
            api_key=self.api_key,
            verify=self.verify,
        )
        self._assert_success(res)
        return res.json()

    def _add_images(self, s: StreamExecutor, data):
        if s.images_:
            assert len(s.images_) == 1, "Only support one image."
            data["image_data"] = s.images_[0][1]

    def _assert_success(self, res):
        if res.status_code != 200:
            try:
                content = res.json()
            except json.JSONDecodeError:
                content = res.text
            raise RuntimeError(content)

def compute_normalized_prompt_logprobs(input_logprobs):
    values = [x[0] for x in input_logprobs if x[0]]
    return sum(values) / len(values)

class Runtime:
    """
    A wrapper for the HTTP server.
    This is used for launching the server in a python program without
    using the command line interface.

    It is mainly used for the frontend language.
    You should use the Engine class if you want to do normal offline processing without the frontend language.
    """

    def __init__(
        self,
        log_level: str = "error",
        *args,
        **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        # We delay the import of any `sglang.srt` components in `sglang.lang`, so users can run
        # client code without installing SRT server and its dependency if they want.
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.utils import is_port_available

        self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)

        # Pre-allocate ports
        for port in range(self.server_args.port, 40000):
            if is_port_available(port):
                break
        self.server_args.port = port

        self.url = self.server_args.url()
        self.generate_url = self.url + "/generate"

        # NOTE: We store pid instead of proc to fix some issues during __delete__
        self.pid = None
        pipe_reader, pipe_writer = multiprocessing.Pipe(duplex=False)

        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=launch_server,
            args=(self.server_args, pipe_writer),
        )
        proc.start()
        pipe_writer.close()
        self.pid = proc.pid

        # Before python program terminates, call shutdown implicitly. Therefore, users don't have to explicitly call .shutdown()
        atexit.register(self.shutdown)

        # TODO: remove this pipe_writer mechanism and use `/health_generate` instead.
        try:
            init_state = pipe_reader.recv()
        except EOFError:
            init_state = ""

        if init_state != "ready":
            self.shutdown()
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )

        self.endpoint = RuntimeEndpoint(self.url)

    def shutdown(self):
        from sglang.srt.utils import kill_process_tree

        if self.pid is not None:
            kill_process_tree(self.pid)
            self.pid = None

    def start_profile(self):
        self.endpoint.start_profile()

    def stop_profile(self):
        self.endpoint.stop_profile()

    def cache_prefix(self, prefix: str):
        self.endpoint.cache_prefix(prefix)

    def get_tokenizer(self):
        from sglang.srt.hf_transformers_utils import get_tokenizer

        return get_tokenizer(
            self.server_args.tokenizer_path,
            tokenizer_mode=self.server_args.tokenizer_mode,
            trust_remote_code=self.server_args.trust_remote_code,
            revision=self.server_args.revision,
        )

    async def async_generate(
        self,
        prompt: str,
        sampling_params: Optional[Dict] = None,
    ):
        if self.server_args.skip_tokenizer_init:
            json_data = {
                "input_ids": prompt,
                "sampling_params": sampling_params,
                "stream": True,
            }
        else:
            json_data = {
                "text": prompt,
                "sampling_params": sampling_params,
                "stream": True,
            }
        pos = 0

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(self.generate_url, json=json_data) as response:
                async for chunk, _ in response.content.iter_chunks():
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        if chunk == "data: [DONE]\n\n":
                            break
                        data = json.loads(chunk[5:].strip("\n"))
                        if "text" in data:
                            cur = data["text"][pos:]
                            if cur:
                                yield cur
                            pos += len(cur)
                        else:
                            yield data

    add_request = async_generate

    def generate(
        self,
        prompt: Union[str, List[str]],
        sampling_params: Optional[Dict] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
    ):
        json_data = {
            "text": prompt,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "lora_path": lora_path,
        }
        assert not isinstance(lora_path, list) or len(lora_path) == len(prompt)
        response = requests.post(
            self.url + "/generate",
            json=json_data,
        )
        return json.dumps(response.json())

    def encode(
        self,
        prompt: Union[str, List[str], List[Dict], List[List[Dict]]],
    ):
        json_data = {"text": prompt}
        response = requests.post(self.url + "/encode", json=json_data)
        return json.dumps(response.json())

    async def get_server_info(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/get_server_info") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.json()
                    raise RuntimeError(
                        f"Failed to get server info. {error_data['error']['message']}"
                    )

    def __del__(self):
        self.shutdown()
