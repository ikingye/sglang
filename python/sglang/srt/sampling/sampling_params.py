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
"""
SGLang文本生成采样参数模块
这个模块定义了文本生成过程中使用的采样参数。

SamplingParams类包含了控制文本生成质量和随机性的所有参数，
包括温度、top-p、top-k、停止条件、约束生成等。
"""

from typing import Any, Dict, List, Optional, Union

_SAMPLING_EPS = 1e-6
TOP_K_ALL = 1 << 30


class SamplingParams:
    """
    SGLang文本生成采样参数类

    这个类定义了文本生成过程中使用的所有采样参数，包括：
    - 生成长度控制（max_new_tokens, min_new_tokens）
    - 采样策略（temperature, top_p, top_k, min_p）
    - 停止条件（stop, stop_token_ids, ignore_eos）
    - 惩罚机制（frequency_penalty, presence_penalty, repetition_penalty）
    - 约束生成（json_schema, regex, ebnf）
    - 输出格式（skip_special_tokens, spaces_between_special_tokens）

    详细文档请参考：
    docs/backend/sampling_params.md 或
    https://docs.sglang.ai/backend/sampling_params.html
    """

    def __init__(
        self,
        max_new_tokens: int = 128,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        n: int = 1,
        json_schema: Optional[str] = None,
        regex: Optional[str] = None,
        ebnf: Optional[str] = None,
        structural_tag: Optional[str] = None,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        no_stop_trim: bool = False,
        custom_params: Optional[Dict[str, Any]] = None,
        stream_interval: Optional[int] = None,
        logit_bias: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        初始化采样参数

        设置所有采样相关的参数，并进行特殊情况的处理。
        """
        self.max_new_tokens = max_new_tokens
        self.stop_strs = stop
        if stop_token_ids:
            self.stop_token_ids = set(stop_token_ids)
        else:
            self.stop_token_ids = None

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p

        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty

        self.min_new_tokens = min_new_tokens
        self.regex = regex
        self.n = n
        self.json_schema = json_schema
        self.ebnf = ebnf
        self.structural_tag = structural_tag
        self.ignore_eos = ignore_eos

        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.no_stop_trim = no_stop_trim

        self.custom_params = custom_params
        self.stream_interval = stream_interval
        self.logit_bias = logit_bias

        if 0 <= self.temperature < _SAMPLING_EPS:
            # 极低温度视为贪心解，直接退化成 top-k=1 防止数值不稳定
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            # -1 表示关闭截断，将上限替换为一个足够大的常量
            self.top_k = TOP_K_ALL

    def verify(self, vocab_size):
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.top_k < 1 or self.top_k == -1:
            raise ValueError(
                f"top_k must be -1 (disable) or at least 1, got {self.top_k}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        if not 0.0 <= self.repetition_penalty <= 2.0:
            raise ValueError(
                "repetition_penalty must be in [0, 2], got "
                f"{self.repetition_penalty}."
            )
        if not 0 <= self.min_new_tokens:
            raise ValueError(
                f"min_new_tokens must be in [0, max_new_tokens], got "
                f"{self.min_new_tokens}."
            )
        if self.max_new_tokens is not None:
            if self.max_new_tokens < 0:
                raise ValueError(
                    f"max_new_tokens must be at least 0, got {self.max_new_tokens}."
                )
            if not self.min_new_tokens <= self.max_new_tokens:
                raise ValueError(
                    f"min_new_tokens must be in [0, max_new_tokens({self.max_new_tokens})], got "
                    f"{self.min_new_tokens}."
                )
        if self.logit_bias is not None:
            # 校验 logit_bias 的键是否落在词表范围内
            for token_id in self.logit_bias:
                if not 0 <= int(token_id) < vocab_size:
                    raise ValueError(
                        f"logit_bias must has keys in [0, {vocab_size - 1}], got "
                        f"{token_id}."
                    )
        grammars = [
            self.json_schema,
            self.regex,
            self.ebnf,
        ]  # since mutually exclusive, only one can be set
        # 多种约束互斥，仅允许设置其中一种
        if sum(x is not None for x in grammars) > 1:
            raise ValueError("Only one of regex, json_schema, or ebnf can be set.")

    def normalize(self, tokenizer):
        # Process stop strings
        if self.stop_strs is None:
            self.stop_strs = []
            self.stop_str_max_len = 0
        else:
            if isinstance(self.stop_strs, str):
                self.stop_strs = [self.stop_strs]

            stop_str_max_len = 0
            for stop_str in self.stop_strs:
                if tokenizer is not None:
                    stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))
                else:
                    stop_str_max_len = max(stop_str_max_len, len(stop_str))
            # stop_str_max_len 记录需要保留的后缀长度，便于后续裁剪
            self.stop_str_max_len = stop_str_max_len
