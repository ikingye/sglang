"""全局配置模块
这个模块定义了SGLang系统的全局配置参数，包括运行时参数、优化设置等。
这些配置影响整个系统的行为，可以通过环境变量进行覆盖。
"""

import os

class GlobalConfig:
    """
    存储全局常量和配置参数。

    这个类管理SGLang系统的全局配置，包括：
    - 日志输出级别
    - 默认后端设置
    - 运行时性能参数
    - 输出格式配置
    - 优化开关

    注意：更多运行时参数存储在 python/sglang/srt/managers/schedule_batch.py::global_server_args_dict 中
    """

    def __init__(self):
        self.verbosity = 0
        self.default_backend = None

        self.default_init_new_token_ratio = float(
            os.environ.get("SGLANG_INIT_NEW_TOKEN_RATIO", 0.7)
        )
        self.default_min_new_token_ratio_factor = float(
            os.environ.get("SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR", 0.14)
        )
        self.default_new_token_ratio_decay_steps = float(
            os.environ.get("SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS", 600)
        )

        self.torch_empty_cache_interval = float(
            os.environ.get("SGLANG_EMPTY_CACHE_INTERVAL", -1)
        )
        self.retract_decode_steps = 20
        self.flashinfer_workspace_size = os.environ.get(
            "FLASHINFER_WORKSPACE_SIZE", 384 * 1024 * 1024
        )

        self.skip_special_tokens_in_output = True
        self.spaces_between_special_tokens_in_out = True

        self.enable_precache_with_tracing = True
        self.enable_parallel_encoding = True

global_config = GlobalConfig()
