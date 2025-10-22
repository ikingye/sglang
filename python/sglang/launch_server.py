"""启动推理服务器
这个脚本是SGLang推理服务器的主入口点，负责解析命令行参数并启动HTTP服务器。
"""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server

from sglang.srt.server_args import prepare_server_args

from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)