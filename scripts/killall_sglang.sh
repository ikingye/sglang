#!/bin/bash
# SGLang进程清理脚本
# 这个脚本用于清理所有SGLang相关的进程，支持NVIDIA和AMD GPU环境

if [ "$1" = "rocm" ]; then  # 检查是否为ROCm模式（AMD GPU）
    echo "Running in ROCm mode"  # 显示ROCm模式信息

    # 清理SGLang相关进程
    # 使用pgrep查找所有包含SGLang关键字的进程并强制终止
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt' | xargs -r kill -9

else  # NVIDIA GPU模式
    # 显示当前GPU状态
    nvidia-smi  # 显示NVIDIA GPU信息

    # 清理SGLang相关进程
    # 查找并终止所有SGLang相关的进程
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt' | xargs -r kill -9

    # 如果提供了任何参数，则清理所有GPU进程
    if [ $# -gt 0 ]; then
        # 检查sudo是否可用
        if command -v sudo >/dev/null 2>&1; then  # 如果sudo命令存在
            sudo apt-get update  # 更新包列表
            sudo apt-get install -y lsof  # 安装lsof工具（列出打开的文件）
        else  # 如果没有sudo权限
            apt-get update  # 更新包列表
            apt-get install -y lsof  # 安装lsof工具
        fi
        # 终止所有GPU进程
        # 从nvidia-smi输出中提取进程ID并强制终止
        kill -9 $(nvidia-smi | sed -n '/Processes:/,$p' | grep "   [0-9]" | awk '{print $5}') 2>/dev/null
        # 终止所有使用NVIDIA设备的进程
        lsof /dev/nvidia* | awk '{print $2}' | xargs kill -9 2>/dev/null
    fi

    # 清理后显示GPU状态
    nvidia-smi  # 再次显示GPU状态，确认清理结果
fi
