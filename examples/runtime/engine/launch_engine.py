"""
离线引擎启动示例
这个示例演示了如何启动和使用SGLang的离线推理引擎。

离线引擎适用于：
- 单机推理任务
- 批量处理
- 不需要HTTP服务器的场景
"""

import sglang as sgl  # 导入SGLang库


def main():
    """
    主函数：演示离线引擎的基本使用方法

    这个函数展示了如何：
    1. 创建离线推理引擎
    2. 执行文本生成
    3. 关闭引擎释放资源
    """
    # 创建离线推理引擎，加载Llama-3.1-8B-Instruct模型
    # Engine类提供了直接的模型推理接口，不需要HTTP服务器
    llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

    # 执行文本生成，询问法国首都
    # generate方法会直接返回生成的文本结果
    llm.generate("What is the capital of France?")

    # 关闭引擎，释放GPU内存和其他资源
    # 这是重要的清理步骤，确保资源被正确释放
    llm.shutdown()


# __main__条件在这里是必要的，因为我们使用"spawn"来创建子进程
# Spawn每次都会启动一个全新的程序，如果没有__main__条件，
# 它会陷入无限循环，不断从sgl.Engine生成进程
if __name__ == "__main__":
    main()
