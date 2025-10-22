"""
本地聊天示例
这个示例展示了如何使用SGLang进行多轮对话，包括单次请求、流式输出和批量处理。

使用方法:
python3 local_example_chat.py
"""

import sglang as sgl  # 导入SGLang库


@sgl.function  # 使用SGLang函数装饰器
def multi_turn_question(s, question_1, question_2):
    """
    多轮问答函数

    这个函数定义了一个多轮对话的模板，包含两轮问答：
    1. 第一轮：用户提问，助手回答
    2. 第二轮：用户再次提问，助手再次回答

    参数:
    s: SGLang状态对象，用于构建对话流程
    question_1: 第一个问题
    question_2: 第二个问题
    """
    # 第一轮对话：用户提问
    s += sgl.user(question_1)
    # 助手回答第一个问题，最多生成256个token
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))

    # 第二轮对话：用户再次提问
    s += sgl.user(question_2)
    # 助手回答第二个问题，最多生成256个token
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))


def single():
    """
    单次请求示例

    演示如何执行一次多轮对话请求，并打印结果。
    """
    # 执行多轮问答，传入具体的问题
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",  # 第一个问题：美国首都是什么？
        question_2="List two local attractions.",  # 第二个问题：列出两个当地景点
    )

    # 打印所有消息（包括用户和助手的对话）
    for m in state.messages():
        print(m["role"], ":", m["content"])

    # 单独打印第一个答案
    print("\n-- answer_1 --\n", state["answer_1"])


def stream():
    """
    流式输出示例

    演示如何以流式方式输出生成的内容，实时显示生成过程。
    """
    # 执行多轮问答，启用流式输出
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",  # 第一个问题
        question_2="List two local attractions.",  # 第二个问题
        stream=True,  # 启用流式输出
    )

    # 实时打印生成的文本
    for out in state.text_iter():
        print(out, end="", flush=True)  # 不换行，立即刷新输出
    print()  # 最后换行


def batch():
    """
    批量处理示例

    演示如何同时处理多个不同的多轮对话请求。
    """
    # 批量执行多个多轮问答请求
    states = multi_turn_question.run_batch(
        [
            {
                "question_1": "What is the capital of the United States?",  # 美国首都
                "question_2": "List two local attractions.",  # 当地景点
            },
            {
                "question_1": "What is the capital of France?",  # 法国首都
                "question_2": "What is the population of this city?",  # 城市人口
            },
        ]
    )

    # 打印每个请求的消息
    for s in states:
        print(s.messages())


if __name__ == "__main__":
    # 创建运行时环境，加载Llama-2-7b-chat模型
    runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-chat-hf")
    # 设置为默认后端
    sgl.set_default_backend(runtime)

    # 运行单次请求示例
    print("\n========== single ==========\n")
    single()

    # 运行流式输出示例
    print("\n========== stream ==========\n")
    stream()

    # 运行批量处理示例
    print("\n========== batch ==========\n")
    batch()

    # 关闭运行时环境，释放资源
    runtime.shutdown()
