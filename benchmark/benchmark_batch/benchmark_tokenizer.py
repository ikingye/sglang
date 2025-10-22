"""
分词器性能基准测试
这个脚本用于测试和比较不同批次大小下的分词器性能，包括：
- 顺序分词 vs 批量分词的性能对比
- 不同批次大小的性能表现
- 分词速度的统计测量

测试目的：
- 评估分词器的批量处理能力
- 找到最优的批次大小
- 为实际应用提供性能参考
"""

import random  # 随机数生成
import time  # 时间测量
from statistics import mean  # 统计平均值

from transformers import AutoTokenizer  # HuggingFace分词器

# 配置参数
TOKENIZER_DIR = (
    "/shared/public/sharing/fait360brew/training/models/meta-llama/Llama-3.2-3B"
)  # 分词器模型路径
NUM_TOKENS = 20000  # 每个提示应该包含的token数量
BATCH_SIZES = [1, 2, 4, 8]  # 测试不同的批次大小
NUM_RUNS = 5  # 每个批次大小的运行次数，用于获得可靠的测量结果


def generate_random_prompts(num_prompts, num_tokens, tokenizer):
    """
    生成指定数量的随机提示，每个提示包含指定数量的token

    参数:
    num_prompts: 要生成的提示数量
    num_tokens: 每个提示的token数量
    tokenizer: 分词器实例

    返回:
    list: 生成的提示列表
    """
    vocab_size = tokenizer.vocab_size  # 获取词汇表大小
    all_prompts = []

    print(f"Generating {num_prompts} random prompts with {num_tokens} tokens each...")
    for i in range(num_prompts):
        # 生成随机token ID - 这直接给我们确切的token数量
        random_token_ids = [
            random.randint(0, vocab_size - 1) for _ in range(num_tokens)
        ]
        # 解码为文本
        random_text = tokenizer.decode(
            random_token_ids, clean_up_tokenization_spaces=True
        )

        # 格式化提示
        prompt = f"Prompt {i}: {random_text}"
        # 验证token数量
        tokens = tokenizer.encode(prompt)
        print(f"  Prompt {i}: {len(tokens)} tokens")
        all_prompts.append(prompt)

    return all_prompts


def benchmark_sequential_vs_batch(prompts, batch_size, tokenizer):
    """
    比较给定批次大小下的顺序分词和批量分词性能

    参数:
    prompts: 提示列表
    batch_size: 批次大小
    tokenizer: 分词器实例

    返回:
    tuple: (顺序分词平均时间, 批量分词平均时间)
    """

    # 使用encode()进行顺序分词
    sequential_times = []
    for run in range(NUM_RUNS):
        batch_prompts = prompts[:batch_size]  # 使用相同的提示进行公平比较

        start_time = time.perf_counter()
        for prompt in batch_prompts:
            tokens = tokenizer.encode(prompt)  # 逐个分词
        sequential_time = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
        sequential_times.append(sequential_time)

    # Batch tokenization using tokenizer()
    batch_times = []
    for run in range(NUM_RUNS):
        batch_prompts = prompts[:batch_size]  # Use same prompts for fair comparison

        start_time = time.perf_counter()
        tokens = tokenizer(batch_prompts)
        batch_time = (time.perf_counter() - start_time) * 1000
        batch_times.append(batch_time)

    return {
        "batch_size": batch_size,
        "avg_sequential_ms": mean(sequential_times),
        "avg_batch_ms": mean(batch_times),
        "speedup_factor": (
            mean(sequential_times) / mean(batch_times) if mean(batch_times) > 0 else 0
        ),
        "sequential_runs": sequential_times,
        "batch_runs": batch_times,
    }


def main():
    print("Tokenizer Benchmark: Sequential vs Batch Processing")
    print("-" * 60)
    print(f"Tokenizer: {TOKENIZER_DIR}")
    print(f"Tokens per prompt: {NUM_TOKENS}")
    print(f"Number of runs per batch size: {NUM_RUNS}")
    print("-" * 60)

    # Load tokenizer once for all operations
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    # The largest batch size determines how many prompts we need
    max_batch_size = max(BATCH_SIZES)
    all_prompts = generate_random_prompts(max_batch_size, NUM_TOKENS, tokenizer)

    results = []
    print("\nRunning benchmark...")

    for batch_size in BATCH_SIZES:
        print(f"\nBenchmarking batch size: {batch_size}")
        result = benchmark_sequential_vs_batch(all_prompts, batch_size, tokenizer)
        results.append(result)

        print(f"  Sequential tokenization (encode):")
        for i, run_time in enumerate(result["sequential_runs"]):
            print(f"    Run {i+1}: {run_time:.2f} ms")
        print(f"    Average: {result['avg_sequential_ms']:.2f} ms")

        print(f"  Batch tokenization (tokenizer):")
        for i, run_time in enumerate(result["batch_runs"]):
            print(f"    Run {i+1}: {run_time:.2f} ms")
        print(f"    Average: {result['avg_batch_ms']:.2f} ms")

        print(f"  Speedup factor: {result['speedup_factor']:.2f}x")

    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(
        f"{'Batch Size':<10} {'Sequential (ms)':<18} {'Batch (ms)':<18} {'Speedup':<10}"
    )
    print("-" * 60)

    for result in results:
        print(
            f"{result['batch_size']:<10} {result['avg_sequential_ms']:.2f} ms{' ' * 8} {result['avg_batch_ms']:.2f} ms{' ' * 8} {result['speedup_factor']:.2f}x"
        )


if __name__ == "__main__":
    random.seed(0)
    main()
