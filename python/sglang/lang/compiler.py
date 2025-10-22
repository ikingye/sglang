"""SGLang 编译器模块

这个模块负责把通过 `trace` 收集到的 SGLang 中间表示转换为可执行的有向
无环图（DAG），并在执行阶段根据图结构复用流式执行器，实现多分支推理和
批量推理等高级能力。相比即时解释执行，预编译能够减少重复工作、最大化
缓存命中率，并在并行场景中显著提升吞吐量。
"""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List, Union

from sglang.global_config import global_config
from sglang.lang.interpreter import ProgramState, StreamExecutor, cache_program
from sglang.lang.ir import SglArgument, SglExpr, SglSamplingParams, SglVariable

def compile_func(function, backend):
    """编译用户定义的 SGLang 程序为 `CompiledFunction`。

    通过 `function.trace` 预先执行一次程序，收集中间表示构成的执行图，随后
    构造 `CompiledFunction` 对象用于复用该图。这样可以在后续执行中跳过
    trace 阶段的开销，同时保证和解释执行一致的语义。

    参数:
        function: 用户以装饰器声明的 SGLang 程序对象。
        backend: 运行时后端，trace 阶段需要它来解析模型能力与配置。

    返回:
        `CompiledFunction`，可以直接调用 `run`/`run_batch` 复用预构建的执行图。
    """

    tracer = function.trace(backend=backend)
    compiler = CompiledFunction(tracer, function)
    return compiler

class CompiledFunction:
    """SGLang 编译产物，封装静态执行图与高性能运行路径。

    `trace` 阶段生成的节点本质上是一个以 `prev_node` 链接的链表，加上变量
    定义的额外引用。`CompiledFunction` 会把这些节点整理成显式的计算图，
    并在初始化时完成拓扑排序，为后续执行提供稳定的顺序和复用机会。
    """

    def __init__(self, tracer, function):
        self.function = function

        self.last_node = CompGraphNode(tracer.last_node)
        self.expr_to_node = {}
        self.build_graph(tracer)
        self.topological_sort()

    def build_graph(self, tracer):
        """把 trace 产生的链表结构转换为便于执行的计算图。"""

        self.nodes = [self.last_node]
        self.expr_to_node[tracer.last_node] = self.nodes[-1]

        rename_pid = {}

        visited = set([tracer.last_node])
        head = 0
        while head < len(self.nodes):
            cur_node = self.nodes[head]

            prev_node = cur_node.expr.prev_node
            if prev_node is not None:
                if prev_node not in visited:
                    visited.add(prev_node)
                    self.nodes.append(CompGraphNode(prev_node))
                    self.expr_to_node[prev_node] = self.nodes[-1]
                cur_node.prev_node = self.expr_to_node[prev_node]
                self.expr_to_node[prev_node].add_next_node(cur_node)

            if isinstance(cur_node.expr, SglVariable):
                if cur_node.expr.name in tracer.variables:
                    source = tracer.variables[cur_node.expr.name].source
                else:
                    source = cur_node.expr.source
                if source not in visited:
                    visited.add(source)
                    self.nodes.append(CompGraphNode(source))
                    self.expr_to_node[source] = self.nodes[-1]
                cur_node.source_node = self.expr_to_node[source]
                self.expr_to_node[source].add_next_node(cur_node)
            head += 1

            if cur_node.expr.pid not in rename_pid:
                rename_pid[cur_node.expr.pid] = len(rename_pid)
            cur_node.expr.pid = rename_pid[cur_node.expr.pid]

    def topological_sort(self):
        """对构建完成的图执行拓扑排序，保证执行顺序合法。"""

        prevd = {}
        cand = Queue()
        for x in self.nodes:
            prevd[x] = (x.prev_node is not None) + (x.source_node is not None)
            if prevd[x] == 0:
                cand.put(x)
        new_list = []
        while cand.qsize() > 0:
            head = cand.get()
            new_list.append(head)
            for x in head.next_nodes:
                prevd[x] -= 1
                if prevd[x] == 0:
                    cand.put(x)
        self.nodes = new_list

    def print_graph(
        self,
    ):
        """调试辅助：以人类可读的格式输出拓扑序中的每个节点。"""
        for node in self.nodes:
            print(node)

    def run_internal(
        self,
        backend,
        kwargs,
        default_sampling_para,
    ):
        """以拓扑顺序执行计算图，复用 StreamExecutor 管理多个流。"""

        stream_executor_ids = set([x.expr.pid for x in self.nodes])
        stream_executors = {}
        for x in stream_executor_ids:
            arguments = kwargs if x == self.last_node.expr.pid else {}
            stream_executors[x] = StreamExecutor(
                backend, arguments, default_sampling_para, None, False
            )
        for node in self.nodes:
            se_id = node.expr.pid
            expr = node.expr
            if isinstance(expr, SglVariable):
                expr = SglVariable(expr.name, expr.source)
                expr.source_stream_executor = stream_executors[
                    node.source_node.expr.pid
                ]
            elif isinstance(expr, SglArgument):
                expr = kwargs[expr.name]
            stream_executors[se_id].submit(expr)
        for stream_executor in stream_executors.values():
            stream_executor.end()
        return ProgramState(stream_executors[self.last_node.expr.pid])

    def run(
        self,
        *,
        max_new_tokens: int = 128,
        stop: Union[str, List[str]] = (),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        backend=None,
        **kwargs,
    ):
        """以单个样本运行编译后的程序。

        该方法会补齐编译时绑定的默认参数，构造推理采样参数，并调度
        `run_internal` 执行拓扑图，保证与解释执行一致的默认行为。
        """

        backend = backend or global_config.default_backend

        kwargs.update(self.function.bind_arguments)

        default_sampling_para = SglSamplingParams(
            max_new_tokens=max_new_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        return self.run_internal(backend, kwargs, default_sampling_para)

    def run_batch(
        self,
        batch_kwargs,
        *,
        max_new_tokens: int = 128,
        stop: Union[str, List[str]] = (),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        backend=None,
        num_threads: Union[str, int] = "auto",
    ):
        """批量运行编译后的程序，支持并行调度和缓存复用。"""

        assert isinstance(batch_kwargs, (list, tuple))
        if len(batch_kwargs) == 0:
            return []
        assert isinstance(batch_kwargs[0], dict)

        backend = backend or global_config.default_backend

        default_sampling_para = SglSamplingParams(
            max_new_tokens=max_new_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        if len(batch_kwargs) > 1:
            cache_program(self.function, backend)

        if num_threads == "auto":
            num_threads = multiprocessing.cpu_count()
        num_threads = min(num_threads, len(batch_kwargs))

        if num_threads == 1:
            rets = []
            for arguments in batch_kwargs:
                rets.append(
                    self.run_internal(backend, arguments, default_sampling_para)
                )
        else:
            with ThreadPoolExecutor(num_threads) as executor:
                futures = []
                for arguments in batch_kwargs:
                    futures.append(
                        executor.submit(
                            self.run_internal, backend, arguments, default_sampling_para
                        )
                    )
                rets = [f.result() for f in futures]
            rets[-1].sync()

        return rets

class CompGraphNode:
    def __init__(
        self, expr: SglExpr, prev_node=None, next_nodes=None, source_node=None
    ):
        """图节点包装类，对 trace 表达式增加执行期需要的邻接信息。"""

        self.expr = expr
        self.next_nodes = next_nodes or []
        self.prev_node = prev_node
        self.source_node = source_node

    def add_next_node(self, other):
        self.next_nodes.append(other)

    def __repr__(self):
        """返回调试友好的字符串，展示流 id、节点 id 及表达式内容。"""
        re = f"stream {self.expr.pid:2d}: "
        re += f"%{self.expr.node_id} = "
        if self.prev_node is not None:
            re += f"%{self.prev_node.expr.node_id} + "
        re += repr(self.expr)
        return re
