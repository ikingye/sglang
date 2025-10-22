import torch

from sglang.srt.distributed import get_world_group


class PollBasedBarrier:
    def __init__(self, noop: bool = False):
        self._noop = noop
        self._local_arrived = False

    def local_arrive(self):
        assert not self._local_arrived
        # 标记本地线程/进程已经抵达同步点
        self._local_arrived = True

    def poll_global_arrived(self) -> bool:
        global_arrived = self._compute_global_arrived()
        output = self._local_arrived and global_arrived
        if output:
            # 达成一次全局同步后立刻复位，允许后续重用同一个 barrier
            self._local_arrived = False
        return output

    def _compute_global_arrived(self) -> bool:
        # noop 模式用于单机调试，直接视作本地已到达
        local_arrived = self._noop or self._local_arrived
        global_arrived = torch.tensor(local_arrived)
        # Can optimize if bottleneck
        torch.distributed.all_reduce(
            global_arrived,
            torch.distributed.ReduceOp.MIN,
            group=get_world_group().cpu_group,
        )
        # 只有所有参与方都到齐 (min=1) 时才算同步完成
        return global_arrived.item()
