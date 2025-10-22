import logging
import threading
import time
from abc import ABC
from contextlib import contextmanager, nullcontext

try:
    import torch_memory_saver

    _memory_saver = torch_memory_saver.torch_memory_saver
    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        if enable and import_error is not None:
            logger.warning(
                "enable_memory_saver is enabled, but "
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )
            # 用户显式打开但缺包时直接抛出异常，防止误以为内存规避已生效
            raise import_error
        return (
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        )

    def check_validity(self, caller_name):
        if not self.enabled:
            logger.warning(
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )
            # 仅告警不抛错，方便在可选依赖缺失时继续执行

    def configure_subprocess(self):
        raise NotImplementedError

    def region(self, tag: str):
        raise NotImplementedError

    def pause(self, tag: str):
        raise NotImplementedError

    def resume(self, tag: str):
        raise NotImplementedError

    @property
    def enabled(self):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    """Adapter for TorchMemorySaver with tag-based control"""

    def configure_subprocess(self):
        # 真正创建子进程前需要让 torch-memory-saver 注入钩子
        return torch_memory_saver.configure_subprocess()

    def region(self, tag: str):
        # 用 tag 区分不同的保存区间，便于分段释放
        return _memory_saver.region(tag=tag)

    def pause(self, tag: str):
        return _memory_saver.pause(tag=tag)

    def resume(self, tag: str):
        return _memory_saver.resume(tag=tag)

    @property
    def enabled(self):
        return _memory_saver is not None and _memory_saver.enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str):
        yield

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
