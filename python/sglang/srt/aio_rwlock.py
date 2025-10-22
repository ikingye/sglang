import asyncio


class RWLock:
    def __init__(self):
        # Protects internal state
        self._lock = asyncio.Lock()

        # Condition variable used to wait for state changes
        # 使用同一个底层锁保证等待与状态写入原子化
        self._cond = asyncio.Condition(self._lock)

        # Number of readers currently holding the lock
        self._readers = 0

        # Whether a writer is currently holding the lock
        self._writer_active = False

        # How many writers are queued waiting for a turn
        self._waiting_writers = 0

    @property
    def reader_lock(self):
        """
        A context manager for acquiring a shared (reader) lock.

        Example:
            async with rwlock.reader_lock:
                # read-only access
        """
        return _ReaderLock(self)

    @property
    def writer_lock(self):
        """
        A context manager for acquiring an exclusive (writer) lock.

        Example:
            async with rwlock.writer_lock:
                # exclusive access
        """
        return _WriterLock(self)

    async def acquire_reader(self):
        async with self._lock:
            # Wait until there is no active writer or waiting writer
            # to ensure fairness.
            while self._writer_active or self._waiting_writers > 0:
                # 有写者活跃或排队时禁止新增读者，避免饥饿
                await self._cond.wait()
            self._readers += 1

    async def release_reader(self):
        async with self._lock:
            self._readers -= 1
            # If this was the last reader, wake up anyone waiting
            # (potentially a writer or new readers).
            if self._readers == 0:
                # 最后一个读者离开后唤醒等待的写者/读者
                self._cond.notify_all()

    async def acquire_writer(self):
        async with self._lock:
            # Increment the count of writers waiting
            self._waiting_writers += 1
            try:
                # Wait while either a writer is active or readers are present
                while self._writer_active or self._readers > 0:
                    # 写者等待所有读者释放，确保独占语义
                    await self._cond.wait()
                self._writer_active = True
            finally:
                # Decrement waiting writers only after we've acquired the writer lock
                self._waiting_writers -= 1

    async def release_writer(self):
        async with self._lock:
            self._writer_active = False
            # Wake up anyone waiting (readers or writers)
            self._cond.notify_all()


class _ReaderLock:
    def __init__(self, rwlock: RWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_reader()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_reader()


class _WriterLock:
    def __init__(self, rwlock: RWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_writer()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_writer()
