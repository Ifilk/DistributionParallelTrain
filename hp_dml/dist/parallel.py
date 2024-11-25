import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, TypeVar

from hp_dml.dist import comm

T = TypeVar('T')

from uuid import uuid4

from multiprocessing import Process

@dataclass
class Task:
    func: Callable[[comm.MetaMessage], ...]
    mutex: bool
    sync: bool
    id: int

class ProcessManager:
    def __init__(self, state: int, max_worker: int):
        self._callable_list: Dict[int, List[Task]]\
            = {i: list() for i in range(0, state)}
        # self._time_out = time_out
        self._max_worker = max_worker
        self._c_lock = threading.Lock()
        self._concurrency: Dict[int, int] = {}

    def add_handler(self, state: int, func: Callable[[comm.MetaMessage], ...],
                    mutex: bool, sync: bool):
        mmh = Task(
            func=func,
            mutex=mutex,
            sync=sync,
            id=uuid4().int
        )
        self._callable_list[state].append(mmh)
        if not sync:
            self._concurrency[mmh.id] = 0

    def handler(self, state: int, mutex: bool, sync: bool=False):
        def wrapper_handler(func):
            # TODO sync support
            assert not sync, 'SYNC handler is not support now'
            self.add_handler(state, func, mutex, sync)
        return wrapper_handler

    def handle(self, meta_message: comm.MetaMessage):
        _state = meta_message.type
        for mmh in self._callable_list[_state]:
            if mmh.mutex:
                if self._concurrency[mmh.id] == 0:
                    self._async_handle(meta_message, mmh)
                else:
                    # drop requirement for busy
                    return
            else:
                self._async_handle(meta_message, mmh)

    def _async_handle(self, meta_message: comm.MetaMessage, mmh: Task):
        if sum(self._concurrency.values()) >= self._max_worker:
            # drop requirement for no available worker
            return False
        def wrapper_func(_meta_message: comm.MetaMessage, _mmh: Task):
            with self._c_lock:
                self._concurrency[mmh.id] += 1
                try:
                    _mmh.func(_meta_message)
                finally:
                    with self._c_lock:
                        self._concurrency[mmh.id] -= 1

        p = Process(target=wrapper_func, args=(meta_message, mmh))
        p.start()
        return True