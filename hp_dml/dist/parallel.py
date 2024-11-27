import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, TypeVar, Deque

T = TypeVar('T')

from uuid import uuid4
from hp_dml.logger import logger
from multiprocessing import Process


class NoAvailableWorkerNow(Exception):
    ...


@dataclass
class Task:
    func: Callable[[], ...]
    mutex: bool
    sync: bool
    id: int


class ProcessManager:
    def __init__(self,
                 state: int,
                 max_worker: int,
                 queue_size: int):
        self._callable_list: Dict[int, List[Task]] \
            = {i: list() for i in range(0, state)}
        # self._time_out = time_out
        self._max_worker = max_worker
        self._c_lock = threading.Lock()
        self._queue_size = queue_size
        self._concurrency: Dict[int, int] = {}
        self._requirements_queue_for_mutex: Deque[Task] = deque()

        self._flash_lock = threading.Lock()
        self._flash_process = None

    def add_handler(self, state: int, func: Callable[[], ...],
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

    def handler(self, state: int, mutex: bool, sync: bool = False):
        def wrapper_handler(func):
            self.add_handler(state, func, mutex, sync)

        return wrapper_handler

    def handle(self, _state):
        if _state not in self._callable_list:
            raise ValueError(f'Unknown state {_state}')
        for task in self._callable_list[_state]:
            if sum(self._concurrency.values()) >= self._max_worker:
                raise NoAvailableWorkerNow(f'No available worker for Task {task.id}')
            self.exec_task(task)

    def exec_task(self, task):
        if task.sync:
            task.func()
        elif task.mutex:
            if self._concurrency[task.id] == 0:
                self._async_handle(task)
            else:
                if len(self._requirements_queue_for_mutex) >= self._queue_size:
                    raise NoAvailableWorkerNow(f'No available worker for Task {task.id}')

                self._requirements_queue_for_mutex.append(task)
                with self._flash_lock:
                    if self._flash_process is None:
                        self._flash_process = Process(target=self._flash_requirements_queue_when_available,
                                                      args=(self._requirements_queue_for_mutex,)).start()
        else:
            self._async_handle(task)

    def _flash_requirements_queue_when_available(self, queue: Deque[Task]):
        while len(queue) > 0:
            task = queue.pop()
            try:
                self.exec_task(task)
            except NoAvailableWorkerNow:
                queue.appendleft(task)
            except Exception as e:
                logger.error(f'Occurring exception when handle {task} : {e}')
        with self._flash_lock:
            self._flash_process = None

    def _async_handle(self, task: Task):
        def wrapper_func():
            with self._c_lock:
                self._concurrency[task.id] += 1
                try:
                    task.func()
                finally:
                    with self._c_lock:
                        self._concurrency[task.id] -= 1

        p = Process(target=wrapper_func)
        p.start()
        return True
