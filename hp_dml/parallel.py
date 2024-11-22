import abc
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List
from inspect import isgenerator
from uuid import uuid4

import torch
from torch import distributed as dist
from multiprocessing import Process

@dataclass
class _MetaMessageHandler:
    func: Callable[[torch.Tensor], ...]
    multistage: bool
    mutex: bool
    sync: bool
    id: int

class SpreadExceptionWithNoReceiveConfirm(Exception):
    ...

class MetaMessageManager:
    def __init__(self, state: int, max_worker: int, dtype=torch.int):
        self._callable_list: Dict[int, List[_MetaMessageHandler]]\
            = {i: list() for i in range(0, state)}
        # self._time_out = time_out
        self._max_worker = max_worker
        self._dtype = dtype
        self._m_lock = threading.Lock()
        self._mutex_lock_dict: Dict[int, bool] = {}
        self._c_lock = threading.Lock()
        self._concurrency: Dict[int, int] = {}

        self._multistage_handler_buffer = deque()

    def add_handler(self, state: int, func: Callable[[torch.Tensor], ...],
                    multistage: bool, mutex: bool, sync: bool):
        mmh = _MetaMessageHandler(
            func=func,
            multistage=multistage,
            mutex=mutex,
            sync=sync,
            id=uuid4().int
        )
        self._callable_list[state].append(mmh)
        if mutex:
            with self._m_lock:
                self._mutex_lock_dict[mmh.id] = False
        if not sync:
            self._concurrency[mmh.id] = 0

    def handler(self, state: int, multistage: bool, mutex: bool, sync: bool=False):
        def wrapper_handler(func):
            # TODO sync support
            assert not sync, 'SYNC handler is not support now'
            self.add_handler(state, func, multistage, mutex, sync)
        return wrapper_handler

    def handle(self, meta_message: torch.Tensor):
        _state = meta_message.item()[0]
        for mmh in self._callable_list[_state]:
            if mmh.mutex:
                if self._mutex_lock_dict[mmh.id]:
                    # drop requirement for busy
                    return
                else:
                    if len(self._multistage_handler_buffer) != 0:
                        def wrapper():
                            _gen = self._multistage_handler_buffer.pop()
                            try:
                                next(_gen)
                                self._multistage_handler_buffer.appendleft(_gen)
                            except StopIteration:
                                with self._c_lock:
                                    self._concurrency[mmh.id] -= 1
                        if mmh.sync:
                            wrapper()
                        else:
                            p = Process(target=wrapper)
                            p.start()
                    else:
                        self._async_handle(meta_message, mmh)

    def _async_handle(self, meta_message: torch.Tensor, mmh: _MetaMessageHandler):
        if sum(self._concurrency.values()) >= self._max_worker:
            # drop requirement for no available worker
            return False
        def wrapper_func(_meta_message: torch.Tensor, _mmh: _MetaMessageHandler):
            with self._c_lock:
                self._concurrency[mmh.id] += 1
            if _mmh.multistage:
                _gen = _mmh.func(_meta_message)
                assert isgenerator(_gen), '_MetaMessageHandler.func must be generator if it is multistage'
                try:
                    next(_gen)
                    self._multistage_handler_buffer.append(_gen)
                except StopIteration:
                    with self._c_lock:
                        self._concurrency[mmh.id] -= 1
            else:
                try:
                    _mmh.func(_meta_message)
                finally:
                    with self._c_lock:
                        self._concurrency[mmh.id] -= 1

        p = Process(target=wrapper_func, args=(meta_message, mmh))
        p.start()
        return True

class Node(abc.ABC):
    def __init__(self, rank: int,
                 node_amount: int,
                 group_density: float,
                 insurance: int,
                 time_out: int,
                 max_cuda_device_count: int):
        self.neighbors = None
        self._is_in_group = False
        self._is_start_node = False
        self._is_end_node = False
        self._training_lock = threading.Lock()
        self._last_sender = -1
        self._last_receiver = -1
        self._is_training = False

        self._rank = rank
        self._node_amount = node_amount
        self._group_density = group_density
        self._spreading_message_insurance = insurance
        self._spreading_message_time_out = time_out
        self._max_cuda_device_count = max_cuda_device_count

    def _reset(self):
        with self._training_lock:
            self._is_training = False
            self._is_in_group = False
            self._is_start_node = False
            self._is_end_node = False
            self._last_sender = -1
            self._last_receiver = -1

    def activate(self):
        ...

    def before_group(self, meta_message: torch.Tensor):
        e = meta_message.item()
        self._last_sender = e[1]
        self._is_in_group = True
        with self._training_lock:
            self.train(meta_message)

    def safe_spread_group_message(self, group_message: torch.Tensor, dest):
        # TODO safely spread group message
        dist.send(group_message, dest)
        # raise SpreadExceptionWithNoReceiveConfirm()

    def group(self, group_message: torch.Tensor):
        valid_indices = (0 < group_message[:self._node_amount]) & (group_message[:self._node_amount] <= 8)
        valid_count = torch.sum(valid_indices).item()
        zero_count = torch.sum((group_message[:self._node_amount] == 0)).item()
        group_message[self._rank] = min(torch.cuda.device_count(), self._max_cuda_device_count)
        if valid_count == 0:
            self._is_end_node = True
        elif valid_count == group_message[-1].item():
            self._is_end_node = True
            self.neighbors = torch.nonzero(valid_indices).flatten().tolist()
            group_message = torch.Tensor([o + self._max_cuda_device_count + 1
                                          for o in group_message if 0 < o <= self._max_cuda_device_count])
        elif valid_count == self._node_amount - 1:
            self.activate()
            return
        try:
            lucky_neighbor = torch.randint(0, zero_count, (1,)).item()
            real_rank = 0
            z = 0
            for o in group_message:
                if z == lucky_neighbor:
                    break
                if o == 0:
                    z += 1
                real_rank += 1
            self.safe_spread_group_message(group_message, real_rank)
        except SpreadExceptionWithNoReceiveConfirm:
            # TODO retry and resume
            ...

    def after_group(self):
        if self._is_end_node:
            with self._training_lock:
                self.spread_model_parameters()

    def chain_train_spread(self, meta_message: torch.Tensor, w: torch.Tensor):
        m = self.get_model()
        self.set_model(self.aggregate(m, w))
        self._last_receiver = self._last_sender
        self._last_sender = meta_message[1].item()
        dist.send(self.get_model(), self._last_receiver)

    def chain_train_synchronize(self, w: torch.Tensor):
        self.set_model(w)
        dist.send(self.get_model(), self._last_sender)
        self._reset()

    @abc.abstractmethod
    def set_model(self, m: torch.Tensor):
        ...

    @abc.abstractmethod
    def get_model(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def aggregate(self, m1, m2) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def spread_model_parameters(self):
        ...

    @abc.abstractmethod
    def train(self, meta_message):
        ...

    @property
    def last_sender(self):
        return self._last_sender

    @property
    def last_receiver(self):
        return self._last_receiver


def receive_group_message() -> torch.Tensor:
    # TODO
    ...

def receive_model_parameters(meta_message: torch.Tensor, dest) -> torch.Tensor:
    # TODO
    ...


def wrapper_node_group(node: Node):
    def wrapper(meta_message: torch.Tensor):
        node.before_group(meta_message)
        yield
        group_message = receive_group_message()
        node.group(group_message)
        node.after_group()
    return wrapper

def wrapper_node_synchronize(node: Node):
    def wrapper(meta_message: torch.Tensor):
        w = receive_model_parameters(meta_message, node.last_sender)
        node.chain_train_spread(meta_message, w)
        yield
        w = receive_model_parameters(meta_message, node.last_receiver)
        node.chain_train_spread(meta_message, w)
    return wrapper