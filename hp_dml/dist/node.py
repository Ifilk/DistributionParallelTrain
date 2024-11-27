import abc
import threading
import torch
from torch import nn

from hp_dml.dist import send, recv, CommException, MetaMessage
from hp_dml.logger import logger

class SendException(Exception):
    ...

class Node(abc.ABC):
    def __init__(self, rank: int,
                 node_amount: int,
                 group_density: float,
                 model: nn.Module,
                 # insurance: int,
                 max_retry: int=4,
                 max_cuda_device_count: int=8):
        self._max_retry = max_retry
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
        # self._spreading_message_insurance = insurance
        self._max_cuda_device_count = max_cuda_device_count

        self.model = model

    def _reset(self):
        with self._training_lock:
            self._is_training = False
            self._is_in_group = False
            self._is_start_node = False
            self._is_end_node = False
            self._last_sender = -1
            self._last_receiver = -1

    def activate(self):
        group_message = torch.zeros((self._node_amount, ))
        group_message[self._rank] = min(torch.cuda.device_count(), self._max_cuda_device_count)
        zero_count = torch.sum((group_message[:self._node_amount] == 0)).item()
        lucky_neighbor = torch.randint(0, zero_count, (1,)).item()
        self._is_start_node = True
        real_rank = 0
        z = 0
        for o in group_message:
            if z == lucky_neighbor:
                break
            if o == 0:
                z += 1
            real_rank += 1
        self.safe_spread_group_message(group_message, real_rank)

    def before_group(self, meta_message: MetaMessage):
        self._last_sender = meta_message.sender
        self._is_in_group = True
        with self._training_lock:
            self.train(meta_message)

    def safe_spread_group_message(self, group_message: torch.Tensor, dest):
        retry_count = 1
        _exception = None
        while self._max_retry >= retry_count:
            try:
                send(group_message, dest)
                return
            except CommException as ce:
                logger.warn(f'Exception occurring while group message sending\n {ce}')
                _exception = ce
                if self._max_retry >= retry_count:
                    logger.info(f'Start retrying, current round: {retry_count}')
                    retry_count += 1
        raise SendException(str(_exception))
    def group(self, group_message: torch.Tensor):
        valid_indices = (0 < group_message[:self._node_amount]) & (group_message[:self._node_amount] <= 8)
        valid_count = torch.sum(valid_indices).item()
        group_message[self._rank] = min(torch.cuda.device_count(), self._max_cuda_device_count)
        zero_count = torch.sum((group_message[:self._node_amount] == 0)).item()
        if valid_count == 0:
            self._is_start_node = True
        elif valid_count == group_message[-1].item():
            self._is_end_node = True
            self.neighbors = torch.nonzero(valid_indices).flatten().tolist()
            group_message = torch.Tensor([o + self._max_cuda_device_count + 1
                                          for o in group_message if 0 < o <= self._max_cuda_device_count])
        elif valid_count == self._node_amount - 1:
            self.activate()
            return
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

    def after_group(self):
        if self._is_end_node:
            with self._training_lock:
                self.send_model(self._last_sender)

    def chain_train_spread(self, meta_message: MetaMessage):
        self.update_model()
        self._last_receiver = self._last_sender
        self._last_sender = meta_message.sender
        self.send_model(self._last_receiver)

    def chain_train_synchronize(self):
        self.sync_model()
        self.send_model(self._last_sender)
        self._reset()

    def sync_model(self):
        for p in self.model.parameters():
            tensor = recv(p.data, self.last_receiver)
            p.data = tensor

    def send_model(self, dest):
        for p in self.model.parameters():
            send(p.data, dest)

    @abc.abstractmethod
    def update_model(self) -> torch.Tensor:
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