import functools
import threading
from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar, cast

T = TypeVar('T')

import torch
from torch import distributed as dist


class CommException(Exception):
    ...

class CommTimeOutException(CommException):
    ...

@dataclass
class MetaMessage(Generic[T]):
    _type: T
    sender: T
    params: Tuple[T] = ()

    @property
    def type(self):
        return self._type

class DistributionComm:
    """
    分布式通信类，用于初始化和管理分布式环境下的通信过程。

    参数:
    - rank: 当前进程的排名，用于标识进程。
    - timeout: 通信超时时间，如果为None，则没有超时限制。
    - max_length: 最大消息长度，默认为8。
    - placeholder: 用于填充消息的值，默认为0。
    - _type: 消息的数据类型，默认为torch.int8。

    该类初始化时会根据提供的参数设置通信的超时时间和消息处理方式。
    """
    def __init__(self,
                 rank,
                 timeout=None,
                 max_length: int = 8,
                 placeholder=0,
                 _type=torch.int8):
        self._rank = rank
        self._timeout = timeout
        self._meta_message_shape = (max_length,)
        self._meta_message_handler = _build_meta_message_parser(_type, max_length, placeholder)
        self._meta_message_builder = _build_meta_message_builder(_type, max_length, placeholder)
        self._meta_type = -1
        if timeout is not None:
            self.send = functools.partial(self._send_tensor, send_with_timeout)
            self.recv = functools.partial(self._recv_tensor, recv_with_timeout)
        else:
            self.send = functools.partial(self._send_tensor,
                                          lambda tensor, dest, *_, **__: dist.send(tensor, dest))
            self.recv = functools.partial(self._recv_tensor,
                                          lambda tensor, dest, *_, **__: dist.recv(tensor, dest))

    @property
    def get_meta_type(self):
        return self._meta_type

    def _send_tensor(self, send_func, meta_type, tensor: torch.Tensor, dest):
        send_func(self._meta_message_builder(MetaMessage(
            _type=meta_type,
            sender=self._rank,
            params=tuple(tensor.shape)
        )), dest, self._timeout)
        send_func(tensor, dest, self._timeout)

    def _recv_tensor(self, recv_func, src=None):
        tensor = torch.zeros(self._meta_message_shape)
        recv_func(tensor, src, self._timeout)
        meta = self._meta_message_handler(tensor)
        tensor = torch.zeros(meta.params)
        self._meta_type = meta.type
        recv_func(tensor, meta.sender, self._timeout)
        return tensor


def send_with_timeout(tensor: torch.Tensor, dest, timeout):
    success = threading.Event()

    def send():
        try:
            dist.send(tensor, dest)
            success.set()
        except Exception as e:
            ...

    send_thread = threading.Thread(target=send)
    send_thread.start()
    send_thread.join(timeout)
    if not success.is_set():
        raise CommTimeOutException(f"Send operation to rank {dest} timed out after {timeout} seconds")

    return True


def recv_with_timeout(tensor: torch.Tensor, dest, timeout):
    success = threading.Event()

    def send():
        try:
            dist.recv(tensor, dest)
            success.set()
        except Exception as e:
            ...

    send_thread = threading.Thread(target=send)
    send_thread.start()
    send_thread.join(timeout)
    if not success.is_set():
        raise CommTimeOutException(f"Recv operation to rank {dest} timed out after {timeout} seconds")

    return True


def _build_meta_message_parser(_type, max_length: int, placeholder):
    ptype = None
    if _type in (torch.int, torch.int8, torch.int16, torch.int32, torch.int64):
        ptype = int
    elif _type in (torch.float, torch.float16, torch.float32, torch.float64):
        ptype = float

    assert isinstance(placeholder, ptype), f'The placeholder{placeholder} must be corresponding to {str(_type)}'

    def meta_message_parser(meta_message: torch.Tensor):
        if meta_message.shape == (max_length,):
            raise CommException('Exceptional meta-message received')

        return MetaMessage(_type=meta_message[0].item(), sender=meta_message[1].item(),
                           params=cast(Tuple[T],
                                       tuple([i.item() for i in meta_message[2:max_length - 1] if i != placeholder])))

    return meta_message_parser


def _build_meta_message_builder(_type, max_length: int, placeholder):
    ptype = None
    if _type in (torch.int, torch.int8, torch.int16, torch.int32, torch.int64):
        ptype = int
    elif _type in (torch.float, torch.float16, torch.float32, torch.float64):
        ptype = float

    assert isinstance(placeholder, ptype), f'The placeholder{placeholder} must be corresponding to {str(_type)}'

    def meta_message_builder(mm: MetaMessage):
        if len(mm.params) + 2 > max_length:
            raise CommException('Too many parameters to send')

        if not (isinstance(mm.type, ptype) and
                isinstance(mm.sender, ptype) and
                all(isinstance(item, int) for item in mm.params)):
            raise CommException('Type Error')
        _data = [mm.type, mm.sender, *mm.params]
        if len(_data) < max_length:
            _data.extend([placeholder for _ in range(max_length)])
        _mm = torch.tensor(_data, dtype=_type)
        return _mm

    return meta_message_builder
