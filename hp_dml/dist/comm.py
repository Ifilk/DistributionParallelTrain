import functools
import threading
from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar, cast, Optional, Any

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
    - max_length: 最大消息长度，默认为8。
    - placeholder: 用于填充消息的值，默认为0。
    - _type: 消息的数据类型，默认为torch.int8。

    该类初始化时会根据提供的参数设置通信的超时时间和消息处理方式。
    """

    def __init__(self,
                 max_length: int = 8,
                 placeholder=0,
                 _type=torch.int8):
        self._rank = dist.get_rank()
        self.meta_message_shape = (max_length,)
        self.meta_message_handler = _build_meta_message_parser(_type, max_length, placeholder)
        self.meta_message_builder = _build_meta_message_builder(_type, max_length, placeholder)
        self._meta_type = -1
        self.send = functools.partial(self._send_tensor, send_with_timeout)
        self.recv = functools.partial(self._recv_tensor, recv_with_timeout)

        self._last_recv = None


    def _send_tensor(self, send_func, meta_type, tensor: torch.Tensor, dest, timeout=None, meta=True):
        if meta:
            send_func(self.meta_message_builder(MetaMessage(
                _type=meta_type,
                sender=self._rank,
                params=tuple(tensor.shape)
            )), dest, timeout)
        send_func(tensor, dest, timeout)

    def _recv_tensor(self, recv_func, src=None, timeout=None, meta=True, shape=None):
        if meta:
            tensor = torch.zeros(self.meta_message_shape)
            recv_func(tensor, src, timeout)
            meta_m = self.meta_message_handler(tensor)
            tensor = torch.zeros(meta_m.params)
            self._meta_type = meta_m.type
            recv_func(tensor, meta_m.sender, timeout)
        else:
            tensor = torch.zeros(shape)
            recv_func(tensor, src, timeout)
            self._meta_type = -1
        self._last_recv = tensor
        return tensor

    @property
    def last_recv(self):
        return self._last_recv

    @property
    def meta_type(self):
        return self._meta_type


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


global_comm = None


def _unimplemented_error():
    raise CommException('Must be initialized first')


send = lambda meta_type, tensor, dest, timeout=None, meta=True: _unimplemented_error()

recv = lambda src=None, timeout=None, meta=True: _unimplemented_error()

last_recv = lambda : _unimplemented_error()

get_meta_state = lambda : _unimplemented_error()


def init_process_group(
        backend: Optional[str] = None,
        init_method: Optional[str] = None,
        world_size: int = -1,
        rank: int = -1,
        store = None,
        group_name: str = "",
        pg_options: Optional[Any] = None,
        device_id: Optional[torch.device] = None,
        meta_message_length: int = 8,
        meta_message_placeholder=0,
        meta_message_type=torch.int8
):
    global global_comm, send, recv, last_recv, get_meta_state
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        store=store,
        group_name=group_name,
        pg_options=pg_options,
        device_id=device_id
    )

    global_comm = DistributionComm(
        max_length=meta_message_length,
        placeholder=meta_message_placeholder,
        _type=meta_message_type
    )

    send = global_comm.send
    recv = global_comm.recv
    last_recv = lambda : global_comm.last_recv
    get_meta_state: lambda : global_comm.meta_type