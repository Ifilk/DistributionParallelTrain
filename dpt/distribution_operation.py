import datetime
import functools
from typing import List, Tuple, Callable, Union
import torch
import torch.distributed as dist


def apply_result(*data: torch.Tensor, dest: int = 0) -> None:
    """
    Send multiple tensors to a destination process.

    Args:
        *data: Variable-length argument list of tensors to send.
        dest: The rank of the destination process (default is 0).

    Raises:
        RuntimeError: If the distributed environment is not initialized.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment is not initialized.")

    for d in data:
        dist.send(d, dest)


def get_results(
        amount: int,
        world_size: int,
        result_size_list: List[Union[Tuple[int, ...], int]],
        time_out: float = None
) -> List[Tuple[torch.Tensor]]:
    """
    Receive multiple tensors from other processes and organize them, with an optional timeout.

    Args:
        amount: The number of tensor groups to receive.
        world_size: Total number of processes in the distributed group.
        result_size_list: A list of shapes for the tensors to be received,
                          where each shape is represented as a tuple of integers.
        time_out: Maximum time (in seconds) to wait for each tensor to be received.
                  If None, wait indefinitely.

    Returns:
        A list of tuples, where each tuple contains tensors from a specific
        group, organized by the sending process.

    Raises:
        RuntimeError: If the distributed environment is not initialized.
        RuntimeError: If the receive operation times out.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment is not initialized.")

    results = []
    timeout_delta = None
    if time_out is not None:
        timeout_delta = datetime.timedelta(seconds=time_out)

    for pth in range(amount):
        current_round = []
        for i in range(1, world_size):
            tensor = torch.zeros(result_size_list[pth])
            try:
                dist.recv(tensor, src=i, timeout=timeout_delta)
                current_round.append(tensor)
            except RuntimeError as e:
                raise RuntimeError(f"Timeout while receiving tensor from process {i}: {e}")
        results.append(current_round)

    return list(zip(*results))


def build_receiver(result_size_list: List[Union[Tuple[int, ...], int]]) -> Callable[[int, int], List[Tuple[torch.Tensor]]]:
    """
    Create a partially applied function for receiving tensors.

    Args:
        result_size_list: A list of shapes for the tensors to be received,
                          where each shape is represented as a tuple of integers.

    Returns:
        A partially applied function of `get_results`, taking `amount` and
        `world_size` as arguments.
    """
    return functools.partial(get_results, result_size_list=result_size_list)




