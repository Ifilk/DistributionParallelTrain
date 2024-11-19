import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Union, Optional

import torch
import torch.distributed as dist

from dpt.distribution_operation import get_results, apply_result


@dataclass
class Progress:
    """
        Represents a single stage in a distributed job pipeline.

        Attributes:
            master (bool): Indicates whether this stage is executed by the master process.
                           If True, this stage is exclusive to the master; otherwise, it is executed by workers.
            amount_of_parameters (int): The number of parameters required by the function in this stage.
                                         This is used to determine the input tensor count for the stage.
            recv_shape_list (List[Tuple[int, ...]]): A list of tensor shapes expected to be received by this stage.
                                                     Each tuple represents the dimensions of a tensor.
            func (Callable[..., Optional[Iterable]]): The function to be executed at this stage.
                                                      It may optionally return results to be broadcast or sent to another process.
    """
    master: bool = False
    amount_of_parameters: int = 0
    recv_shape_list: List[Tuple[int, ...]] = field(default_factory=list)
    func: Callable[..., Optional[Iterable]] = None


class Job:
    """
    Represents a distributed job pipeline for managing and executing tasks across multiple processes.

    Attributes:
        pipeline (List[Progress]): A list of `Progress` objects representing the stages of the pipeline.
        world_size (int): The total number of processes participating in the job.
        master_rank (int): The rank of the master process, responsible for coordinating tasks.

    Methods:
        master_progress(recv_shape_list: List[Optional[Tuple[int, ...], int]] = None) -> Callable:
            Decorator to register a stage to be executed only by the master process.

        worker_progress(recv_shape_list: List[Optional[Tuple[int, ...], int]] = None) -> Callable:
            Decorator to register a stage to be executed only by worker processes.

        public_progress() -> Callable:
            Decorator to register a stage to be executed by both master and worker processes.

        build() -> Callable:
            Builds and returns the pipeline executor function, which executes all registered stages in sequence.
    """

    def __init__(self,
                 world_size: int,
                 master_rank: int = 0):
        """
        Initializes a Job object.

        Args:
            world_size (int): The total number of processes participating in the distributed job.
            master_rank (int): The rank of the master process (default is 0).
        """
        self.pipeline: List[Progress] = []
        self.world_size = world_size
        self.master_rank = master_rank

    def master_progress(self, recv_shape_list: List[Union[Tuple[int, ...], int]] = None):
        """
        Registers a stage to be executed only by the master process.

        Args:
            recv_shape_list (List[Optional[Tuple[int, ...], int]]): A list specifying the shapes of tensors to be received
                by this stage. Each element is a tuple representing the dimensions of a tensor.

        Returns:
            Callable: A decorator to wrap the function for this stage.
        """

        def master_progress_wrapper(func):
            amount = len(inspect.signature(func).parameters.items())
            self.pipeline.append(Progress(
                master=True,
                amount_of_parameters=amount,
                recv_shape_list=recv_shape_list,
                func=func))

        return master_progress_wrapper

    def worker_progress(self, recv_shape_list: List[Union[Tuple[int, ...], int]] = None):
        """
        Registers a stage to be executed only by worker processes.

        Args:
            recv_shape_list (List[Optional[Tuple[int, ...], int]]): A list specifying the shapes of tensors to be received
                by this stage. Each element is a tuple representing the dimensions of a tensor.

        Returns:
            Callable: A decorator to wrap the function for this stage.
        """

        def worker_progress_wrapper(func):
            amount = len(inspect.signature(func).parameters.items())
            self.pipeline.append(Progress(
                amount_of_parameters=amount,
                recv_shape_list=recv_shape_list,
                func=func))

        return worker_progress_wrapper

    def public_progress(self):
        """
        Registers a stage to be executed by both the master and worker processes.

        Returns:
            Callable: A decorator to wrap the function for this stage.
        """

        def master_progress_wrapper(func):
            self.pipeline.extend([
                Progress(amount_of_parameters=0, func=func),
                Progress(master=True, amount_of_parameters=0, func=func)
            ])

        return master_progress_wrapper

    def build(self):
        """
        Builds and returns the pipeline executor function.

        The executor function iterates through the registered stages (`Progress` objects) and executes
        them based on their type (master or worker). It manages communication between processes and
        ensures correct data flow through the pipeline.

        Returns:
            Callable: A function that executes all registered stages in the pipeline.
        """

        def execute_job_pipeline():
            def broadcast(r, source):
                for i in range(0, self.world_size):
                    if i != source:
                        apply_result(*(r if isinstance(r, Iterable) else [r]), dest=i)

            for process in self.pipeline:
                rank = dist.get_rank()

                if rank == self.master_rank and process.master:
                    if process.amount_of_parameters > 0:
                        result = get_results(process.amount_of_parameters, self.world_size, process.recv_shape_list)
                        result = process.func(*result)
                    else:
                        result = process.func()
                    if result is not None:
                        if not isinstance(result, torch.Tensor):
                            raise ValueError('Only torch.Tensor could be send')
                        broadcast(result, self.master_rank)

                if rank != self.master_rank and not process.master:
                    if process.amount_of_parameters > 0:
                        result = get_results(process.amount_of_parameters, self.world_size, process.recv_shape_list)
                        result = [r if len(r) > 1 else r[0] for r in result]
                        result = process.func(*result)
                    else:
                        result = process.func()
                    if result is not None:
                        if not isinstance(result, torch.Tensor):
                            raise ValueError('Only torch.Tensor could be send')
                        apply_result(*(result if isinstance(result, Iterable) else [result]), dest=self.master_rank)

        return execute_job_pipeline
