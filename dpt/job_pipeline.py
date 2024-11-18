from typing import List

from dpt.job_manager import Job
import torch

def job_init(world_size):
    job = Job(world_size)

    @job.public_progress()
    def data_parallel():
        ...

    @job.worker_progress()
    def calculate_p():
        p_loc = 0
        # simulated
        return p_loc

    @job.master_progress([world_size - 1])
    def train_parameters_broadcast(p_loc_list: List[torch.Tensor]):
        return p_loc_list

    @job.worker_progress([...])
    def train(p_loc_list):
        model = None
        gradient = None
        # simulated
        return model, gradient

    @job.master_progress([world_size - 1, [...]])
    def aggregate_parameters_broadcast(model_list, gradients):
        return model_list, gradients

    @job.worker_progress([...])
    def aggregate(model_list, gradients):
        aggregated_model = None
        alpha = None
        # simulated
        return aggregated_model, alpha

    @job.master_progress([...])
    def gradiant_step_parameters_broadcast(aggregated_model_list, alpha_list):
        return aggregated_model_list, alpha_list

    @job.worker_progress([...])
    def gradiant_step(aggregated_model_list, alpha_list):
        # simulated
        ...

    return job.build()