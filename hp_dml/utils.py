from collections import OrderedDict
from datetime import datetime
import random
from typing import Union

import numpy as np
import torch


def get_data_stamp() -> str:
    return datetime.now().strftime('%Y%m%d%H%M%S')

def set_random_seed(seed: int, device: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'gpu':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def clone_parameters(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )

def tensor_power(tensor: torch.tensor, p: float):
    singp = torch.sign(tensor)
    temp = (tensor.abs())**p
    temp.mul_(singp)
    return temp

def model_power(model, p1, p2):
    for name in model.keys():
        if 'conv' in name:
            model[name] = tensor_power(model[name], p1)
        else:
            model[name] = tensor_power(model[name], p2)
    return model


def weighted_sum(tensors, weights):
    weighted_tensors = [tensor * weight for tensor, weight in zip(tensors, weights)]
    result = torch.sum(torch.stack(weighted_tensors), dim=0)
    return result