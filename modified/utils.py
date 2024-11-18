from datetime import datetime
import random
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