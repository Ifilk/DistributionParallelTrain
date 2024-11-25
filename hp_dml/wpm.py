import torch

from hp_dml.dist import Node

from enum import IntEnum

class MetaMessageType(IntEnum):
    GROUP = 0
    P_AGG = 1
    P_SYN = 2

class ModuleProxy(Node):
    def train(self, meta_message):
        pass

    def update_model(self) -> torch.Tensor:
        pass