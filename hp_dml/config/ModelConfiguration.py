import enum
from dataclasses import dataclass
from typing import Optional


class IterMethod(str, enum.Enum):
    Iteration = 'iteration'
    Epoch = 'epoch'


@dataclass
class ModelConfiguration:
    model_name: str
    learning_rate: float
    iter_method: Optional[str, IterMethod]