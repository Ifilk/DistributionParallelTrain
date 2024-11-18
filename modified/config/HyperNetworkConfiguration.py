from dataclasses import dataclass
from os import PathLike


@dataclass(frozen=True)
class HyperNetworkConfiguration:
    """
    hn configuration class
    """
    world_size: int
    embedding_dim: int
    hidden_dim: int
    cache_dir: PathLike
    hn_lr: float