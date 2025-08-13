from dataclasses import dataclass
from typing import Callable, Dict, Literal

from gpytorch.models.exact_gp import ExactGP
from omegaconf import OmegaConf

from hydra.core.config_store import ConfigStore

from ..models.gp.models import GPModel
from ..preprocessing.scalers import StandardScaler_toTensor


@dataclass
class Scalers:
    features: StandardScaler_toTensor
    target: StandardScaler_toTensor


@dataclass
class Data:
    training: Dict[str, Callable]
    inference: Dict[str, Callable]


@dataclass
class Preprocessing:
    train_test_split: Callable
    training: Callable
    inference: Callable


@dataclass
class PamirConfig:
    target: str
    features: list[str]
    idx_rbf: list[int]
    num_rbf: int
    idx_linear: list[int]
    idx_categorical: list[int]
    num_categories: int
    seed: int
    learning_rate: float
    scalers: Scalers
    data: Data
    preprocessing: Preprocessing
    output_dir: str
    device: str  # e.g., 'cpu' or 'cuda:0'
    mlflow: dict


def process_config(cfg: PamirConfig) -> PamirConfig:
    """
    Process the PamirConfig to ensure all fields are properly initialized.
    This can include setting defaults or validating values.
    """
    from munch import munchify

    from hydra.utils import instantiate

    from .config import MunchRich

    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = instantiate(cfg)
    cfg = munchify(cfg, factory=MunchRich)

    return cfg


def get_class_name(class_path: str) -> str:
    return class_path.split(".")[-1]


def get_hash(text: str) -> str:
    import hashlib

    text = str(text)
    hash_object = hashlib.sha1(text.encode())
    return hash_object.hexdigest()[:8]  # Return the first 8 characters


cs = ConfigStore.instance()
cs.store(name="config", node=PamirConfig)

OmegaConf.register_new_resolver("class_name", get_class_name, replace=True)
OmegaConf.register_new_resolver("hash", get_hash, replace=True)
