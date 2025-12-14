from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Literal
import yaml

from hash_embeddings import HashEmbeddingParams
from model import FeedForwardPredictorParams
from cattrs import structure


@dataclass
class OptimizerParams:
    optimizer_type: Literal["AdamW", "SGD"] = "AdamW"
    optimizer_options: dict = field(default_factory=lambda: {"lr": 5e-3})
    use_scheduler: bool = True


@dataclass
class TrainingParams:
    batch_size: int = 1024
    epochs: int = 5
    logging_frequency: int = 400
    shuffle: bool = True
    device: str = "cuda"
    optimizer_parms: OptimizerParams = field(default_factory=OptimizerParams)


@dataclass
class ModelConfig:
    model_type: str = "FeedForwardPredictor"
    params: FeedForwardPredictorParams = field(default_factory=FeedForwardPredictorParams)


@dataclass
class Config:
    training_params: TrainingParams = field(default_factory=TrainingParams)
    model_config: ModelConfig = field(default_factory=ModelConfig)    

    @classmethod
    def from_yaml(cls, file_path: str) -> 'Config':
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return structure(data, cls)