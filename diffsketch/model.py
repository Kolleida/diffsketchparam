import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import Linear, ReLU, RMSNorm
from .hash_embeddings import HashEmbedding
from dataclasses import dataclass, field, asdict
from .hash_embeddings import HashEmbeddingParams


@dataclass
class FeedForwardPredictorParams:
    input_dim: int = 2
    hidden_dim: int = 1024
    output_dim: int = 2
    num_hidden_layers: int = 8
    hash_embedding_params: HashEmbeddingParams = field(default_factory=HashEmbeddingParams)


class FeedForwardPredictor(nn.Module):
    """
    Predicts CountMin Sketch parameters from input features using a feedforward neural network with hash embeddings.
    Keys are embedded using hash embeddings and concatenated with input (nuermic) features before being passed through the network.
    
    Args:
        config (FeedForwardPredictorParams): Configuration parameters for the model.
    Inputs:
        X (torch.Tensor): Input feature tensor of shape (batch_size, input_dim).
        keys (torch.Tensor): Tensor of keys for hash embeddings of shape (batch_size, num_keys).
    Outputs:
        torch.Tensor: Predicted sketch parameters of shape (batch_size, output_dim).
    """

    def __init__(
        self, 
        config: FeedForwardPredictorParams
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding_layer = HashEmbedding(**asdict(config.hash_embedding_params), append_weight=False)
        self.input_fc = nn.Linear(self.embedding_layer.embedding_dim + config.input_dim, config.hidden_dim)
        self.activation = nn.ReLU()
        layers = []
        for _ in range(config.num_hidden_layers):
            layers += [nn.Linear(config.hidden_dim, config.hidden_dim), self.activation]
        self.fc_stack = nn.Sequential(*layers)
        self.output_fc = nn.Linear(config.hidden_dim, config.output_dim)
        self.output_activation = nn.Softplus()

    def forward(self, X, keys):
        embeddings = self.embedding_layer(keys).squeeze()
        input = torch.concat([embeddings, X], dim=-1)
        proj_input = self.activation(self.input_fc(input))
        output = self.output_activation(self.output_fc(self.fc_stack(proj_input)))
        return output
    
    def save(self, path: str) -> None:
        model_info = {
            "config": self.config,
            "state_dict": self.state_dict()
        }
        torch.save(model_info, path)

    @classmethod
    def load(cls, path: str) -> FeedForwardPredictor:
        model_info = torch.load(path, weights_only=False)
        model = cls(model_info["config"])
        model.load_state_dict(model_info["state_dict"])
        return model



