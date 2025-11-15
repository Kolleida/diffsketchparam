import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import Linear, ReLU, RMSNorm
from hash_embeddings import HashEmbedding
from dataclasses import dataclass, field, asdict
from collections import OrderedDict


@dataclass
class HashEmbeddingConfig:
    num_embeddings: int = field(default=10 ** 6)
    embedding_dim: int = field(default=10 ** 2)
    num_buckets: int = field(default=10 ** 4)
    num_hashes: int = field(default=4)


class SketchParameterPredictor(nn.Module):

    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        num_hidden_layers: int = 8, 
        hash_embedding_config: HashEmbeddingConfig = HashEmbeddingConfig()
    ) -> None:
        super().__init__()
        self.embedding_layer = HashEmbedding(**asdict(hash_embedding_config), append_weight=False)
        self.input_fc = nn.Linear(self.embedding_layer.embedding_dim + input_dim, hidden_dim)
        self.activation = nn.ReLU()
        layers = []
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), self.activation]
        self.fc_stack = nn.Sequential(*layers)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Softplus()

    def forward(self, X, keys):
        embeddings = self.embedding_layer(keys).squeeze()
        input = torch.concat([embeddings, X], dim=-1)
        proj_input = self.activation(self.input_fc(input))
        output = self.output_activation(self.output_fc(self.fc_stack(proj_input)))
        return output



