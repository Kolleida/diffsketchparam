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

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, hash_embedding_config: HashEmbeddingConfig) -> None:
        super().__init__()
        self.embedding_layer = HashEmbedding(**asdict(hash_embedding_config), append_weight=False)
        print(self.embedding_layer.embedding_dim)
        layers = [nn.Linear(input_dim + self.embedding_layer.embedding_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.linear = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, keys):
        embeddings = self.embedding_layer(keys).squeeze()
        # print(embeddings.shape)
        # print(X.shape)
        inputs = torch.concat([embeddings, X], dim=-1)
        outputs = f.softplus(self.output_layer(self.linear(inputs)))
        return outputs
        

class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.ReLU) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation()
        self.linear2 = nn.Linear(hidden_dim, output_dim)


