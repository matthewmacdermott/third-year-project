"""Pairwise Deep Q-Network for multi-objective HLS optimization.

The network takes concatenated stencil and config features and returns one
Q-value per objective.
"""

import torch
import torch.nn as nn
from typing import List


class DQNNetwork(nn.Module):
    """Pairwise multi-objective DQN.

    Input: concatenated stencil and normalised config features.
    Output: one Q-value per objective.
    """

    def __init__(self, stencil_dim: int, config_dim: int,
                 hidden_dims: List[int] = [128, 64], num_objectives: int = 6):
        super().__init__()

        self.stencil_dim = stencil_dim
        self.config_dim = config_dim
        self.num_objectives = num_objectives
        input_dim = stencil_dim + config_dim

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h

        self.network = nn.Sequential(*layers)
        self.output_head = nn.Linear(prev, num_objectives)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, stencil_dim + config_dim)  →  (batch, num_objectives)."""
        return self.output_head(self.network(x))
