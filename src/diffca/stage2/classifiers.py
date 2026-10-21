"""Referee classifiers on the conditioning tokens (32 x 128). They score the swaps during separator training.

Parameter names are those of the original code, so the trained classifier checkpoints load unchanged.
"""

import torch
import torch.nn as nn

FFHQ_ATTRIBUTES = ("glasses", "smile", "gender", "headPose")
ATTRIBUTES = FFHQ_ATTRIBUTES + ("afhq", "brats")


class LinearProbe(nn.Module):
    """LayerNorm on each token, then one linear layer on the flattened tokens (glasses, smile, gender)."""

    def __init__(self, num_tokens: int = 32, token_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(num_tokens * token_dim, num_classes))
        self.layernorm = nn.LayerNorm(token_dim)

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        context_tokens = self.layernorm(context_tokens)
        return self.classifier(context_tokens.view(context_tokens.shape[0], -1))


class MLPProbe(nn.Module):
    """LayerNorm on each token, then a 2-hidden-layer MLP on the flattened tokens (AFHQ, BraTS)."""

    def __init__(self, num_tokens: int = 32, token_dim: int = 128, hidden_dim: int = 128, dropout_rate: float = 0.5,
                 num_classes: int = 3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_tokens * token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )
        self.layernorm = nn.LayerNorm(token_dim)

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        context_tokens = self.layernorm(context_tokens)
        return self.classifier(context_tokens.view(context_tokens.shape[0], -1))


class YawRegressor(nn.Module):
    """MLP regressing the head yaw (degrees) from the flattened tokens."""

    def __init__(self, num_tokens: int = 32, token_dim: int = 128, hidden_dim: int = 512, dropout_rate: float = 0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(num_tokens * token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        return self.regressor(context_tokens.view(context_tokens.shape[0], -1)).squeeze(-1)


def build_referee(attribute: str, num_tokens: int = 32, num_classes: int | None = None) -> nn.Module:
    """The referee architecture used for `attribute` (num_classes may be overridden for glasses)."""
    if attribute == "headPose":
        return YawRegressor(num_tokens=num_tokens)
    if attribute == "glasses":
        return LinearProbe(num_tokens=num_tokens, num_classes=num_classes or 3)
    if attribute in ("smile", "gender"):
        return LinearProbe(num_tokens=num_tokens, num_classes=2)
    if attribute == "afhq":
        return MLPProbe(num_tokens=num_tokens, num_classes=4)
    if attribute == "brats":
        return MLPProbe(num_tokens=num_tokens, num_classes=2)
    raise ValueError(f"Attribute not supported: {attribute}")
