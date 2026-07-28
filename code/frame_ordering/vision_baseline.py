import itertools
from typing import Iterable, Tuple

import torch
from torch import nn

from frame_ordering.permutations import Permutation, validate_permutation

PERMUTATIONS: Tuple[Permutation, ...] = tuple(
    itertools.permutations((1, 2, 3, 4))
)


def ranks_to_class(ranks: Iterable[int]) -> int:
    return PERMUTATIONS.index(validate_permutation(ranks))


def class_to_ranks(class_index: int) -> Permutation:
    if not 0 <= class_index < len(PERMUTATIONS):
        raise ValueError(f"Invalid class index: {class_index}")
    return PERMUTATIONS[class_index]


def reorder_feature_batch(
    features: torch.Tensor,
    ranks: torch.Tensor,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 3 or features.shape[1] != 4:
        raise ValueError(f"Expected features shaped (B,4,D), got {features.shape}")
    if ranks.shape != features.shape[:2]:
        raise ValueError(f"Ranks shape {ranks.shape} does not match features")
    order = torch.rand(
        features.shape[0], 4, generator=generator
    ).argsort(dim=1)
    return (
        features.gather(1, order.unsqueeze(-1).expand_as(features)),
        ranks.gather(1, order),
    )


def score_rank_logits(rank_logits: torch.Tensor) -> torch.Tensor:
    if rank_logits.ndim != 3 or rank_logits.shape[1:] != (4, 4):
        raise ValueError(f"Expected rank logits shaped (B,4,4), got {rank_logits.shape}")
    permutations = torch.tensor(
        PERMUTATIONS, device=rank_logits.device, dtype=torch.long
    ) - 1
    batch_size = rank_logits.shape[0]
    indices = permutations.view(1, 24, 4, 1).expand(
        batch_size, -1, -1, -1
    )
    expanded = rank_logits.unsqueeze(1).expand(-1, 24, -1, -1)
    return expanded.gather(-1, indices).squeeze(-1).sum(dim=-1)


class VisionOrderModel(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dim: int, num_layers: int = 2
    ) -> None:
        super().__init__()
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, 4, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.rank_head = nn.Linear(hidden_dim, 4)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.frame_projection(features) + self.position_embedding
        rank_logits = self.rank_head(self.encoder(hidden))
        return score_rank_logits(rank_logits)
