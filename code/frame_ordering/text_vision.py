import ast
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from frame_ordering.permutations import validate_permutation
from frame_ordering.vision_baseline import (
    ranks_to_class,
    reorder_feature_batch,
    score_rank_logits,
)


class TextVisionOrderModel(nn.Module):
    """Frame tokens cross-attend caption tokens; rank logits feed the
    24-way valid-permutation scorer shared with the vision baseline."""

    def __init__(
        self,
        frame_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, hidden_dim),
            nn.GELU(),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, 4, hidden_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.rank_head = nn.Linear(hidden_dim, 4)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        frame_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        if frame_features.ndim != 3 or frame_features.shape[1] != 4:
            raise ValueError(
                f"Expected frames shaped (B,4,D), got {frame_features.shape}"
            )
        if text_mask.shape != text_features.shape[:2]:
            raise ValueError(
                f"Text mask {text_mask.shape} does not match {text_features.shape}"
            )
        frames = self.frame_projection(frame_features) + self.position_embedding
        text = self.text_projection(text_features)
        decoded = self.decoder(
            tgt=frames,
            memory=text,
            memory_key_padding_mask=~text_mask.bool(),
        )
        return score_rank_logits(self.rank_head(decoded))


def _load_ids(feature_path: Path):
    return json.loads(
        feature_path.with_suffix(".ids.json").read_text(encoding="utf-8")
    )


class CachedTextVisionDataset(Dataset):
    def __init__(
        self,
        frame_feature_path: Path,
        text_feature_path: Path,
        train_csv: Path,
        split_csv: Path,
        split: str,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(f"Unknown split: {split}")
        self.frames = np.load(frame_feature_path, mmap_mode="r")
        self.text = np.load(text_feature_path, mmap_mode="r")
        self.mask = np.load(
            text_feature_path.with_suffix(".mask.npy"), mmap_mode="r"
        )
        frame_ids = _load_ids(frame_feature_path)
        text_ids = _load_ids(text_feature_path)
        labels = []
        csv_ids = []
        with train_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                csv_ids.append(row["Id"])
                labels.append(
                    validate_permutation(ast.literal_eval(row["Answer"]))
                )
        if frame_ids != csv_ids or text_ids != csv_ids:
            raise ValueError("Cache IDs do not exactly match train.csv IDs")
        if self.frames.shape[:2] != (len(csv_ids), 4):
            raise ValueError(
                f"Frame cache shape {self.frames.shape} does not match CSV"
            )
        if self.text.shape[0] != len(csv_ids) or (
            self.mask.shape != self.text.shape[:2]
        ):
            raise ValueError(
                f"Text cache {self.text.shape}/{self.mask.shape} mismatch"
            )
        split_by_id = {}
        with split_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                split_by_id[row["Id"]] = row["Split"]
        if set(split_by_id) != set(csv_ids):
            raise ValueError("Split IDs do not exactly match train.csv IDs")
        self.indices = [
            index
            for index, sample_id in enumerate(csv_ids)
            if split_by_id[sample_id] == split
        ]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source = self.indices[index]
        frames = torch.from_numpy(
            np.array(self.frames[source], dtype=np.float32, copy=True)
        )
        text = torch.from_numpy(
            np.array(self.text[source], dtype=np.float32, copy=True)
        )
        mask = torch.from_numpy(np.array(self.mask[source], copy=True)).bool()
        ranks = torch.tensor(self.labels[source], dtype=torch.long)
        return frames, text, mask, ranks


def _classes_from_ranks(ranks: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [ranks_to_class(row.tolist()) for row in ranks],
        dtype=torch.long,
        device=device,
    )


def train_text_vision_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for frames, text, mask, ranks in loader:
        frames, ranks = reorder_feature_batch(frames, ranks, generator)
        frames = frames.to(device)
        text = text.to(device)
        mask = mask.to(device)
        targets = _classes_from_ranks(ranks, device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(frames, text, mask), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach()) * frames.shape[0]
        total_count += frames.shape[0]
    return total_loss / total_count


def evaluate_text_vision(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.inference_mode():
        for frames, text, mask, ranks in loader:
            frames = frames.to(device)
            text = text.to(device)
            mask = mask.to(device)
            targets = _classes_from_ranks(ranks, device)
            scores = model(frames, text, mask)
            loss = F.cross_entropy(scores, targets)
            total_loss += float(loss) * frames.shape[0]
            total_correct += int((scores.argmax(dim=1) == targets).sum())
            total_count += frames.shape[0]
    return {
        "exact_match": total_correct / total_count,
        "loss": total_loss / total_count,
        "sample_count": total_count,
    }
