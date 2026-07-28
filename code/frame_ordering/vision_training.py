import ast
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from frame_ordering.permutations import validate_permutation
from frame_ordering.vision_baseline import ranks_to_class, reorder_feature_batch


class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        train_csv: Path,
        split_csv: Path,
        split: str,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(f"Unknown split: {split}")
        self.features = np.load(feature_path, mmap_mode="r")
        cache_ids = json.loads(
            feature_path.with_suffix(".ids.json").read_text(encoding="utf-8")
        )
        labels = []
        csv_ids = []
        with train_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                csv_ids.append(row["Id"])
                labels.append(
                    validate_permutation(ast.literal_eval(row["Answer"]))
                )
        if cache_ids != csv_ids:
            raise ValueError("Cache IDs do not exactly match train.csv IDs")
        if self.features.shape[:2] != (len(csv_ids), 4):
            raise ValueError(
                f"Feature shape {self.features.shape} does not match CSV"
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
        source_index = self.indices[index]
        features = torch.from_numpy(
            np.array(self.features[source_index], dtype=np.float32, copy=True)
        )
        ranks = torch.tensor(self.labels[source_index], dtype=torch.long)
        return features, ranks


def _classes_from_ranks(ranks: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [ranks_to_class(row.tolist()) for row in ranks],
        dtype=torch.long,
        device=device,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for features, ranks in loader:
        features, ranks = reorder_feature_batch(features, ranks, generator)
        features = features.to(device)
        targets = _classes_from_ranks(ranks, device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(features), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach()) * features.shape[0]
        total_count += features.shape[0]
    return total_loss / total_count


def evaluate(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.inference_mode():
        for features, ranks in loader:
            features = features.to(device)
            targets = _classes_from_ranks(ranks, device)
            scores = model(features)
            loss = F.cross_entropy(scores, targets)
            total_loss += float(loss) * features.shape[0]
            total_correct += int((scores.argmax(dim=1) == targets).sum())
            total_count += features.shape[0]
    return {
        "exact_match": total_correct / total_count,
        "loss": total_loss / total_count,
        "sample_count": total_count,
    }
