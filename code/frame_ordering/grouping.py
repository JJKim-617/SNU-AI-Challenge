import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

from frame_ordering.data import TrainSample
from frame_ordering.split import DatasetSplit

# Grayscale std below this on a 64x64 thumbnail marks a near-constant frame
# (black frames, fade-outs). Those appear verbatim in unrelated videos and
# must not link samples into one group.
FLAT_FRAME_STD_THRESHOLD = 4.0


def _is_flat_frame(path: Path) -> bool:
    with Image.open(path) as image:
        thumbnail = image.convert("L").resize((64, 64))
    return float(np.asarray(thumbnail, dtype=np.float32).std()) < (
        FLAT_FRAME_STD_THRESHOLD
    )


def image_digest_groups(
    samples: Sequence[TrainSample], image_root: Path
) -> Dict[str, int]:
    parent = {sample.sample_id: sample.sample_id for sample in samples}

    def find(sample_id: str) -> str:
        while parent[sample_id] != sample_id:
            parent[sample_id] = parent[parent[sample_id]]
            sample_id = parent[sample_id]
        return sample_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owner_by_digest: Dict[str, str] = {}
    flat_digests = set()
    for sample in samples:
        for image_name in sample.image_names:
            image_path = image_root / sample.sample_id / image_name
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest in flat_digests:
                continue
            if digest not in owner_by_digest:
                if _is_flat_frame(image_path):
                    flat_digests.add(digest)
                else:
                    owner_by_digest[digest] = sample.sample_id
                continue
            union(owner_by_digest[digest], sample.sample_id)

    group_index_by_root: Dict[str, int] = {}
    groups = {}
    for sample_id in sorted(parent):
        root = find(sample_id)
        if root not in group_index_by_root:
            group_index_by_root[root] = len(group_index_by_root)
        groups[sample_id] = group_index_by_root[root]
    return groups


def grouped_stratified_split(
    samples: Sequence[TrainSample],
    group_by_id: Dict[str, int],
    validation_fraction: float,
    seed: int,
) -> DatasetSplit:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample Id")
    missing = sorted(set(ids) - set(group_by_id))
    if missing:
        raise ValueError(f"Samples without a group assignment: {missing[:5]}")

    ranks_by_id = {sample.sample_id: sample.ranks for sample in samples}
    members_by_group: Dict[int, List[str]] = defaultdict(list)
    for sample_id in sorted(ids):
        members_by_group[group_by_id[sample_id]].append(sample_id)

    rank_counts = defaultdict(int)
    for sample in samples:
        rank_counts[sample.ranks] += 1
    validation_quota = {
        ranks: max(1, round(count * validation_fraction))
        for ranks, count in rank_counts.items()
    }
    assigned = defaultdict(int)
    rng = random.Random(seed)
    validation_ids: List[str] = []
    train_ids: List[str] = []

    multi_groups = sorted(
        (members for members in members_by_group.values() if len(members) > 1),
        key=lambda members: members[0],
    )
    rng.shuffle(multi_groups)
    for members in multi_groups:
        group_rank_counts = defaultdict(int)
        for sample_id in members:
            group_rank_counts[ranks_by_id[sample_id]] += 1
        fits_validation = all(
            assigned[ranks] + count <= validation_quota[ranks]
            for ranks, count in group_rank_counts.items()
        )
        if fits_validation:
            validation_ids.extend(members)
            for ranks, count in group_rank_counts.items():
                assigned[ranks] += count
        else:
            train_ids.extend(members)

    singles_by_rank: Dict[tuple, List[str]] = defaultdict(list)
    for members in members_by_group.values():
        if len(members) == 1:
            singles_by_rank[ranks_by_id[members[0]]].append(members[0])
    for ranks in sorted(singles_by_rank):
        group_ids = sorted(singles_by_rank[ranks])
        rng.shuffle(group_ids)
        remaining = max(0, validation_quota[ranks] - assigned[ranks])
        validation_ids.extend(group_ids[:remaining])
        train_ids.extend(group_ids[remaining:])
        assigned[ranks] += min(remaining, len(group_ids))

    return DatasetSplit(
        train_ids=tuple(sorted(train_ids)),
        validation_ids=tuple(sorted(validation_ids)),
    )
