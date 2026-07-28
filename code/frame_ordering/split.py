import random
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List, Sequence, Tuple

from frame_ordering.data import TrainSample
from frame_ordering.permutations import Permutation


@dataclass(frozen=True)
class DatasetSplit:
    train_ids: Tuple[str, ...]
    validation_ids: Tuple[str, ...]


def stratified_split(
    samples: Sequence[TrainSample], validation_fraction: float, seed: int
) -> DatasetSplit:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample Id")
    groups: DefaultDict[Permutation, List[str]] = defaultdict(list)
    for sample in samples:
        groups[sample.ranks].append(sample.sample_id)
    rng = random.Random(seed)
    train_ids = []
    validation_ids = []
    for ranks in sorted(groups):
        group_ids = sorted(groups[ranks])
        rng.shuffle(group_ids)
        validation_count = max(1, round(len(group_ids) * validation_fraction))
        validation_ids.extend(group_ids[:validation_count])
        train_ids.extend(group_ids[validation_count:])
    return DatasetSplit(
        train_ids=tuple(sorted(train_ids)),
        validation_ids=tuple(sorted(validation_ids)),
    )
