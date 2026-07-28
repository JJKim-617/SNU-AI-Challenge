import itertools
from collections import Counter
from typing import Dict, Sequence

from frame_ordering.permutations import Permutation, reorder_ranks


def exact_match(
    targets: Sequence[Permutation], predictions: Sequence[Permutation]
) -> float:
    if not targets or len(targets) != len(predictions):
        raise ValueError(
            "Targets and predictions must have the same non-zero length"
        )
    return sum(
        target == prediction
        for target, prediction in zip(targets, predictions)
    ) / len(targets)


def majority_ranks(targets: Sequence[Permutation]) -> Permutation:
    if not targets:
        raise ValueError("Cannot compute majority of empty targets")
    counts = Counter(targets)
    best_count = max(counts.values())
    return min(
        ranks for ranks, count in counts.items() if count == best_count
    )


def exhaustive_reordering_counts(
    targets: Sequence[Permutation],
) -> Dict[Permutation, int]:
    if not targets:
        raise ValueError("Cannot audit empty targets")
    counts = Counter()
    for target in targets:
        for new_input_order in itertools.permutations((1, 2, 3, 4)):
            counts[reorder_ranks(target, new_input_order)] += 1
    return dict(counts)
