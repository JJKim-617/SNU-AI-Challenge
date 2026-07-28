from typing import Dict, Optional, Sequence, Tuple

from frame_ordering.qwen_ordering import ANSWER_CANDIDATES, candidate_to_ranks
from frame_ordering.retrieval import Constraint, rescore_candidates


RANKS_BY_CANDIDATE: Tuple[Tuple[int, int, int, int], ...] = tuple(
    candidate_to_ranks(candidate) for candidate in ANSWER_CANDIDATES
)
IDENTITY_INDEX = RANKS_BY_CANDIDATE.index((1, 2, 3, 4))


def apply_identity_bonus(scores: Sequence[float], bonus: float) -> list:
    if len(scores) != len(ANSWER_CANDIDATES):
        raise ValueError(f"Expected 24 scores, got {len(scores)}")
    adjusted = [float(score) for score in scores]
    adjusted[IDENTITY_INDEX] += float(bonus)
    return adjusted


def postprocess_scores(
    scores: Sequence[float],
    constraints: Optional[Dict[Constraint, int]] = None,
    retrieval_weight: float = 0.0,
    identity_bonus: float = 0.0,
) -> list:
    adjusted = [float(score) for score in scores]
    if constraints:
        adjusted = rescore_candidates(adjusted, constraints, retrieval_weight)
    return apply_identity_bonus(adjusted, identity_bonus)


def pick_ranks(
    scores: Sequence[float],
    constraints: Optional[Dict[Constraint, int]] = None,
    retrieval_weight: float = 0.0,
    identity_bonus: float = 0.0,
) -> Tuple[int, int, int, int]:
    adjusted = postprocess_scores(scores, constraints, retrieval_weight, identity_bonus)
    pick = max(range(len(adjusted)), key=lambda index: adjusted[index])
    return RANKS_BY_CANDIDATE[pick]
