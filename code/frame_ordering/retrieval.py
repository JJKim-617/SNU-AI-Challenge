import ast
from typing import Dict, List, Optional, Sequence, Tuple

from frame_ordering.permutations import order_to_ranks
from frame_ordering.qwen_ordering import ANSWER_CANDIDATES

Constraint = Tuple[int, int]

# Precomputed time position of each test frame (0-based) for every
# candidate: candidate "[2, 4, 1, 3]" puts image 2 first, so frame 1
# (0-based) has position 0.
_CANDIDATE_POSITIONS: List[Dict[int, int]] = []
for _text in ANSWER_CANDIDATES:
    _order = ast.literal_eval(_text)
    _CANDIDATE_POSITIONS.append(
        {image - 1: position for position, image in enumerate(_order)}
    )

# Submission-space ranks aligned to ANSWER_CANDIDATES ordering.
_ALL_RANKS: List[Tuple[int, int, int, int]] = [
    order_to_ranks(ast.literal_eval(_text)) for _text in ANSWER_CANDIDATES
]


def aggregate_pairwise_constraints(
    test_digests: Sequence[str],
    train_orders: Dict[str, Dict[str, int]],
) -> Dict[Constraint, int]:
    """Vote counts for 'test frame i precedes test frame j in the source
    video timeline', from train samples sharing identical frames.

    train_orders maps a train sample id to {digest: chronological rank}
    restricted to digests it shares with the test sample.
    """
    constraints: Dict[Constraint, int] = {}
    for shared in train_orders.values():
        matched = [
            (position, shared[digest])
            for position, digest in enumerate(test_digests)
            if digest in shared
        ]
        for index_a in range(len(matched)):
            for index_b in range(len(matched)):
                (frame_a, rank_a) = matched[index_a]
                (frame_b, rank_b) = matched[index_b]
                if rank_a < rank_b:
                    key = (frame_a, frame_b)
                    constraints[key] = constraints.get(key, 0) + 1
    return constraints


def full_match_orders(
    test_digests: Sequence[str],
    train_orders: Dict[str, Sequence[str]],
) -> Optional[List[Tuple[int, int, int, int]]]:
    """When some train sample contains all four test frames, return the
    two allowed rank tuples (forward source order and its reverse).

    Returns None if no train sample is a full match, or if full matches
    disagree on the source order (ambiguous → let the soft path handle
    it).
    """
    digest_to_frame = {digest: index for index, digest in enumerate(test_digests)}
    if len(digest_to_frame) != 4:
        return None
    source_orders = set()
    for chronological in train_orders.values():
        if len(chronological) < 4:
            continue
        frames = [
            digest_to_frame[digest]
            for digest in chronological
            if digest in digest_to_frame
        ]
        if len(frames) == 4 and len(set(frames)) == 4:
            source_orders.add(tuple(frames))
    if len(source_orders) != 1:
        return None
    frame_order = next(iter(source_orders))
    forward_images = tuple(frame + 1 for frame in frame_order)
    reverse_images = tuple(reversed(forward_images))
    return [order_to_ranks(forward_images), order_to_ranks(reverse_images)]


def restrict_to_candidates(
    scores: Sequence[float],
    allowed_ranks: Sequence[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """Pick the highest-scoring order among an allowed shortlist."""
    if not allowed_ranks:
        return None
    rank_to_index = {rank: index for index, rank in enumerate(_ALL_RANKS)}
    best = max(allowed_ranks, key=lambda rank: scores[rank_to_index[rank]])
    return best


def rescore_candidates(
    scores: Sequence[float],
    constraints: Dict[Constraint, int],
    weight: float,
) -> List[float]:
    """Add a retrieval bonus to the 24 candidate scores.

    The source timeline may be narrated in either direction by the test
    caption, so the bonus for a candidate is the better of its forward
    and reverse constraint agreement — the direction is resolved per
    candidate as a whole, never per pair.
    """
    if len(scores) != len(ANSWER_CANDIDATES):
        raise ValueError(f"Expected 24 scores, got {len(scores)}")
    adjusted = []
    for candidate_index, base in enumerate(scores):
        positions = _CANDIDATE_POSITIONS[candidate_index]
        forward = 0
        reverse = 0
        for (frame_a, frame_b), votes in constraints.items():
            if positions[frame_a] < positions[frame_b]:
                forward += votes
            else:
                reverse += votes
        adjusted.append(base + weight * max(forward, reverse))
    return adjusted
