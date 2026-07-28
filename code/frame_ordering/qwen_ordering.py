import ast
import itertools
import random
import re
from typing import List, Optional, Sequence, Tuple

from frame_ordering.permutations import (
    Permutation,
    order_to_ranks,
    validate_permutation,
)

# Order-preserving narrative connectives ("A before B" still narrates A
# first), shared by clause-based augmentation and analysis.
SPLIT_PATTERN = re.compile(
    r";\s*|,\s*then[,\s]|\bthen[,\s]|\bbefore\b|,\s*ending with|\bends with\b"
    r"|\bfinally[,\s]|\bafterwards[,\s]|,\s*and lastly|\blastly[,\s]",
    re.IGNORECASE,
)

PROMPT_TEMPLATE = (
    "The four images above are frames taken from one video, shown in "
    "shuffled order.\n"
    'Story: "{caption}"\n'
    "According to the story, list the image numbers in chronological "
    "order as a JSON list, for example [2, 4, 1, 3]. Answer with the "
    "list only."
)

# The 24 possible answers, serialized exactly as the model is trained to
# emit them: image numbers in chronological order.
ANSWER_CANDIDATES: Tuple[str, ...] = tuple(
    str(list(order)) for order in itertools.permutations((1, 2, 3, 4))
)


def split_clauses(sentence: str) -> List[str]:
    parts = [part.strip(" ,.") for part in SPLIT_PATTERN.split(sentence)]
    parts = [part for part in parts if len(part.split()) >= 2]
    return parts if parts else [sentence.strip()]


def reverse_caption(sentence: str) -> Optional[str]:
    """Rule-based clause reversal; returns None when the caption has no
    splittable narrative structure."""
    parts = split_clauses(sentence)
    if len(parts) < 2:
        return None
    reversed_parts = [parts[-1][0].upper() + parts[-1][1:]] + [
        part[0].lower() + part[1:] for part in reversed(parts[:-1])
    ]
    return ", then ".join(reversed_parts) + "."


def reverse_ranks(ranks: Sequence[int]) -> Permutation:
    checked = validate_permutation(ranks)
    return validate_permutation(5 - rank for rank in checked)


def permute_presentation(
    image_names: Sequence[str], ranks: Sequence[int], rng: random.Random
) -> Tuple[Tuple[str, ...], Permutation]:
    checked = validate_permutation(ranks)
    order = list(range(4))
    rng.shuffle(order)
    return (
        tuple(image_names[i] for i in order),
        validate_permutation(checked[i] for i in order),
    )


def candidate_to_ranks(candidate_text: str) -> Permutation:
    chronological_order = validate_permutation(
        ast.literal_eval(candidate_text)
    )
    return order_to_ranks(chronological_order)


def ranks_to_candidate(ranks: Sequence[int]) -> str:
    from frame_ordering.permutations import ranks_to_order

    return str(list(ranks_to_order(ranks)))


def build_messages(image_paths: Sequence[str], caption: str) -> list:
    if len(image_paths) != 4:
        raise ValueError(f"Expected 4 image paths, got {len(image_paths)}")
    content = []
    for index, path in enumerate(image_paths, start=1):
        content.append({"type": "text", "text": f"Image {index}:"})
        content.append({"type": "image", "image": str(path)})
    content.append(
        {"type": "text", "text": PROMPT_TEMPLATE.format(caption=caption)}
    )
    return [{"role": "user", "content": content}]
