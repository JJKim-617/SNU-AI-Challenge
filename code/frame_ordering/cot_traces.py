"""Deterministic CoT trace builder (no generative model).

Composes a short frame-anchored ``<think>`` block from the gold order + caption, ending
consistent with ``ranks_to_candidate``. Used only in a training ablation.
"""
from typing import Sequence

from frame_ordering.permutations import ranks_to_order, validate_permutation
from frame_ordering.qwen_ordering import split_clauses


def _clip(txt: str, n: int = 60) -> str:
    t = " ".join(txt.split())
    return (t[:n] + "…") if len(t) > n else t


def build_trace(caption: str, ranks: Sequence[int], no_ordering: bool,
                reversed_flag: bool, style: str = "caption") -> str:
    ranks = validate_permutation(list(ranks))
    order = list(ranks_to_order(ranks))            # chronological input-number order
    if no_ordering:
        return ("<think>\n"
                "Step 0 (order gate): the story implies no temporal change; "
                "the frames are already in order.\n</think>")
    clauses = split_clauses(caption)
    lines = ["Step 0 (order gate): the story describes a progression, so the frames need reordering."]
    clean4 = (len(clauses) >= 4 and not reversed_flag
              and all(len(c.split()) >= 2 for c in clauses[:4]))
    if clean4 and style == "caption":
        lines.append("Step 1 (match each story event to a frame, narration order = time order):")
        for j in range(4):
            lines.append(f'  - event {j + 1} "{_clip(clauses[j])}" -> Input {order[j]}.')
    elif len(clauses) >= 2:
        lines.append(f'Step 1: the earliest event ("{_clip(clauses[0])}") is Input {order[0]}; '
                     f'the latest ("{_clip(clauses[-1])}") is Input {order[3]}.')
        lines.append(f"Step 2: placing the remaining two frames by the story's progression "
                     f"gives Input {order[1]} then {order[2]}.")
    else:
        lines.append(f"Step 1: reading the story from start to end, the first moment is "
                     f"Input {order[0]} and the final moment is Input {order[3]}.")
        lines.append(f"Step 2: the two middle frames follow the story's progression as "
                     f"Input {order[1]} then {order[2]}.")
    lines.append(f"Step 3 (synthesize): chronological order is Input {', '.join(map(str, order))}.")
    return "<think>\n" + "\n".join(lines) + "\n</think>"
