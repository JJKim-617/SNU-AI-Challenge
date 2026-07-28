#!/usr/bin/env python3
"""Rule-compliant training augmentation.

Structural augmentation (changes text/order, so the label changes with it):
  - caption reversal (reverse_caption + reverse_ranks, as a pair)
  - presentation re-permutation (re-labelled accordingly)
Photometric augmentation (brightness/contrast/crop, label-invariant) is applied
on-the-fly at train load time via apply_aug/draw_aug. Everything is a re-labelling or a
photometric/crop transform of the provided data; no generative-AI data is created.

No_ordering policy:
  policy="gate" (default): No_ordering rows are left as-is -> answer stays identity
                           (teaches the natural 15.5% "already sorted" gate).
  policy="ej"            : shuffle + reverse everything -> balanced answer distribution.
"""
import ast, sys, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from frame_ordering.permutations import validate_permutation, order_to_ranks, ranks_to_order  # noqa: E402
from frame_ordering.qwen_ordering import (  # noqa: E402
    reverse_caption, reverse_ranks, permute_presentation, split_clauses,
)

IDENTITY = (1, 2, 3, 4)

# Near-identity ranks: Kendall distance 1-2 from identity (identity itself excluded)
NEAR_ID_RANKS = (
    (2, 1, 3, 4), (1, 3, 2, 4), (1, 2, 4, 3),                    # distance 1
    (2, 3, 1, 4), (3, 1, 2, 4), (2, 1, 4, 3), (1, 3, 4, 2), (1, 4, 2, 3),  # distance 2
)


# ---- photometric (per-sample consistent, no flip/rotate) ----
def draw_aug(rng):
    return {"brightness": rng.uniform(0.85, 1.15), "contrast": rng.uniform(0.85, 1.15),
            "scale": rng.uniform(0.85, 1.0), "fx": rng.random(), "fy": rng.random()}


def apply_aug(img, p):
    from PIL import Image, ImageEnhance
    w, h = img.size
    cw, ch = max(1, round(w * p["scale"])), max(1, round(h * p["scale"]))
    x0, y0 = round((w - cw) * p["fx"]), round((h - ch) * p["fy"])
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR)
    img = ImageEnhance.Brightness(img).enhance(p["brightness"])
    img = ImageEnhance.Contrast(img).enhance(p["contrast"])
    return img


# ---- structural augmentation (materialized) ----
def augment_structural(row, rng, reverse_prob=0.5, policy="gate", near_q=0.0):
    """Row -> augmented (presentation frame names, caption, submission ranks), label-consistent.
    near_q>0: with probability near_q, draw the presentation from NEAR_ID_RANKS (an
    "almost sorted but not" input) to sharpen the identity boundary; labels stay consistent."""
    ranks = validate_permutation(ast.literal_eval(row["Answer"]))
    caption = row["Sentence"]
    no = str(row.get("No_ordering", "")).strip().lower() == "true"
    names = tuple(row[f"Input_{i}"].strip() for i in range(1, 5))
    reversed_, shuffled = False, False

    if policy == "gate" and no:
        # already-sorted identity sample: keep as-is (gate signal), answer = identity
        return dict(order_names=names, caption=caption, ranks=tuple(ranks),
                    no_ordering=True, reversed=False, shuffled=False, near=False)

    # (1) reverse the caption (when possible) and the ranks together
    if rng.random() < reverse_prob:
        rc = reverse_caption(caption)
        if rc is not None:
            caption, ranks, reversed_ = rc, reverse_ranks(ranks), True
    # (2) re-permute the presentation and re-derive ranks
    near = bool(near_q) and rng.random() < near_q
    if near:
        rho = NEAR_ID_RANKS[rng.randrange(len(NEAR_ID_RANKS))]
        chrono_idx = sorted(range(4), key=lambda i: ranks[i])   # chronological frame indices
        names = tuple(names[chrono_idx[rho[i] - 1]] for i in range(4))
        ranks = validate_permutation(rho)
    else:
        names, ranks = permute_presentation(names, ranks, rng)
    shuffled = True
    return dict(order_names=tuple(names), caption=caption, ranks=tuple(ranks),
                no_ordering=no, reversed=reversed_, shuffled=shuffled, near=near)


def _selfcheck():
    """order<->ranks round-trip + augmentation consistency."""
    rng = random.Random(0)
    row = {"Id": "x", "Input_1": "a", "Input_2": "b", "Input_3": "c", "Input_4": "d",
           "Sentence": "The man opens the door, then he walks in, then he sits down, and finally he reads.",
           "Answer": "[3, 1, 4, 2]", "No_ordering": "False"}
    r = validate_permutation([3, 1, 4, 2])
    assert tuple(order_to_ranks(ranks_to_order(r))) == tuple(r), "order<->ranks round-trip failed"
    for _ in range(200):
        a = augment_structural(row, rng)
        assert sorted(a["ranks"]) == [1, 2, 3, 4], "augmented ranks are not a valid permutation"
        assert len(a["order_names"]) == 4 and len(set(a["order_names"])) == 4
    # gate policy: No_ordering preserved
    g = augment_structural({**row, "Answer": "[1, 2, 3, 4]", "No_ordering": "True"}, rng, policy="gate")
    assert tuple(g["ranks"]) == IDENTITY and not g["shuffled"], "gate policy did not preserve No_ordering"
    # near branch: label consistency (name->orig rank preserved), no identity, distance<=2, freq~q
    def _inv(p):
        return sum(1 for a in range(4) for b in range(a + 1, 4) if p[a] > p[b])
    orig = validate_permutation(ast.literal_eval(row["Answer"]))
    name2rank = {row[f"Input_{i}"].strip(): orig[i - 1] for i in range(1, 5)}
    nn = 0
    for _ in range(3000):
        a = augment_structural(row, rng, reverse_prob=0.0, policy="gate", near_q=0.3)
        assert sorted(a["ranks"]) == [1, 2, 3, 4]
        assert tuple(a["ranks"]) == tuple(name2rank[n] for n in a["order_names"]), "near re-label mismatch"
        if a["near"]:
            nn += 1
            assert tuple(a["ranks"]) != IDENTITY and 1 <= _inv(a["ranks"]) <= 2, "near distance violated"
    assert 0.25 < nn / 3000 < 0.35, f"near frequency {nn/3000:.3f} != ~0.3"
    # gate rows are unaffected by near_q
    g2 = augment_structural({**row, "Answer": "[1, 2, 3, 4]", "No_ordering": "True"}, rng,
                            policy="gate", near_q=0.9)
    assert tuple(g2["ranks"]) == IDENTITY and not g2["near"], "gate row contaminated by near"
    print(f"augment.py selfcheck OK (round-trip, valid perms, gate policy, near q={nn/3000:.3f})")


if __name__ == "__main__":
    _selfcheck()
