#!/usr/bin/env python3
"""Near-duplicate post-processing: turn the 24 candidate log-probabilities into the
submitted permutation. Only train labels are read; no test label, external data, or
second model.

Rule: fuzzy-match each of a video's 4 frames against the labelled train frames by dhash
Hamming distance <= tau. A train video matched on >= 2 frames votes on the relative order
of those frames from its own labels; only train-rank-adjacent votes become constraints. A
candidate survives if it satisfies the constraints in the forward OR the reverse direction;
among survivors take the model's argmax, else fall back to plain argmax. Videos whose 6
pairs are all constrained are functional duplicates, so the model is trusted. tau and
min_votes come from assets/nd2_calib_best.json (tuned on a held-out train slice).

Usage:
  python nd2_postprocess.py --scores scores.json --test-dhash test_dhash.npz \
      --train-index assets/train_index_dhash.npz --train-labels assets/train_labels.csv \
      --config assets/nd2_calib_best.json --out submission.csv
"""
import argparse
import ast
import csv
import json
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np


def order_to_ranks(o):
    """[2,3,4,1] (chronological list of input indices) -> (4,1,2,3) (rank of each input)."""
    r = [0] * 4
    for pos, img in enumerate(o, start=1):
        r[img - 1] = pos
    return tuple(r)


CANDS = [list(p) for p in permutations((1, 2, 3, 4))]
RANKS = [order_to_ranks(o) for o in CANDS]
CPOS = [{img - 1: p for p, img in enumerate(o)} for o in CANDS]
POP = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


def load_hashes(path, key=None):
    z = np.load(path, allow_pickle=True)
    if key:
        return dict(z[key].item())
    merged = {}
    for k in z.files:                      # train index ships as tr_h + va_h
        merged.update(z[k].item())
    return merged


def build_index(index_h):
    ids, slots, hashes = [], [], []
    for sid, hs in index_h.items():
        for k, hv in enumerate(hs):
            if hv is None:
                continue
            ids.append(sid)
            slots.append(k)
            hashes.append(hv)
    T = np.array([[(h >> (8 * b)) & 0xFF for b in range(8)] for h in hashes], dtype=np.uint8)
    return np.array(ids), slots, T


def constraints(hs, ids, slots, T, labels, tau):
    """-> (adjacent-rank votes, far-rank votes), each {(slot_a, slot_b): n_votes}."""
    per_train = defaultdict(dict)
    for slot in range(4):
        hv = hs[slot]
        if hv is None:
            continue
        V = np.array([(hv >> (8 * b)) & 0xFF for b in range(8)], dtype=np.uint8)
        d = POP[np.bitwise_xor(T, V[None, :])].sum(axis=1)
        for ix in np.where(d <= tau)[0]:
            sid = ids[ix]
            cur = per_train[sid].get(slot)
            if cur is None or d[ix] < cur[0]:          # keep the closest match per slot
                per_train[sid][slot] = (int(d[ix]), slots[ix])
    adj, far = defaultdict(int), defaultdict(int)
    for sid, m in per_train.items():
        if len(m) < 2:
            continue
        it = [(sl, int(labels[sid][ts])) for sl, (_, ts) in m.items()]
        for a in range(len(it)):
            for b in range(len(it)):
                (sa, ra), (sb, rb) = it[a], it[b]
                if ra < rb:
                    (adj if rb - ra == 1 else far)[(sa, sb)] += 1
    return dict(adj), dict(far)


def pick(scores, adj, far, min_votes):
    argmax = max(range(24), key=lambda k: scores[k])
    if len(set(frozenset(p) for p in list(adj) + list(far))) == 6:
        return argmax, "skip_twin"                     # functional duplicate
    strong = {k: v for k, v in adj.items() if v >= min_votes}
    if not strong:
        return argmax, "no_constraint"
    fwd = [c for c in range(24) if all(CPOS[c][a] < CPOS[c][b] for a, b in strong)]
    rev = [c for c in range(24) if all(CPOS[c][a] > CPOS[c][b] for a, b in strong)]
    surv = fwd + rev
    if not surv:
        return argmax, "no_survivor"
    return max(surv, key=lambda k: scores[k]), "filtered"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True, help="score_frames.py output json")
    ap.add_argument("--test-dhash", type=Path, required=True)
    ap.add_argument("--train-index", type=Path, required=True)
    ap.add_argument("--train-labels", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--also-pure", type=Path, help="optionally write the plain-argmax submission too")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    tau, min_votes = cfg["tau"], cfg["min_votes"]

    labels = {}
    with args.train_labels.open(encoding="utf-8-sig") as h:
        for r in csv.DictReader(h):
            labels[r["Id"].strip()] = ast.literal_eval(r["Answer"])

    index_h = load_hashes(args.train_index)
    te = load_hashes(args.test_dhash, key="te_h")
    ids, slots, T = build_index(index_h)
    print(f"index: {len(index_h)} labelled train videos / {T.shape[0]} frames | "
          f"queries: {len(te)} test videos | tau={tau} min_votes={min_votes}")

    payload = json.load(open(args.scores))
    per_sample = payload.get("per_sample", payload)
    scores = {r["id"]: r["scores"] for r in per_sample}

    missing = [t for t in scores if t not in te]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} scored ids have no dhash "
                         f"(e.g. {missing[:3]}). Rerun build_dhash.py on the test split.")

    rows_nd2, rows_pure, reasons, changed = [], [], defaultdict(int), 0
    for tid, s in scores.items():
        if len(s) != 24:
            raise SystemExit(f"FATAL: {tid} has {len(s)} scores, expected 24")
        adj, far = constraints(te[tid], ids, slots, T, labels, tau)
        argmax = max(range(24), key=lambda k: s[k])
        chosen, why = pick(s, adj, far, min_votes)
        reasons[why] += 1
        changed += int(chosen != argmax)
        rows_nd2.append((tid, RANKS[chosen]))
        rows_pure.append((tid, RANKS[argmax]))

    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Id", "Answer"])
            for tid, rk in rows:
                w.writerow([tid, "[" + ", ".join(map(str, rk)) + "]"])

    write(args.out, rows_nd2)
    if args.also_pure:
        write(args.also_pure, rows_pure)

    print(f"wrote {args.out} | {len(rows_nd2)} rows | nd2 changed {changed} vs plain argmax")
    print(f"  reasons: {dict(reasons)}")
    print("  (reference: on our 819-sample test set nd2 changed 22 rows; "
          "coverage 453/819 with >=1 constraint)")


if __name__ == "__main__":
    main()
