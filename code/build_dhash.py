#!/usr/bin/env python3
"""Compute the 64-bit difference-hash (dhash) of every frame in a split.

Used for the TEST split being scored -> the query hashes nd2_postprocess.py needs. (The
train-side index is already shipped as assets/train_index_dhash.npz.)

Hash definition is fixed: greyscale, resize to 9x8, one bit per horizontal neighbour pair,
row-major, MSB first -> 64 bits. It must match the function the tau in
assets/nd2_calib_best.json was calibrated against.

CSV columns: Id, Input_1..Input_4; frame at <image-root>/<Id>/<Input_k>. A frame that
fails to open becomes None (treated as "no constraint").

Usage:
  python build_dhash.py --csv data/test.csv --image-root data/test --out test_dhash.npz
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

HX, HY = 9, 8  # frozen


def dhash(path):
    im = Image.open(path).convert("L").resize((HX, HY))
    px = list(im.getdata())
    bits = 0
    for r in range(HY):
        for c in range(HX - 1):
            bits = (bits << 1) | (1 if px[r * HX + c] < px[r * HX + c + 1] else 0)
    return bits


def one(job):
    vid, slot, path = job
    try:
        return vid, slot, dhash(path)
    except Exception:
        return vid, slot, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="split csv with Id,Input_1..4")
    ap.add_argument("--image-root", type=Path, required=True, help="dir holding <Id>/<frame>")
    ap.add_argument("--out", type=Path, required=True, help="output .npz")
    ap.add_argument("--key", default="te_h", help="array name inside the npz (te_h for test)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    jobs, ids = [], []
    with args.csv.open(encoding="utf-8-sig") as h:
        for r in csv.DictReader(h):
            vid = r["Id"].strip()
            ids.append(vid)
            for k in range(1, 5):
                jobs.append((vid, k - 1, args.image_root / vid / r[f"Input_{k}"].strip()))
    print(f"{len(ids)} samples / {len(jobs)} frames", flush=True)

    with Pool(args.workers) as p:
        res = p.map(one, jobs, chunksize=256)

    out = {vid: [None] * 4 for vid in ids}
    for vid, slot, hv in res:
        out[vid][slot] = hv
    missing = sum(1 for v in out.values() for x in v if x is None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **{args.key: np.array(out, dtype=object)})
    print(f"wrote {args.out} | {len(out)} samples | unreadable frames: {missing}")


if __name__ == "__main__":
    main()
