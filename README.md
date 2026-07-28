# Frame Ordering — Team Hobby

Given a caption and four shuffled video frames, predict their chronological order. This repository
contains the complete, runnable code to **reproduce our submission**, the trained LoRA adapter, and an
automatic download script for the base model.

- **Method:** Qwen3.6-VL-27B, 4-bit NF4, + a LoRA adapter, decoded by **24-candidate teacher-forced
  log-probability scoring** (the output is a permutation by construction), followed by a
  **near-duplicate retrieval** post-processing stage that uses training labels only.
- **Result:** public leaderboard **0.91972**.
- **Compute:** a single RTX 3090 (24 GB), fully offline; a full test run takes ≈ 6.7 h at ≈ 22.8 GB peak.

---

## 1. Repository layout

```
submission/
├── README.md                     ← this file
├── requirements.txt              ← exact library versions
├── ENVIRONMENT.md                ← OS / CUDA / GPU the result was produced on
├── run_inference.sh              ← one entry point: scoring → dhash → retrieval → submission.csv
├── code/
│   ├── score_frames.py           ← stage 1: 24-candidate log-prob scoring (the GPU stage)
│   ├── build_dhash.py            ← stage 2: perceptual hashes of the input frames
│   ├── nd2_postprocess.py        ← stage 3: near-duplicate retrieval → submission.csv
│   └── frame_ordering/           ← full methodology package (split, retrieval, scoring, baselines)
├── assets/                       ← train-derived retrieval index + labels + calibrated config
│   ├── train_index_dhash.npz
│   ├── train_labels.csv
│   └── nd2_calib_best.json
├── weights/
│   ├── adapter/                  ← OUR trained LoRA adapter (Git LFS, 305 MB)
│   └── BASE_MODEL.md             ← how to obtain the 52 GB base model
├── scripts/
│   └── download_base_model.py    ← auto-downloads Qwen/Qwen3.6-27B from Hugging Face
└── training/                     ← training driver + recipe (methodology reference)
```

## 2. Model weights

| Component | Size | Where it is | How to get it |
|---|---|---|---|
| **LoRA adapter** (ours) | 305 MB | `weights/adapter/` (in this repo, via **Git LFS**) | `git lfs pull` after clone |
| **Base model** (Qwen3.6-27B) | 52 GB | not in the repo | auto-download, see below |

**Base model — Qwen3.6-27B (Apache-2.0), open weights released within the competition window.**
Hugging Face: <https://huggingface.co/Qwen/Qwen3.6-27B>

```bash
python scripts/download_base_model.py            # -> weights/base_model/ (~52 GB)
# equivalently:  huggingface-cli download Qwen/Qwen3.6-27B --local-dir weights/base_model
```

If the evaluation machine already has a local copy, just point the expected path at it:

```bash
ln -sfn /path/to/Qwen3.6-27B weights/base_model
```

The base weights are **never modified**; everything we trained is the LoRA adapter.

## 3. Environment & installation

Details in [ENVIRONMENT.md](ENVIRONMENT.md). In short: Linux, one RTX 3090 (24 GB), CUDA 12.8, Python 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
# torch is the cu128 build — install it from the PyTorch index first:
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`transformers==5.12.1` is required — the backbone is a hybrid Gated-DeltaNet model (arch id `qwen3_5`)
that older releases cannot instantiate.

## 4. Reproducing the submission

```bash
# 0. base model in place (Section 2) and deps installed (Section 3)
git lfs pull                       # materialize weights/adapter/adapter_model.safetensors

# 1. quick end-to-end smoke test (8 samples, ~a few minutes on a GPU)
./run_inference.sh --test-csv /path/to/test.csv --image-root /path/to/test --limit 8 \
                   --out work/smoke.csv

# 2. the full run
CUDA_VISIBLE_DEVICES=0 ./run_inference.sh \
    --test-csv  /path/to/test.csv \
    --image-root /path/to/test \
    --out submission.csv
```

Inputs the organizers provide:
- `--test-csv`  : CSV with columns `Id, Input_1, Input_2, Input_3, Input_4, Sentence`
- `--image-root`: directory holding `<Id>/<frame>.jpg`

`run_inference.sh` runs three stages and is **resumable** — if `work/scores.json` already exists it is
reused, so a crash in a later stage costs seconds, not hours. Every emitted row is validated to be a
permutation of `[1,2,3,4]`. Honored env vars: `PYTHON=<python>` and `CUDA_VISIBLE_DEVICES=<gpu>`.

## 5. How it works (reproduction-relevant settings)

1. **Scoring** (`code/score_frames.py`) — the base model is loaded in 4-bit NF4 with the LoRA adapter
   attached; for each sample all 24 permutations are scored by teacher-forced log-probability and the
   argmax is taken. `--max-pixels 262144`, `--candidate-batch 8`.
2. **Perceptual hashing** (`code/build_dhash.py`) — 64-bit difference hash of each input frame.
3. **Near-duplicate retrieval** (`code/nd2_postprocess.py`) — matches input frames to the training
   frames in `assets/train_index_dhash.npz` (Hamming ≤ τ = 4), keeps only within-shot adjacency
   constraints from `assets/train_labels.csv`, and filters the 24 candidates to the consistent ones;
   parameters in `assets/nd2_calib_best.json` were calibrated on a training slice. Uses training
   labels only.

## 6. Training (methodology reference)

The adapter was produced by QLoRA supervised fine-tuning; see [training/README.md](training/README.md).
Recipe (`training/a27b_recipe.json`): base `Qwen/Qwen3.6-27B`, 4-bit NF4, LoRA `r=16 / α=32 /
dropout=0.05`, `lr=1e-4`, `epochs=3`, `grad_accum=8`, `seed=42`, presentation augmentation
(`aug_prob=0.5`, `reverse_prob=0.5`), vision tower frozen; target = the chronological index list.
The deployed adapter was trained on all 9,535 labeled samples.

## 7. Full methodology

See the accompanying technical report. The `code/frame_ordering/` package also contains the honest
grouped-split construction (`grouping.py`, `split.py`), the retrieval logic (`retrieval.py`), the
permutation-scoring interface (`qwen_ordering.py`, `permutations.py`), and the feature-baseline code
(`vision_baseline.py`, `text_vision.py`) used in our ablations.
