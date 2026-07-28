# Training (methodology reference)

This directory documents how the deployed LoRA adapter (`../weights/adapter/`) was produced. It is
provided for methodology review; **reproducing the submission does not require retraining** — the
trained adapter ships in the repo.

## Recipe

`a27b_recipe.json` holds the supervised fine-tuning configuration:

| setting | value |
|---|---|
| base model | `Qwen/Qwen3.6-27B`, 4-bit NF4 (frozen) |
| adapter | LoRA on attention + MLP projections, vision tower frozen |
| `lora_r` / `lora_alpha` / `lora_dropout` | 16 / 32 / 0.05 |
| `lr` | 1e-4 |
| `epochs` | 3 (convergence point; measured on a calibration slice) |
| `gradient_accumulation` | 8 |
| `seed` | 42 |
| presentation augmentation | `aug_prob=0.5`, `reverse_prob=0.5` |
| target | chronological index list as a short JSON array (cross-entropy on ordering tokens only) |

The **deployed** adapter was trained on all 9,535 labeled samples. `a27b_recipe.json` shows the clean
sibling that excludes a 500-sample calibration slice (used only to report honest held-out numbers);
the deployed run uses the same hyperparameters over the full training set.

## Driver

`train_a0.py` is the QLoRA SFT driver. It targets the chronological ordering directly (the same 24-way
answer space used at inference) and saves a checkpoint per epoch. `augment.py` holds the rule-compliant
augmentation it uses. Imports are repo-relative (`../code` for `frame_ordering`, this directory for
`augment`), so no path editing is needed.

```bash
# place the competition training data at  ../data/train.csv  and  ../data/train/<Id>/<frame>.jpg
# (or set DATA_ROOT to a directory that contains train.csv and train/), then:
CUDA_VISIBLE_DEVICES=0 python train_a0.py --config a27b_recipe.json
# quick plumbing check:
CUDA_VISIBLE_DEVICES=0 python train_a0.py --config a27b_recipe.json --smoke --max-steps 6
```

`a27b_recipe.json` uses relative paths (`weights/base_model`, `data/train.csv`, `outputs/adapter`);
adjust them if your layout differs. The training data is the competition-provided `train.csv` and
`train/` image directory (not redistributed here).
