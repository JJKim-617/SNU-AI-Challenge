# Base model

The base model is **Qwen3.6-27B** (`Qwen/Qwen3.6-27B`), open weights, Apache-2.0, released within the
competition's open-model window. It is **not** stored in this repository (52 GB, 15 safetensors shards).

- Hugging Face: <https://huggingface.co/Qwen/Qwen3.6-27B>
- Architecture id `qwen3_5` — hybrid, 16 of 64 layers full attention, the rest Gated DeltaNet.
- Loaded in **4-bit NF4** (bitsandbytes) at inference, with our LoRA adapter on top.
- **The base weights are used unmodified.** Everything we trained is in `weights/adapter/` (305 MB).

## Get it

```bash
python scripts/download_base_model.py            # -> weights/base_model/  (recommended)
# or:
huggingface-cli download Qwen/Qwen3.6-27B --local-dir weights/base_model
```

If the machine already holds a copy, symlink it instead of downloading again:

```bash
ln -sfn /path/to/Qwen3.6-27B weights/base_model
```

`run_inference.sh` expects the base at `weights/base_model/` (checks `config.json`); pass a different
location with `--model-dir` to `code/score_frames.py` if you prefer.
