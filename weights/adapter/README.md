# LoRA adapter (our trained weights)

`adapter_model.safetensors` (305 MB) + `adapter_config.json` — the LoRA adapter we trained on top of
`Qwen/Qwen3.6-27B`. This is the only weight we trained; the base model is unmodified.

Tracked with **Git LFS** (see `../../.gitattributes`). After cloning:

```bash
git lfs install
git lfs pull
```

Attached at inference by `code/score_frames.py --adapter weights/adapter` (done automatically by
`run_inference.sh`).
