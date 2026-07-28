# Execution environment

The 0.91972 result was produced and verified on the following setup.

| | |
|---|---|
| OS | Linux (x86-64), kernel 5.15 |
| GPU | 1× NVIDIA RTX 3090, 24 GB — inference uses a single GPU |
| CUDA | 12.8 (PyTorch cu128 build) |
| Python | 3.11 |
| Peak VRAM (inference) | ≈ 22.8 GB / 24 GB |
| Full test run | ≈ 6.7 h (offline) |

Inference and training both fit on one 24 GB GPU because the base model is loaded in 4-bit NF4 and only
a LoRA adapter is trained/attached.

## Libraries

Exact pins are in [`requirements.txt`](requirements.txt). Key packages:

```
torch==2.11.0            # install from the cu128 index (see below), not PyPI
torchvision==0.26.0
transformers==5.12.1     # REQUIRED — older releases cannot instantiate the qwen3_5 backbone
tokenizers==0.22.2
peft==0.19.1
bitsandbytes==0.49.2     # 4-bit NF4 quantization
accelerate==1.14.0
qwen-vl-utils==0.0.14
safetensors==0.8.0
huggingface_hub==1.22.0
numpy==2.3.5
pillow==11.3.0
```

Install order (torch first, from the CUDA 12.8 index):

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The pipeline runs fully offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` are set inside
`run_inference.sh`); only `scripts/download_base_model.py` needs network access.
