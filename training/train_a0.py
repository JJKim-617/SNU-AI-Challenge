#!/usr/bin/env python3
"""QLoRA supervised fine-tuning of the base model for frame ordering.

Target = the chronological input-index list (same 24-candidate space as inference).
Saves one adapter checkpoint per epoch.

Run: python train_a0.py --config a27b_recipe.json [--smoke --max-steps 6]
"""
import argparse, ast, csv, json, math, os, random, sys, time
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "code"))   # frame_ordering package
sys.path.insert(0, str(_HERE))                   # augment (this dir)
from frame_ordering.permutations import validate_permutation           # noqa
from frame_ordering.qwen_ordering import ranks_to_candidate            # noqa
from frame_ordering.cot_traces import build_trace                      # noqa  (CoT)
import augment as A                                                     # op4/data/augment

PROMPT = ('These four images are frames from one video, shown in shuffled order (Input 1-4).\n'
          'Story: "{caption}"\n'
          'List the input image numbers in chronological order as a JSON list, e.g. [2, 4, 1, 3].')
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(_HERE.parent / "data")))
IMG_ROOT = DATA_ROOT / "train"
COT_EVAL = [False]          # set in main() from cfg["cot"]; toggles CoT target in eval_val


def build_batch(proc, tok, frames, caption, answer, upscale, aug_params, device):
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in frames]
    if upscale and upscale != 1:
        imgs = [im.resize((im.width * upscale, im.height * upscale)) for im in imgs]
    if aug_params:
        imgs = [A.apply_aug(im, aug_params) for im in imgs]
    content = [{"type": "image", "image": str(f)} for f in frames]
    content.append({"type": "text", "text": PROMPT.format(caption=caption)})
    text = proc.apply_chat_template([{"role": "user", "content": content}],
                                    tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=imgs, return_tensors="pt")
    ans_ids = tok(answer + tok.eos_token, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([inp["input_ids"], ans_ids], dim=1)
    labels = torch.cat([torch.full_like(inp["input_ids"], -100), ans_ids], dim=1)
    batch = {"input_ids": input_ids.to(device), "attention_mask": torch.ones_like(input_ids).to(device),
             "labels": labels.to(device)}
    for k in ("pixel_values", "image_grid_thw"):
        if k in inp:
            batch[k] = inp[k].to(device)
    if "mm_token_type_ids" in inp:                       # candidate/answer tokens are text (type 0)
        mm = inp["mm_token_type_ids"]
        cand = torch.zeros((mm.shape[0], ans_ids.shape[1]), dtype=mm.dtype)
        batch["mm_token_type_ids"] = torch.cat([mm, cand], dim=1).to(device)
    return batch


@torch.no_grad()
def eval_val(model, proc, tok, val_rows, upscale, device, n):
    model.eval(); tot, cnt = 0.0, 0
    for row in val_rows[:n]:
        frames = [IMG_ROOT / row["Id"] / row[f"Input_{i}"].strip() for i in range(1, 5)]
        ranks = validate_permutation(ast.literal_eval(row["Answer"]))
        tgt = ranks_to_candidate(ranks)
        if COT_EVAL[0]:
            no_ord = row.get("No_ordering", "").strip().lower() == "true"
            tgt = build_trace(row["Sentence"], ranks, no_ord, False) + "\n" + tgt
        b = build_batch(proc, tok, frames, row["Sentence"], tgt, upscale, None, device)
        tot += float(model(**b).loss); cnt += 1
    model.train()
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    COT_EVAL[0] = bool(cfg.get("cot"))
    # DDP: full model per rank; only LoRA grads are all-reduced
    if "LOCAL_RANK" in os.environ:
        # gloo + CPU tensors (NCCL collectives can hang on some hosts)
        dist.init_process_group(cfg.get("ddp_backend", "nccl"))
        local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        local_rank, rank, world = 0, 0, 1
    is_main = rank == 0
    rng = random.Random(cfg["seed"] + rank * 9973)   # per-rank augmentation seed
    torch.manual_seed(cfg["seed"] + rank)
    up = int(cfg.get("upscale", 2)); policy = cfg.get("policy", "gate")
    out = Path(cfg["output_dir"])
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    # data
    rows = {r["Id"].strip(): r for r in csv.DictReader(open(DATA_ROOT / "train.csv", encoding="utf-8-sig"))}
    split = {r["Id"]: r["Split"] for r in csv.DictReader(open(cfg["split_csv"]))}
    train_rows = [rows[i] for i in rows if split.get(i) == "train"]
    if cfg.get("skip_no_ordering"):        # CoT ordering-only ablation: drop identity cases from training
        train_rows = [r for r in train_rows if r.get("No_ordering", "").strip().lower() != "true"]
    val_rows = [rows[i] for i in rows if split.get(i) == "validation"]
    if args.smoke:
        train_rows = train_rows[:12]; val_rows = val_rows[:6]
    if is_main:
        print(f"train {len(train_rows)} / val {len(val_rows)} | upscale {up}x policy {policy} | "
              f"DDP world={world}", flush=True)

    # model
    proc = AutoProcessor.from_pretrained(cfg["model_dir"], local_files_only=True)
    tok = proc.tokenizer
    load_kw = dict(device_map={"": local_rank}, local_files_only=True, dtype=torch.bfloat16)
    if not cfg.get("prequantized"):        # quantize a bf16 model to 4-bit at load; skip if already 4-bit
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(cfg["model_dir"], **load_kw)
    if cfg.get("light_kbit_prep"):
        # keep bf16 and enable grad-ckpt / input-grads manually; the kbit-prep fp32
        # upcast (embeddings, norms, vision) would OOM at 27B on 24GB
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    dev = model.device

    epochs = int(cfg["epochs"]); accum = int(cfg["gradient_accumulation"])
    per_rank = len(train_rows) // world                 # each rank takes 1/world of the data
    total_opt = math.ceil(epochs * per_rank / accum)
    warmup = int(cfg.get("warmup_steps", 30))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["lr"]), weight_decay=0.0)
    def lr_l(s):
        if s < warmup: return s / max(1, warmup)
        pr = (s - warmup) / max(1, total_opt - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_l)
    log = (out / "train_log.jsonl").open("w", encoding="utf-8") if is_main else None
    eval_every = int(cfg.get("eval_every_steps", 100)); val_n = int(cfg.get("val_n", 64))
    log_every = int(cfg.get("log_every_steps", 10)); save_every = int(cfg.get("save_every_steps", 400))

    def write(d):
        if is_main:
            log.write(json.dumps(d) + "\n"); log.flush()

    micro = ostep = 0; run = 0.0; t0 = time.perf_counter(); model.train(); done = False
    peak = lambda: round(torch.cuda.max_memory_allocated() / 1e9, 2)
    for epoch in range(epochs):
        shuf = random.Random(cfg["seed"] * 7 + epoch)         # same shuffle across ranks
        order = list(range(len(train_rows))); shuf.shuffle(order)
        order = order[:per_rank * world][rank::world]         # disjoint, equal-size shard per rank (avoids deadlock)
        for ri in order:
            row = rows[train_rows[ri]["Id"].strip()]
            a = A.augment_structural(row, rng, float(cfg.get("reverse_prob", 0.5)), policy)
            frames = [IMG_ROOT / row["Id"] / n for n in a["order_names"]]
            answer = ranks_to_candidate(validate_permutation(list(a["ranks"])))
            if cfg.get("cot"):
                style = "caption" if (ri % 2 == 0) else "frame"
                answer = build_trace(a["caption"], a["ranks"], a["no_ordering"], a["reversed"], style) + "\n" + answer
            augp = A.draw_aug(rng) if (cfg.get("aug_prob", 0.5) and rng.random() < cfg["aug_prob"]) else None
            b = build_batch(proc, tok, frames, a["caption"], answer, up, augp, dev)
            loss = model(**b).loss / accum
            loss.backward(); run += float(loss.detach()) * accum; micro += 1
            if micro % accum == 0:
                if world > 1:                                  # average LoRA grads across ranks
                    gloo_cpu = cfg.get("ddp_backend") == "gloo"
                    for p in model.parameters():
                        if p.grad is not None:
                            if gloo_cpu:                       # gloo needs CPU tensors
                                g = p.grad.detach().cpu()
                                dist.all_reduce(g)
                                p.grad.copy_((g / world).to(p.grad.device))
                            else:
                                dist.all_reduce(p.grad); p.grad /= world
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True); ostep += 1
                if is_main and ostep % log_every == 0:
                    write({"elapsed_s": round(time.perf_counter() - t0, 1), "epoch": epoch, "opt_step": ostep,
                           "total_steps": total_opt, "train_loss": round(run / (accum * log_every), 5),
                           "lr": sch.get_last_lr()[0], "vram_gb": peak()})
                    run = 0.0
                if is_main and ostep % eval_every == 0:
                    vl = eval_val(model, proc, tok, val_rows, up, dev, val_n)
                    write({"epoch": epoch, "opt_step": ostep, "val_loss": round(vl, 5), "vram_gb": peak()})
                    print(f"  [step {ostep}] val_loss {vl:.4f} vram {peak()}GB", flush=True)
                if is_main and ostep % save_every == 0:
                    model.save_pretrained(out / "adapter_last")
            if args.max_steps and ostep >= args.max_steps:
                done = True; break
        if is_main:                                            # end of epoch: val loss + checkpoint (rank 0 only)
            vl = eval_val(model, proc, tok, val_rows, up, dev, val_n)
            model.save_pretrained(out / f"adapter_epoch{epoch+1}")
            write({"epoch_end": epoch + 1, "opt_step": ostep, "val_loss": round(vl, 5), "vram_gb": peak()})
            print(f"=== epoch {epoch+1} done | val_loss {vl:.4f} | ckpt adapter_epoch{epoch+1} | vram {peak()}GB", flush=True)
        if done:
            break

    if is_main:
        model.save_pretrained(out / "adapter_final")
        (out / "train_meta.json").write_text(json.dumps(
            {"config": cfg, "elapsed_min": (time.perf_counter() - t0) / 60, "opt_steps": ostep,
             "world": world, "train_samples": len(train_rows), "peak_vram_gb": peak()}, indent=2) + "\n")
        print(f"DONE opt_steps {ostep} | {(time.perf_counter()-t0)/60:.1f}min | peak {peak()}GB")
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
