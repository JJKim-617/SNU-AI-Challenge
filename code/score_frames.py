import argparse
import ast
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from frame_ordering.permutations import validate_permutation
from frame_ordering.qwen_ordering import (
    ANSWER_CANDIDATES,
    build_messages,
    candidate_to_ranks,
)


def load_rows(train_csv, split_csv, split, limit, shard_index=0, shard_count=1, ids_file=None):
    rows = []
    with train_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if split_csv is None:
        selected = rows
    else:
        split_by_id = {}
        with split_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                split_by_id[row["Id"]] = row["Split"]
        selected = [row for row in rows if split_by_id[row["Id"]] == split]
    if ids_file is not None:
        keep = set(json.loads(Path(ids_file).read_text()))
        selected = [row for row in selected if row["Id"].strip() in keep]
    if limit:
        stride = max(1, len(selected) // limit)
        selected = selected[::stride][:limit]
    if shard_count > 1:
        selected = selected[shard_index::shard_count]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--upscale", type=float, default=1.0,
                        help="PIL upscale factor; raise --max-pixels together")
    parser.add_argument("--candidate-batch", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--load-4bit", action="store_true",
                        help="load the base model in 4-bit NF4")
    parser.add_argument("--prompt-a0", action="store_true",
                        help="use the training prompt for scoring")
    parser.add_argument("--slot-tokens", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--sofa-alpha", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sofa-large", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sofa-path-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sofa_path_only and args.sofa_alpha is None:
        args.sofa_alpha = 1.0  # value is ignored in path-only mode
    if args.slot_tokens and not args.prompt_a0:
        parser.error("--slot-tokens requires --prompt-a0")

    rows = load_rows(
        args.train_csv,
        args.split_csv,
        args.split,
        args.limit,
        args.shard_index,
        args.shard_count,
        ids_file=args.ids_file,
    )
    processor = AutoProcessor.from_pretrained(
        str(args.model_dir), local_files_only=True
    )
    load_kwargs = dict(dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(str(args.model_dir), **load_kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    if args.sofa_alpha is not None:
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sofa_patch import enable_sofa

        enable_sofa(model, args.sofa_alpha, large=args.sofa_large,
                    path_only=args.sofa_path_only)
    device = model.device
    tokenizer = processor.tokenizer

    slot_active = False
    if args.slot_tokens:
        import hashlib
        st = torch.load(args.slot_tokens, map_location="cpu", weights_only=False)
        sp = st["slot_params"].float()
        sha = hashlib.sha256(sp.numpy().tobytes()).hexdigest()[:16]
        assert sha == st["sha"], f"slot_tokens sha mismatch: file {st['sha']} vs computed {sha}"
        n_added = tokenizer.add_tokens(st["tokens"], special_tokens=True)
        got_ids = [tokenizer.convert_tokens_to_ids(t) for t in st["tokens"]]
        assert got_ids == list(st["token_ids"]), f"token id mismatch: {got_ids} vs {st['token_ids']}"
        emb_rows = model.get_input_embeddings().weight.shape[0]
        assert max(got_ids) < emb_rows, f"no vocab headroom at eval: {max(got_ids)} >= {emb_rows}"
        slot_vals = sp.to(device)
        def _slot_hook(module, inputs, output, _ids=got_ids, _vals=slot_vals):
            ids = inputs[0]
            out = output
            cloned = False
            for j, tid in enumerate(_ids):
                m = ids == tid
                if m.any():
                    if not cloned:
                        out = out.clone(); cloned = True
                    out[m] = _vals[j].to(out.dtype)
            return out
        model.get_input_embeddings().register_forward_hook(_slot_hook)
        slot_active = True
        print(f"[slot-tokens] injected {n_added} tokens ids {got_ids[0]}..{got_ids[-1]} sha {sha}", flush=True)

    candidate_ids = [
        tokenizer(text, add_special_tokens=False).input_ids
        for text in ANSWER_CANDIDATES
    ]
    lengths = {len(ids) for ids in candidate_ids}
    if len(lengths) != 1:
        raise ValueError(f"Candidate token lengths differ: {lengths}")
    candidate_length = lengths.pop()
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long)

    per_sample = []
    correct = 0
    correct_no_order = total_no_order = 0
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_suffix(".progress.jsonl")
    progress = progress_path.open("w", encoding="utf-8")

    for index, row in enumerate(rows):
        image_paths = [
            args.image_root / row["Id"] / row[f"Input_{i}"].strip()
            for i in range(1, 5)
        ]
        if args.prompt_a0:
            a0_prompt = ('These four images are frames from one video, shown in shuffled order (Input 1-4).\n'
                         'Story: "{caption}"\n'
                         'List the input image numbers in chronological order as a JSON list, e.g. [2, 4, 1, 3].')
            if slot_active:      # interleave slot/register tokens with the frames
                content = []
                for k, p in enumerate(image_paths, start=1):
                    content.append({"type": "text", "text": f"<|slot{k}|>"})
                    content.append({"type": "image", "image": str(p)})
                    content.append({"type": "text", "text": f"<|reg{k}a|><|reg{k}b|><|reg{k}c|>"})
                content.append({"type": "text", "text": a0_prompt.format(caption=row["Sentence"])})
                messages = [{"role": "user", "content": content}]
            else:
                messages = [{"role": "user", "content":
                             [{"type": "image", "image": str(p)} for p in image_paths]
                             + [{"type": "text", "text": a0_prompt.format(caption=row["Sentence"])}]}]
        else:
            messages = build_messages([str(p) for p in image_paths], row["Sentence"])
        for item in messages[0]["content"]:
            if item["type"] == "image":
                item["max_pixels"] = args.max_pixels
                if args.upscale != 1.0:
                    im = Image.open(item["image"]).convert("RGB")
                    item["image"] = im.resize(
                        (int(im.width * args.upscale), int(im.height * args.upscale))
                    )
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=images, return_tensors="pt")
        prompt_ids = inputs["input_ids"]
        prompt_length = prompt_ids.shape[1]

        scores = torch.empty(24)
        with torch.inference_mode():
            for start in range(0, 24, args.candidate_batch):
                stop = min(start + args.candidate_batch, 24)
                count = stop - start
                batch_ids = torch.cat(
                    [
                        prompt_ids.expand(count, -1),
                        candidate_tensor[start:stop],
                    ],
                    dim=1,
                ).to(device)
                batch = {
                    "input_ids": batch_ids,
                    "attention_mask": torch.ones_like(batch_ids),
                    "pixel_values": inputs["pixel_values"]
                    .repeat(count, 1)
                    .to(device),
                    "image_grid_thw": inputs["image_grid_thw"]
                    .repeat(count, 1)
                    .to(device),
                }
                if "mm_token_type_ids" in inputs:  # candidate tokens are text (type 0)
                    mm = inputs["mm_token_type_ids"]
                    cand_mm = torch.zeros((count, candidate_length), dtype=mm.dtype)
                    batch["mm_token_type_ids"] = torch.cat(
                        [mm.expand(count, -1), cand_mm], dim=1).to(device)
                # apply lm_head only at the candidate positions (avoids OOM)
                logits = model(**batch, logits_to_keep=candidate_length + 1).logits
                shift_logits = logits[:, :-1]
                log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
                targets = candidate_tensor[start:stop].to(device)
                token_scores = log_probs.gather(
                    2, targets.unsqueeze(-1)
                ).squeeze(-1)
                scores[start:stop] = token_scores.sum(dim=1).cpu()

        best = int(scores.argmax())
        predicted_ranks = candidate_to_ranks(ANSWER_CANDIDATES[best])
        record = {
            "id": row["Id"],
            "predicted_ranks": list(predicted_ranks),
            "scores": [round(float(s), 4) for s in scores],
        }
        if "Answer" in row and row["Answer"]:
            answer = validate_permutation(ast.literal_eval(row["Answer"]))
            hit = predicted_ranks == answer
            correct += hit
            record["answer"] = list(answer)
            record["correct"] = hit
            if row.get("No_ordering", "").strip().lower() == "true":
                total_no_order += 1
                correct_no_order += hit
        per_sample.append(record)
        done = index + 1
        if done % 20 == 0:
            progress.write(
                json.dumps(
                    {
                        "done": done,
                        "total": len(rows),
                        "em": round(correct / done, 4),
                        "elapsed_s": round(time.perf_counter() - started, 1),
                    }
                )
                + "\n"
            )
            progress.flush()
    progress.close()

    total = len(rows)
    metrics = {
        "adapter": str(args.adapter) if args.adapter else None,
        "candidate_length_tokens": candidate_length,
        "exact_match": correct / total if total else 0.0,
        "exact_match_no_ordering": (
            correct_no_order / total_no_order if total_no_order else None
        ),
        "exact_match_ordering": (
            (correct - correct_no_order) / (total - total_no_order)
            if total > total_no_order
            else None
        ),
        "max_pixels": args.max_pixels,
        "upscale": args.upscale,
        "model_dir": str(args.model_dir),
        "sample_count": total,
        "seconds_per_sample": (time.perf_counter() - started) / max(total, 1),
        "sofa_alpha": args.sofa_alpha,
        "sofa_large": args.sofa_large,
        "sofa_path_only": args.sofa_path_only,
        "split": args.split,
    }
    if args.sofa_alpha is not None:
        from sofa_patch import sofa_stats

        metrics["sofa_stats"] = sofa_stats()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metrics": metrics, "per_sample": per_sample}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
