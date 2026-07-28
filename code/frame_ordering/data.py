import ast
import csv
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

from frame_ordering.permutations import Permutation, validate_permutation

REQUIRED_COLUMNS = (
    "Id",
    "Input_1",
    "Input_2",
    "Input_3",
    "Input_4",
    "Sentence",
    "Answer",
    "No_ordering",
)


@dataclass(frozen=True)
class TrainSample:
    sample_id: str
    image_names: Tuple[str, str, str, str]
    sentence: str
    ranks: Permutation
    no_ordering: bool


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected True or False, got {value!r}")


def load_train_samples(
    csv_path: Path, image_root: Path, verify_images: bool = True
) -> List[TrainSample]:
    samples = []
    seen_ids = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(f"Unexpected columns: {reader.fieldnames}")
        for row in reader:
            sample_id = row["Id"].strip()
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"Invalid or duplicate Id: {sample_id!r}")
            seen_ids.add(sample_id)
            image_names = tuple(
                row[f"Input_{index}"].strip() for index in range(1, 5)
            )
            if verify_images:
                for image_name in image_names:
                    image_path = image_root / sample_id / image_name
                    if not image_path.is_file():
                        raise FileNotFoundError(str(image_path))
                    with Image.open(image_path) as image:
                        image.verify()
            samples.append(
                TrainSample(
                    sample_id=sample_id,
                    image_names=image_names,  # type: ignore[arg-type]
                    sentence=row["Sentence"],
                    ranks=validate_permutation(ast.literal_eval(row["Answer"])),
                    no_ordering=parse_bool(row["No_ordering"]),
                )
            )
    if not samples:
        raise ValueError("Training CSV contains no samples")
    return samples


def dataset_report(samples: Sequence[TrainSample]) -> Dict[str, object]:
    if not samples:
        raise ValueError("Cannot report an empty dataset")
    answer_counts = Counter(sample.ranks for sample in samples)
    no_ordering_counts = Counter(sample.no_ordering for sample in samples)
    lengths = [len(sample.sentence.split()) for sample in samples]
    return {
        "sample_count": len(samples),
        "unique_permutation_count": len(answer_counts),
        "answer_counts": {
            str(list(ranks)): count
            for ranks, count in sorted(answer_counts.items())
        },
        "no_ordering_counts": {
            "false": no_ordering_counts[False],
            "true": no_ordering_counts[True],
        },
        "unique_sentence_count": len({sample.sentence for sample in samples}),
        "sentence_words": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "mean": statistics.fmean(lengths),
            "max": max(lengths),
        },
    }
