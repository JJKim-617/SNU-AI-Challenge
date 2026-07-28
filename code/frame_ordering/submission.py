import csv
from pathlib import Path
from typing import Dict, Iterable

from frame_ordering.permutations import validate_permutation


def write_submission(
    predictions: Dict[str, Iterable[int]],
    sample_submission_path: Path,
    output_path: Path,
) -> None:
    """Write competition Answers (per-input chronological ranks) in the
    exact row order of sample_submission.csv."""
    submission_ids = []
    with sample_submission_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or reader.fieldnames[:2] != ["Id", "Answer"]:
            raise ValueError(
                f"Unexpected sample submission columns: {reader.fieldnames}"
            )
        for row in reader:
            submission_ids.append(row["Id"].strip())
    if len(submission_ids) != len(set(submission_ids)):
        raise ValueError("Duplicate Ids in sample submission")

    missing = [
        sample_id for sample_id in submission_ids if sample_id not in predictions
    ]
    if missing:
        raise ValueError(f"Missing predictions for: {missing[:5]}")
    extra = sorted(set(predictions) - set(submission_ids))
    if extra:
        raise ValueError(f"Predictions contain extra Ids: {extra[:5]}")

    validated = {
        sample_id: validate_permutation(predictions[sample_id])
        for sample_id in submission_ids
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Id", "Answer"])
        for sample_id in submission_ids:
            writer.writerow([sample_id, str(list(validated[sample_id]))])
