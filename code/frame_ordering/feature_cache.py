import json
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np

from frame_ordering.data import TrainSample


def write_global_feature_cache(
    samples: Sequence[TrainSample],
    image_root: Path,
    encode_paths: Callable[[Sequence[Path]], np.ndarray],
    batch_size: int,
    output_path: Path,
) -> Dict[str, object]:
    if not samples:
        raise ValueError("Cannot extract features from an empty sample list")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    paths = [
        image_root / sample.sample_id / image_name
        for sample in samples
        for image_name in sample.image_names
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = None
    flat_cache = None
    feature_dim = None

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        encoded = np.asarray(encode_paths(batch_paths))
        if encoded.ndim != 2 or encoded.shape[0] != len(batch_paths):
            raise ValueError(
                f"Encoder returned {encoded.shape} for {len(batch_paths)} paths"
            )
        if cache is None:
            feature_dim = int(encoded.shape[1])
            cache = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(samples), 4, feature_dim),
            )
            flat_cache = cache.reshape(len(paths), feature_dim)
        if encoded.shape[1] != feature_dim:
            raise ValueError("Encoder feature dimension changed between batches")
        flat_cache[start : start + len(batch_paths)] = encoded.astype(
            np.float16, copy=False
        )

    cache.flush()
    ids = [sample.sample_id for sample in samples]
    output_path.with_suffix(".ids.json").write_text(
        json.dumps(ids, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "dtype": "float16",
        "feature_dim": feature_dim,
        "image_count": len(paths),
        "sample_count": len(samples),
        "shape": [len(samples), 4, feature_dim],
    }
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
