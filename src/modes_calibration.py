from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def calibration_path(results_root: Path, model_path: str, loss_type: str, num_samples: int) -> Path:
    model_name = Path(model_path).name
    return results_root / "calibration" / "wiki" / model_name / f"{loss_type}_0_{num_samples}.pkl"


def calibration_payload_is_valid(payload: Any) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    for layer_idx, modality_scores in payload.items():
        if not isinstance(layer_idx, int):
            return False
        if not isinstance(modality_scores, dict) or not modality_scores:
            return False
        for modality, score in modality_scores.items():
            if not isinstance(modality, str):
                return False
            if not isinstance(score, (int, float)):
                return False
    return True


def calibration_file_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        if path.stat().st_size <= 0:
            return False
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, TypeError, ValueError):
        return False
    return calibration_payload_is_valid(payload)