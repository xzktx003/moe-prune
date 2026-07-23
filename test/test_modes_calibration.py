from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "code" / "src" / "modes_calibration.py"
SPEC = importlib.util.spec_from_file_location("modes_calibration_local", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_calibration_payload_requires_non_empty_layer_scores() -> None:
    assert MODULE.calibration_payload_is_valid({0: {"text": 0.5}})
    assert not MODULE.calibration_payload_is_valid({})
    assert not MODULE.calibration_payload_is_valid({0: {}})
    assert not MODULE.calibration_payload_is_valid({"0": {"text": 0.5}})


def test_calibration_file_validation_rejects_empty_and_accepts_pickle(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.pkl"
    empty_path.write_bytes(b"")
    assert not MODULE.calibration_file_is_valid(empty_path)

    valid_path = tmp_path / "valid.pkl"
    with valid_path.open("wb") as handle:
        pickle.dump({0: {"text": 0.25}, 3: {"text": 0.75}}, handle)

    assert MODULE.calibration_file_is_valid(valid_path)