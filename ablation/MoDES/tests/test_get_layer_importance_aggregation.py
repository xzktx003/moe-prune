from pathlib import Path


def test_get_layer_importance_uses_reduce_and_zero_guard() -> None:
    text = Path(
        Path(__file__).resolve().parents[1] / "get_layer_importance_ddp.py"
    ).read_text(encoding="utf-8")

    assert 'accelerator.reduce(' in text
    assert 'No completion tokens were counted during MoDES calibration aggregation.' in text
    assert 'final_layer_loss_dict[layer_idx].get(modality, 0.0)' in text
