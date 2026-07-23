from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = ROOT / "ablation" / "MoDES" / "utils.py"
SPEC = importlib.util.spec_from_file_location("modes_utils", UTILS_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_create_mask_from_prompt_attention_handles_left_padding() -> None:
    full_attention = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    prompt_attention = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    mask = MODULE.create_mask_from_prompt_attention(full_attention, prompt_attention)

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [False, False, False, False, True, True],
                [False, False, False, False, True, True],
            ]
        ),
    )


def test_create_mask_from_prompt_attention_returns_empty_mask_when_no_completion() -> None:
    attention = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)

    mask = MODULE.create_mask_from_prompt_attention(attention, attention)

    assert torch.equal(mask, torch.tensor([[False, False, False, False]]))
