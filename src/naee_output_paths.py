from __future__ import annotations

from pathlib import Path


def model_tag_for_path(model_path: str) -> str:
    return Path(str(model_path)).name or 'model'


def legacy_shared_output_root(method: str) -> Path:
    normalized = str(method).lower()
    if normalized == 'score_only':
        return Path('results/score_only')
    return Path('results/NAEE')


def resolve_naee_output_dir(output_dir: Path, *, method: str, model_path: str) -> Path:
    model_tag = model_tag_for_path(model_path)
    shared_root = legacy_shared_output_root(method)

    if output_dir == shared_root:
        return output_dir / model_tag
    if output_dir.name == shared_root.name and output_dir.parent.name == shared_root.parent.name:
        return output_dir / model_tag
    return output_dir


def naee_ppl_result_path(output_dir: Path, beta: float) -> Path:
    return output_dir / 'ppl_by_tau' / f'tau_{float(beta):.4f}' / 'wikitext_ppl.json'
