"""Canonical dataset registry for the dataset-driven pruning-ratio search.

All CLI entrypoints share these identifiers. The search driver stages every
result under ``results/<method>/<model_name>/<dataset>_search/`` so plots and
reuse checks can find per-dataset artefacts in a single place.

``wikitext`` is the PPL dataset and is served by the in-house PPL evaluator.
Every other identifier maps to an evalscope benchmark (optionally with a
fixed subset, e.g. ``arc-e`` -> ``arc`` + ``ARC-Easy``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    """Describes one dataset slot in the staged search layout."""

    name: str                       # canonical CLI name (e.g. "arc-e")
    kind: str                       # "ppl" or "evalscope"
    metric_key: str                 # key used in search_records (e.g. "ppl", "arc_easy")
    evalscope_benchmark: Optional[str] = None   # name passed to TaskConfig.datasets
    evalscope_subset: Optional[str] = None      # optional subset_list member
    evalscope_report_filename: Optional[str] = None  # filename under reports/<model>/
    higher_is_better: bool = True


_PPL = 'ppl'
_EVALSCOPE = 'evalscope'


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    'wikitext': DatasetSpec(
        name='wikitext',
        kind=_PPL,
        metric_key='ppl',
        higher_is_better=False,
    ),
    'humaneval': DatasetSpec(
        name='humaneval',
        kind=_EVALSCOPE,
        metric_key='humaneval',
        evalscope_benchmark='humaneval',
        evalscope_report_filename='humaneval.json',
    ),
    'piqa': DatasetSpec(
        name='piqa',
        kind=_EVALSCOPE,
        metric_key='piqa',
        evalscope_benchmark='piqa',
        evalscope_report_filename='piqa.json',
    ),
    'math500': DatasetSpec(
        name='math500',
        kind=_EVALSCOPE,
        metric_key='math500',
        evalscope_benchmark='math_500',
        evalscope_report_filename='math_500.json',
    ),
    'arc-e': DatasetSpec(
        name='arc-e',
        kind=_EVALSCOPE,
        metric_key='arc_easy',
        evalscope_benchmark='arc',
        evalscope_subset='ARC-Easy',
        evalscope_report_filename='arc.json',
    ),
    'arc-c': DatasetSpec(
        name='arc-c',
        kind=_EVALSCOPE,
        metric_key='arc_challenge',
        evalscope_benchmark='arc',
        evalscope_subset='ARC-Challenge',
        evalscope_report_filename='arc.json',
    ),
    'aime25': DatasetSpec(
        name='aime25',
        kind=_EVALSCOPE,
        metric_key='aime25',
        evalscope_benchmark='aime25',
        evalscope_report_filename='aime25.json',
    ),
    'gpqa': DatasetSpec(
        name='gpqa',
        kind=_EVALSCOPE,
        metric_key='gpqa',
        evalscope_benchmark='gpqa_diamond',
        evalscope_report_filename='gpqa_diamond.json',
    ),
    'livecodebench': DatasetSpec(
        name='livecodebench',
        kind=_EVALSCOPE,
        metric_key='livecodebench',
        evalscope_benchmark='live_code_bench',
        evalscope_report_filename='live_code_bench.json',
    ),
}


PPL_DATASETS: Tuple[str, ...] = tuple(n for n, s in DATASET_REGISTRY.items() if s.kind == _PPL)
EVALSCOPE_DATASETS: Tuple[str, ...] = tuple(n for n, s in DATASET_REGISTRY.items() if s.kind == _EVALSCOPE)
ALL_DATASETS: Tuple[str, ...] = tuple(DATASET_REGISTRY.keys())


def require_dataset(name: str) -> DatasetSpec:
    key = str(name).strip().lower()
    if key not in DATASET_REGISTRY:
        raise SystemExit(
            f'Unknown dataset {name!r}. Expected one of: {sorted(DATASET_REGISTRY)}.'
        )
    return DATASET_REGISTRY[key]


def default_knob_bounds_for_dataset(dataset: str, method: str) -> Tuple[float, float]:
    """Default tau/beta bounds.

        - PPL (wikitext) uses [0,0.4] for ACE/GSP/RCR and AIMER.
    - Evalscope accuracy datasets use method-specific bounds:
    ACE/GSP/RCR [0,0.4], DiEP [0,2],
      MoDES [0,0.012], NAEE [0,0.8], score_only [0,0.8],
    ExpertSparsity [0,1], AIMER dynamic adaptation [0,1], TopP [0,1],
    SERE-style rerouting [0,1],
      XShare-style batch-aware pruning [0,1].
    """
    spec = require_dataset(dataset)
    method_lc = str(method).lower()
    if spec.kind == _PPL:
        if method_lc in {'ace', 'gsp', 'rcr', 'aimer'}:
            return 0.0, 0.4
        if method_lc == 'naee':
            return 0.0, 1.0
        if method_lc == 'expert_sparsity':
            return 0.0, 1.0
        if method_lc in {'top_p', 'sere', 'xshare'}:
            return 0.0, 1.0
        return 0.0, 1.0
    # evalscope accuracy datasets
    if method_lc in {'ace', 'gsp', 'rcr'}:
        return 0.0, 0.4
    if method_lc == 'aimer':
        return 0.0, 1.0
    if method_lc == 'diep':
        return 0.0, 2.0
    if method_lc == 'modes':
        return 0.0, 0.012
    if method_lc in {'naee', 'score_only'}:
        return 0.0, 0.8
    if method_lc == 'expert_sparsity':
        return 0.0, 1.0
    if method_lc in {'top_p', 'sere', 'xshare'}:
        return 0.0, 1.0
    return 0.0, 2.0


def default_evalscope_generation_max_tokens(dataset: str) -> int:
    """Conservative generation caps for search-time evalscope runs.

    Search repeatedly probes high-pruning knobs where models may emit long
    garbage continuations instead of terminating naturally. Keep search-time
    caps much lower than the final 8192-token evaluation default so obviously
    bad runs fail fast without changing the full-eval entrypoint defaults.
    """
    key = require_dataset(dataset).name
    if key in {'arc-e', 'arc-c', 'piqa', 'gpqa'}:
        return 1024
    if key in {'aime25', 'math500', 'humaneval', 'livecodebench'}:
        return 2048
    return 2048


def model_name_for_path(model_path: str) -> str:
    """Derive the <model_name> folder name used in staged result paths."""
    from pathlib import Path
    return Path(str(model_path)).name


def dataset_search_dir(
    results_root,  # type: ignore[no-untyped-def]
    method_dir_name: str,
    model_name: str,
    dataset: str,
):
    """Canonical staged search directory: <results>/<method>/<model>/<dataset>_search."""
    from pathlib import Path
    return Path(results_root) / method_dir_name / model_name / f'{dataset}_search'
