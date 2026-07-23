# ACE: Parameter-Driven Expert Skipping for MoE LLMs

This repository contains the implementation accompanying the ACE paper. ACE
combines two parameter-derived expert contribution views:

- **GSP (Global Spectral Proxy):** estimates global transformation capacity from
  frozen expert and normalization parameters.
- **RCR (Router-Conditioned Refinement):** evaluates each expert along a
  router-conditioned direction derived from the layer's router weights.
- **ACE:** normalizes the runtime gate-weighted GSP and RCR scores separately and
  uses their elementwise maximum. A routed slot is skipped only when both views
  assign it low contribution.

The repository exposes only the methods and component ablations reported in
the paper. `gsp` and `rcr` are retained only for the reported component study.

## Released methods

Core method and ablations:

- `ace`
- `gsp`
- `rcr`

Paper baselines included in the common runtime/evaluation code:

- `score_only` (router Score)
- `naee`
- `modes`
- `diep`
- `aimer`
- `expert_sparsity`
- `top_p`
- `sere`
- `xshare`

Third-party source subsets are under `ablation/`. See
`THIRD_PARTY_NOTICES.md` and the nested license files.

## Installation

Python 3.10 is recommended. Install the pinned environment:

```bash
pip install -r requirements.txt
```

FlashAttention is optional and should be installed separately when needed:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

Run commands from the repository root or add the repository root to
`PYTHONPATH`. The compatibility package preserves imports under
`moe_prune.code.*` without requiring a particular checkout directory name.

## Models

Default public model identifiers are:

- Qwen3: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Qwen3.6 family: `Qwen/Qwen3.5-35B-A3B`
- Gemma-4: `google/gemma-4-26b-a4b-it`

A local checkpoint path may be supplied with `--model-path`. Defaults can also
be overridden with `ACE_QWEN3_MODEL`, `ACE_QWEN36_MODEL`, and
`ACE_GEMMA4_MODEL`.

Model weights and datasets are not distributed by this repository.

## ACE perplexity evaluation

The paper's WikiText-2 PPL protocol uses sequence length 2048:

```bash
bash scripts/ACE/run_ace_ppl.sh --taus 0.10 0.20 0.30
```

Equivalent module invocation:

```bash
python -m moe_prune.code.scripts.shared.ppl_eval \
  --method ace \
  --model-family qwen3 \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --n-ctx 2048 \
  --n-batch 2048 \
  --taus 0.10 0.20 0.30 \
  --output-dir results/ACE/ppl
```

Use `--method gsp` or `--method rcr` only to reproduce the component ablation.

## ACE EvalScope evaluation

```bash
bash scripts/ACE/run_ace_evalscope.sh \
  --tau 0.20 \
  --datasets arc piqa math_500 gpqa humaneval live_code_bench
```

Inspect the exact benchmark identifiers accepted by the installed EvalScope
version with:

```bash
python -m moe_prune.code.scripts.shared.run_evalscope_eval --help
```

## Threshold calibration

ACE supports one-pass quantile threshold construction. The candidate pool
excludes the original router top-1 slot, matching runtime behavior. The top-1
slot is always retained, remaining gates are renormalized, and `min_keep` may
enforce additional active experts.

```bash
python -m moe_prune.code.scripts.shared.quantile_calibration \
  --method ace \
  --model-family qwen3 \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --target-pruning-ratios 0.1 0.2 0.3 0.4 0.5 0.6 \
  --n-ctx 2048 \
  --n-batch 2048 \
  --output-path results/ACE/threshold_table.json
```

## Router-weight centering

RCR supports layer-wise router-weight centering:

```text
W_router <- W_router - mean(W_router, dim=experts)
```

The implementation switch is `--router-weight-centering`. It is disabled by
default for backward-compatible execution. Enable it when reproducing the
centered RCR formulation in the paper:

```bash
bash scripts/ACE/run_ace_ppl.sh --router-weight-centering --taus 0.20
```

Centering affects only the RCR branch.

## Tests

```bash
pytest -q test
```

The tests cover ACE fusion and top-1 semantics, GSP/RCR table construction,
router-centering invariance, supported baseline selectors, model-family
resolution, and retained third-party adapters.

## Repository scope

The release excludes generated outputs, model checkpoints, datasets, caches,
private paths, cluster-specific launch files, and algorithms not reported in
the paper. The repository contains no Git remote configuration by default.

## License

ACE-authored code is licensed under Apache License 2.0. Third-party components
remain under their respective nested licenses. See `LICENSE`, `NOTICE`, and
`THIRD_PARTY_NOTICES.md`.
