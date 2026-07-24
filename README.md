# ACE

This directory contains only the end-to-end implementation and evaluation
pipeline for ACE. Auxiliary experimental implementations are intentionally
excluded from this release.

## Environment

The validated environment is the repository Conda environment `xh2`:

```bash
conda activate xh2
cd code
```

Install the minimal runtime dependencies from `requirements.txt` when using a
new environment.

## Single experiment entrypoint

All ACE stages are launched through `scripts/ACE/run_ace_experiment.sh`.
The default model is `Qwen/Qwen3-30B-A3B-Instruct-2507`; set `MODEL_FAMILY`
and `MODEL_PATH` to evaluate another supported model.

### 1. Calibrate a target pruning ratio

```bash
./scripts/ACE/run_ace_experiment.sh calibrate \
  --output-path results/ACE/thresholds.json \
  --target-pruning-ratios 0.1 0.2 0.3 0.4 0.5 0.6
```

Calibration uses the first consecutive `128 * 2048` tokens from the WikiText
training split. The ACE score is
`max(1.0 * normalized_GSP, 0.1 * normalized_RCR)`.

### 2. Evaluate WikiText-2 perplexity

```bash
./scripts/ACE/run_ace_experiment.sh ppl --taus 0.0 0.05 0.10
```

PPL uses the standard sequence length of 2048.

### 3. Evaluate downstream tasks with EvalScope

```bash
./scripts/ACE/run_ace_experiment.sh evalscope \
  --tau 0.10 \
  --datasets arc math_qa openbookqa
```

Use `--router-weight-centering` only when explicitly evaluating the optional
RCR centering variant; the paper-compatible default is disabled.

## Tests

```bash
pytest -q
```

The tests cover ACE branch fusion, amplification tables, model-family
selection, quantile threshold construction, and cache metadata.