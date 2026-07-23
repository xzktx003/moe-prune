# Third-Party Notices

ACE is distributed under the Apache License 2.0 at the repository root. Some
subdirectories contain third-party research implementations under their own
licenses. Those licenses are not replaced by the root license.

## DiEP

- Location: `ablation/DiEP/`
- Project: DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning
- License: MIT
- License file: `ablation/DiEP/LICENSE`
- Upstream authorship and citation are retained in `ablation/DiEP/README.md`.

The bundled copy is reduced to the files needed by this repository. Large
sample data, generated pruning caches, and a vendored LM Evaluation Harness
copy are not redistributed. The official `lm-eval` package is used instead.

## NAEE / Expert Sparsity

- Location: `ablation/NAEE/`
- Project: Not All Experts Are Equal / Expert Sparsity research code
- License: MIT
- License file: `ablation/NAEE/LICENSE`
- Upstream attribution is retained in `ablation/NAEE/README.md`.

## MoDES

- Location: `ablation/MoDES/`
- Project: MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
- License: Apache License 2.0
- License files: `ablation/MoDES/LICENSE` and `ablation/MoDES/LICENCE`
- Upstream attribution and scope are documented in `ablation/MoDES/README_RELEASE.md`.

Only the text-model runtime pieces used by the paper are included. The upstream
multimodal datasets, Kimi-VL support, and LMMS evaluation integration are not
redistributed.

## Other dependencies

This project uses PyTorch, Hugging Face Transformers, Datasets, Accelerate,
EvalScope, Triton, and LM Evaluation Harness. They are installed as external
packages and remain subject to their respective upstream licenses.

Model weights and datasets are not included. Users must obtain them from their
original providers and comply with the corresponding licenses and terms.
