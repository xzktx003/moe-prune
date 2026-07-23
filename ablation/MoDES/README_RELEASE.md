# MoDES subset included with ACE

This directory contains the text-model runtime subset used to reproduce the
MoDES baseline reported in the ACE paper.

Upstream project:

- **MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping**
- Project page/source: https://github.com/ModelTC/MoDES
- License: Apache License 2.0; see `LICENSE` and `LICENCE` in this directory.

The original project supports multimodal models, Kimi-VL, GQA, COCO,
VideoMMMU, and LMMS evaluation. Those components are outside the experiments
reported in the ACE paper and are intentionally not redistributed here.

Included files provide the Qwen3/Gemma text MoE runtime patch, WikiText helpers,
and the text PPL/MCQA evaluation adapters used by the surrounding scripts.
MoDES calibration artifacts are generated data and are not included. Supply a
valid layer-importance pickle with the relevant ACE evaluation/search command.

For the full upstream implementation, multimodal datasets, or original
frontier-search pipeline, obtain the official MoDES repository and follow its
license and setup instructions.
