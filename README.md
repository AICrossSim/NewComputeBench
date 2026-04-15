# NewComputeBench

[![Doc](https://img.shields.io/badge/docs-online-blue)](https://aicrosssim.github.io/NewComputeBench/)
[![GitHub](https://img.shields.io/badge/github-AICrossSim%2FNewComputeBench-black)](https://github.com/AICrossSim/NewComputeBench)

**NewComputeBench** is a benchmark suite for new compute paradigms — Spiking Neural Networks, Optical computation, Processing-in-Memory, and more — via software emulation.
We aim to predict the scaling law of neural networks trained with new compute paradigms by running small- and medium-scale experiments and extrapolating observed trends.

> **📖 Full documentation: [aicrosssim.github.io/NewComputeBench](https://aicrosssim.github.io/NewComputeBench/)**

---

## Overview

The project is structured around three phases:

1. Build a scaling framework for language model pretraining up to 1.1B parameters (AICrossSim-CLM series)
2. Implement software emulation of new compute paradigms
3. Filter out promising paradigms through small- and medium-scale experiments, then scale up

## Quick Start

```bash
git clone https://github.com/AICrossSim/NewComputeBench.git
cd NewComputeBench
git submodule update --init
```

**Option 1 — uv** (recommended, assumes CUDA is pre-installed on the system)

```bash
uv sync
uv pip install -e ./submodules/mase   # install MASE quantization backend
```

**Option 2 — conda + pip** (use this if CUDA is not pre-installed)

```bash
conda env create -f environment.yaml   # installs Python 3.11 + CUDA Toolkit
conda activate new-compute
pip install -r requirements.txt
pip install -e ./submodules/mase
```

```bash
# Run inference with a pretrained model
cd experiments/llm-digital/pretrain
python run.py hf-gen --model_name AICrossSim/clm-60m --prompt "London is"
```

See the [Installation Guide](https://aicrosssim.github.io/NewComputeBench/modules/getting_started/installation.html) for full setup instructions.

## Tutorials

| Topic | Link |
|-------|------|
| LLM Pretraining & Evaluation | [docs](https://aicrosssim.github.io/NewComputeBench/modules/tutorials/pretraining/llm_pretrain_eval.html) |
| Random Bitflip on CLM | [docs](https://aicrosssim.github.io/NewComputeBench/modules/tutorials/simulations/bitflip_clm.html) |
| Bitflip-Aware LoRA Fine-Tuning (Llama-3.1-8B) | [docs](https://aicrosssim.github.io/NewComputeBench/modules/tutorials/simulations/bitflip_lora.html) |
| Optical Neural Networks on RoBERTa | [docs](https://aicrosssim.github.io/NewComputeBench/modules/tutorials/simulations/onn_roberta.html) |
| Optical Neural Networks on CLM | [docs](https://aicrosssim.github.io/NewComputeBench/modules/tutorials/simulations/onn_clm.html) |
| Spiking Neural Networks on RoBERTa | [docs](https://aicrosssim.github.io/NewComputeBench/modules/tutorials/simulations/snn_roberta.html) |
| Processing-in-Memory on RoBERTa | [docs](https://aicrosssim.github.io/NewComputeBench/modules/tutorials/simulations/pim_roberta.html) |
| Processing-in-Memory on ViT | [docs](https://aicrosssim.github.io/NewComputeBench/modules/tutorials/simulations/pim_vit.html) |

## Pretrained Models

Our pretrained AICrossSim-CLM checkpoints are available on HuggingFace:

| Model | HuggingFace |
|-------|-------------|
| CLM-60M (clean) | [AICrossSim/clm-60m](https://huggingface.co/AICrossSim/clm-60m) |
| CLM-200M (clean) | [AICrossSim/clm-200m](https://huggingface.co/AICrossSim/clm-200m) |
| CLM-400M (clean) | [AICrossSim/clm-400m](https://huggingface.co/AICrossSim/clm-400m) |
| CLM-1.1B (clean) | [AICrossSim/clm-1.1b](https://huggingface.co/AICrossSim/clm-1.1b) |
| CLM-60M (bitflip-aware) | [AICrossSim/bitflip-fc-clm-60m](https://huggingface.co/AICrossSim/bitflip-fc-clm-60m) |
| CLM-200M (bitflip-aware) | [AICrossSim/bitflip-fc-clm-200m](https://huggingface.co/AICrossSim/bitflip-fc-clm-200m) |
| CLM-400M (bitflip-aware) | [AICrossSim/bitflip-fc-clm-400m](https://huggingface.co/AICrossSim/bitflip-fc-clm-400m) |
| CLM-1.1B (bitflip-aware) | [AICrossSim/bitflip-fc-clm-1.1b](https://huggingface.co/AICrossSim/bitflip-fc-clm-1.1b) |

## About

This project is led by [Dr. Yiren Zhao](https://aaron-zhao123.github.io/) (Imperial College London), [Dr. Luo Mai](https://luomai.github.io/) (University of Edinburgh), and [Prof. Robert Mullins](https://www.cl.cam.ac.uk/~rdm34/) (University of Cambridge), funded by [ARIA](https://www.aria.org.uk/).
