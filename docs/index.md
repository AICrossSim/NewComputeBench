# AICrossSim/NewComputeBench

<figure markdown="span">
  ![NewComputeBench](./images/logo.png){ width="200" }
  <figcaption>NewComputeBench</figcaption>
</figure>

**[AICrossSim/NewComputeBench](https://github.com/AICrossSim/NewComputeBench)** is a benchmark suite for new compute paradigms (Spiking neural networks, Optical computation, In-Memory computation, etc) via software emulation. We aim to predict the scaling law of neural networks trained with new compute paradigms by running small & medium scale experiments and extrapolate the trends we observed. NewComputeBench project mainly consists of the following steps:

- [x] Build a scaling framework to support the pre-training of language models up to 1.1B parameters (CLM model series)
- [x] Implement software emulation of new compute paradigms (e.g., optical compute, spiking neural networks, in-memory compute, etc)
- [x] Filter out promising new compute paradigms by running small & medium scale experiments (Roberta on GLUE)
- [ ] Scale up the promising new compute paradigms to large-scale language models
  - [ ] Fine-tuning/pretraining of CLM models (60M - 1.1B)
  - [ ] Parameter-efficient fine-tuning of larger LLMs (e.g., Llama-3.1-8B)


## What's New

- 🚧 Fine-tuning/pretraining of alternative compute paradigms on CLMs.

- 🚩**1th Oct, 2025 Milestone**: Fine-tuning/pretraining of alternative compute paradigms on Roberta

    | Item | Description |
    | ---- | ----------- |
    | Optical Transformer | [Tutorial](02-model-behaviour-level-simulation/roberta-onn.md) |
    | **CompleteThis** |  |

- 🚩 **9th, Jun, 2025 Milestone**: Our Software-emulation & acceleration backend, Mase-triton, is released on [PyPI](https://pypi.org/project/mase-triton/). Try it via `pip install mase-triton`.
    - For more details, please refer to [Intro to Mase-triton](/docs/02-model-behaviour-level-simulation/mase-triton.md) and [Mase-triton GitHub](https://github.com/DeepWok/mase-triton/tree/master)

- 🚩 **15th April, 2025 Milestone**: System and model-level training simulation (Small Language Models).

    | Item | Description |
    | ---- | ----------- |
    | Environment setup | [Tutorial](env-setup.md) |
    | Pretraining AICrossSim LLMs (60M, 200M, 400M, 1.1B) & evaluation | [Tutorial](01-model-training/llm-pretrain-and-eval.md) |
    | Software-emulated bitflip-aware pretraining & evaluation | [Tutorial](02-model-behaviour-level-simulation/llm-bitflip.md) |

## Roadmap

- Model Training & Evaluation
    - LLMs
        - [x] Pretraining of LLMs (60M, 200M, 400M, 1.1B) using the Llama-3 architecture.
        - [x] `lm-eval-harness` evaluation of LLMs.
        - [x] Parameter-efficient fine-tuning
        - [ ] Supervised fine-tuning
- Model Behavior-Level Simulation
    - [x] Post-training bitflip transform & bitflip-aware pretraining
    - [ ] Optical compute
        - [x] Roberta
        - [ ] CLM
    - [ ] Spiking neural networks
    - [ ] In-memory compute

## About the Project

This project is led by [Dr. Yiren Zhao](https://aaron-zhao123.github.io/) at Imperial College London, [Dr. Luo Mai](https://luomai.github.io/) at University of Edinburgh, [Prof. Robert Mullins](https://www.cl.cam.ac.uk/~rdm34/) at University of Cambridge, and funded by [Advanced Research + Invention Agency (ARIA)](https://www.aria.org.uk/).