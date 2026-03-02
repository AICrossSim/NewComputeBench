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
    - [x] Random bitflip
    - [x] Optical compute
    - [ ] Spiking neural networks
    - [ ] In-memory compute
  - [ ] Parameter-efficient fine-tuning of larger LLMs (e.g., Llama-3.1-8B)
    - [x] Random bitflip (promising results)
    - [x] Optical compute (failed to converge)


## What's New

- **4th, Feb, 2026 Milestone**: We have successfully fine-tuned Llama-3.1-8B with random bitflip noise injected in forward passes, and observed promising results that the LoRA adapters with only 1.2% trainable parameters can effectively mitigate the effect of noise (reducing perplexity from 1008.95 to 11.01, with the original clean perplexity at 7.91).

    | Item | Description |
    | ---- | ----------- |
    | Llama-3.1-8B with random bitflip noise | [Tutorial](./02-model-behaviour-level-simulation/clm-bitflip-lora-finetune.md)

- **4th Oct, 2025 Milestone**: Fine-tuning/pretraining of alternative compute paradigms on CLMs.

    | Item | Description |
    | ---- | ----------- |
    | Optical Transformer | [Tutorial](02-model-behaviour-level-simulation/clm-onn.md) |

- 🚩**1th Oct, 2025 Milestone**: Fine-tuning/pretraining of alternative compute paradigms on Roberta

    | Item | Description |
    | ---- | ----------- |
    | Optical Transformer | [Tutorial](02-model-behaviour-level-simulation/roberta-onn.md) |
    | **CompleteThis** |  |

- 🚩 **9th, Jun, 2025 Milestone**: Our Software-emulation & acceleration backend, Mase-triton, is released on [PyPI](https://pypi.org/project/mase-triton/). Try it via `pip install mase-triton`.
    - For more details, please refer to [Intro to Mase-triton](02-model-behaviour-level-simulation/mase-triton.md) and [Mase-triton GitHub](https://github.com/DeepWok/mase-triton/tree/master)

- 🚩 **15th April, 2025 Milestone**: System and model-level training simulation (Small Language Models).

    | Item | Description |
    | ---- | ----------- |
    | Environment setup | [Tutorial](env-setup.md) |
    | Pretraining AICrossSim LLMs (60M, 200M, 400M, 1.1B) & evaluation | [Tutorial](01-model-training/llm-pretrain-and-eval.md) |
    | Software-emulated bitflip-aware pretraining & evaluation | [Tutorial](02-model-behaviour-level-simulation/clm-bitflip.md) |

## Roadmap

- Model Training & Evaluation
    - LLMs
        - [x] Pretraining of LLMs (60M, 200M, 400M, 1.1B) using the Llama-3 architecture.
        - [x] `lm-eval-harness` evaluation of LLMs.
        - [x] Parameter-efficient fine-tuning
- Model Behavior-Level Simulation
    - [x] Lossy Communication (random bitflip)
        - [x] Post-training bitflip transform
        - [x] Bitflip-aware pretraining (60M - 1.1B)
        - [x] Bitflip-aware parameter-efficient fine-tuning (Llama-3.1-8B)
    - [x] Optical compute
        - [x] Roberta Fine-tuning (125M)
        - [x] CLM full fine-tuning (60M - 1.1B)
        - [x] CLM parameter efficient fine-tuning (60M - 1.1B)
    - [ ] Spiking neural networks
    - [ ] In-memory compute

## About the Project

This project is led by [Dr. Yiren Zhao](https://aaron-zhao123.github.io/) at Imperial College London, [Dr. Luo Mai](https://luomai.github.io/) at University of Edinburgh, [Prof. Robert Mullins](https://www.cl.cam.ac.uk/~rdm34/) at University of Cambridge, and funded by [Advanced Research + Invention Agency (ARIA)](https://www.aria.org.uk/).