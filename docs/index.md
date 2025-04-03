# AICrossSim/NewComputeBench

<figure markdown="span">
  ![NewComputeBench](./images/logo.png){ width="200" }
  <figcaption>NewComputeBench</figcaption>
</figure>

**[AICrossSim/NewComputeBench](https://github.com/AICrossSim/NewComputeBench)** is a benchmark suite for new compute paradigms (Spiking neural networks, Optical computation, In-Memory computation, etc) via software emulation. We aim to predict the scaling law of neural networks trained with new compute paradigms by running small & medium scale experiments and extrapolate the trends we observed. NewComputeBench project mainly consists of three parts:

- Model Training
- Model Behavior-Level Simulation
- Hardware-Performance Simulation (`🚧 TODO`)


## What's New

- 🚩 **15th April Milestone**: System and model-level training simulation (Small Language Models).

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
        - [ ] Parameter-efficient fine-tuning
        - [ ] Supervised fine-tuning
- Model Behavior-Level Simulation
    - [x] Post-training bitflip transform & bitflip-aware pretraining
    - [ ] Optical compute
    - [ ] Spiking neural networks
    - [ ] In-memory compute
- Hardware-Performance Simulation
    - [ ] Hardware performance prediction
    - `🚧 TODO`

## About the Project

This project is led by [Dr. Yiren Zhao](https://aaron-zhao123.github.io/) at Imperial College London, [Dr. Luo Mai](https://luomai.github.io/) at University of Edinburgh, [Prof. Robert Mullins](https://www.cl.cam.ac.uk/~rdm34/) at University of Cambridge, and funded by [Advanced Research + Invention Agency (ARIA)](https://www.aria.org.uk/).