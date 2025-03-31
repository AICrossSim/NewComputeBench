# LLM Pretraining

This document provides a tutorial on how to pretrain [AICrossSim-CLM](../model-list.md) using NewComputeBench.

## Overview

- We aim to pretrain AICrossSim-CLM models (60M, 200M, 400M, 1.1B) on the [Fineweb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset.
    - We followed the [Chinchilla scaling law](https://arxiv.org/abs/2203.15556) to determine the number of training tokens: `num_tokens = 22 * num_params`.
    - As the model size increases, the training time and memory requirements will increase significantly. For example, we pretrained the 1.1B model on 8 NVIDIA H100 80GB GPUs for 3 days, while the 60M model can be pretrained on 2 NVIDIA H100 80GB GPUs within 1 hour.
- The pretraining entrypoint is at `experiments/llm-digital/pretrain/run.py`
    - `run.py` supports multiple subcommands, including `pretrain`, `eval`, `generate-hf`, `convert-ckpt`, and `generate-cfg`.
        - Run `python run.py -h` to see the available subcommands.
        - Run `python run.py <subcommand> -h` to see the help message for a specific subcommand.
    - To run distributed training, we use `torchrun` to launch the training script.

## Pretraining

!!! info "Environment Setup?"

    If you have not set up environments, please follow the guidelines in [Environment Setup](../env-setup.md).

### AICrossSim-CLM-60M

1. Change the working directory to `experiments/llm-digital/pretrain` and activate the conda environment.

    ```bash
    cd experiments/llm-digital/pretrain
    conda activate new-compute
    ```

2. Generate PreTraining Config

    ```bash
    python run.py generate-cfg \
        --model_flavor 60M --batch_size 48 \
        --data_parallel_replicate_degree 2 \
        --compile true \
        --save_path ./configs/tutorial-60M.yaml
    ```


