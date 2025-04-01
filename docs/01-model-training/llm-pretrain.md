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
- We uploaded the pretrained models to HuggingFace for easy access: [NewComputeBench-CLM-Digital](https://huggingface.co/collections/AICrossSim/newcomputebench-clm-digital-67d19e95ebacdbc3e5752be3)

## Pretraining

!!! info "Environment Setup?"

    If you have not set up environments, please follow the guidelines in [Environment Setup](../env-setup.md).

### AICrossSim-CLM-60M

1. Change the working directory to `experiments/llm-digital/pretrain` and activate the conda environment.

    ```bash
    cd experiments/llm-digital/pretrain
    conda activate new-compute
    ```

2. Generate pretraining config

    !!! info "Fast Development Run?"

        `generate-cfg` has several default arguments. You may want to change them for a fast development run:

        - `--batch_size`: a smaller batch size to avoid out-of-memory errors.
        - `--data_parallel_replicate_degree`: set this to the number of GPUs.
        - `--token_num_scale`: the scale used to determine the number of training tokens: `num_tokens = token_num_scale * num_params`, 22 by default. Set this to a small value like `1` to reduce the number of training steps.

    ```bash
    data_parallel="2"       # Number of GPUs per node
    batch_size="48"         # Per-device batch size
    token_num_scale="22"    # Scale for number of training tokens
    python run.py generate-cfg \
        --model_flavor 60M --batch_size ${batch_size} \
        --data_parallel_replicate_degree ${data_parallel} \
        --compile true \
        --save_path ./configs/tutorial-60M.yaml
    ```

    - This will generate a training configuration file `configs/tutorial-60M.yaml` for pretraining `AICrossSim-CLM-60M` model using a per-device batch size of 48 and a data parallel replicate degree of 2 on a FineWeb-Edu subset of `22 * 60M` tokens.
    - Subcommand `generate-cfg` automatically calculates the number of training steps.
    - The `--compile` flag enables the use of `torch.compile` for optimizing the training process.

3. Launch pretraining

    ```bash
    num_gpus="2" # Number of GPUs per node. We only use one node for this example
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" STREAM_HF_DATA="1" \
    torchrun --nproc_per_node=${num_gpus} --rdzv_backend c10d \
        --rdzv_endpoint="localhost:0" --local-ranks-filter 0 \
        --role rank --tee 3 \
        run.py pretrain --config configs/tutorial-60M.yaml \
        --metrics_args.enable_wandb false
    ```

    - This will pass the generated configuration file and launch the pretraining job on a single node of 2 GPUs using `torchrun`.
    - The `--metrics_args.enable_wandb` flag disables Weights and Biases logging. You can enable it by setting it to `true`.
    - The `STREAM_HF_DATA` environment variable is set to `1` to enable streaming data loading from Hugging Face datasets instead of downloading the huge dataset to the local disk.
    - When the training is finished, the model checkpoint will be saved at `./outputs/checkpoints/aixsim-60M/<timestamp>`.

    ??? failure "Fatal Python error: Aborted ?"

        We noticed that after the training is finished, `torchrun` may raise the error "Fatal Python error: Aborted" when destroying process group. **This does not affect the training results as long as the error is raised after the final checkpoint is saved** (messages like "[rank0]:2025-04-01 00:25:59,616 - root - INFO - Finished saving the checkpoint (or staging if async is enabled)in 5.53 seconds.")

        Here is an example of the error message:

        ```text
        [rank0]:2025-04-01 00:25:47,084 - root - INFO - step: 640/644 = 99.3789%  loss:  6.2116  memory: 87.04GiB(93.48%)  tps: 52,142  mfu: 3.09%
        [rank0]:2025-04-01 00:25:54,090 - root - INFO - Saving a full checkpoint at last step, step 644.
        [rank0]:2025-04-01 00:25:59,616 - root - INFO - Finished saving the checkpoint (or staging if async is enabled)in 5.53 seconds.
        [rank0]:2025-04-01 00:25:59,617 - root - INFO - Sleeping 2 seconds for other ranks to complete
        [rank0]:2025-04-01 00:26:01,706 - root - INFO - Training completed
        [rank0]:terminate called without an active exception
        [rank0]:Fatal Python error: Aborted
        [rank0]:
        [rank0]:Current thread 0x00007f0360ff9640 (most recent call first):
        [rank0]:  Garbage-collecting
        [rank0]:  <no Python frame>
        [rank0]:
        [rank0]:Thread 0x00007f06b6639200 (most recent call first):
        [rank0]:  <no Python frame>
        W0401 00:26:06.901000 3016633 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3016738 closing signal SIGTERM
        E0401 00:26:08.212000 3016633 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: -6) local_rank: 1 (pid: 3016739) of binary: /home/zz7522/miniconda3/envs/new-compute/bin/python3.11
        Traceback (most recent call last):
        File "/home/zz7522/miniconda3/envs/new-compute/bin/torchrun", line 8, in <module>
            sys.exit(main())
                    ^^^^^^
        File "/home/zz7522/miniconda3/envs/new-compute/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
            return f(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^
        File "/home/zz7522/miniconda3/envs/new-compute/lib/python3.11/site-packages/torch/distributed/run.py", line 918, in main
            run(args)
        File "/home/zz7522/miniconda3/envs/new-compute/lib/python3.11/site-packages/torch/distributed/run.py", line 909, in run
            elastic_launch(
        File "/home/zz7522/miniconda3/envs/new-compute/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
            return launch_agent(self._config, self._entrypoint, list(args))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/home/zz7522/miniconda3/envs/new-compute/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
            raise ChildFailedError(
        torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
        ========================================================
        run.py FAILED
        --------------------------------------------------------
        Failures:
        <NO_OTHER_FAILURES>
        --------------------------------------------------------
        Root Cause (first observed failure):
        [0]:
        time      : 2025-04-01_00:26:06
        host      : ee-tiamat.ee.ic.ac.uk
        rank      : 1 (local_rank: 1)
        exitcode  : -6 (pid: 3016739)
        error_file: <N/A>
        traceback : Signal 6 (SIGABRT) received by PID 3016739
        ========================================================
        ```

4. (Optional) Convert to HuggingFace checkpoint

    !!! info "HuggingFace Checkpoint"

        To support distributed training, the training code defines custom model classes, and the checkpoints are saved in a custom format by `torchrun`.
        To exploit the HuggingFace ecosystem, we provide a script to convert the custom checkpoint to HuggingFace format.

    ```bash
    python run.py convert-ckpt aixsim 60M \
        ./outputs/checkpoints/aixsim-60M/<timestamp>/<step-xxx> \
        path/to/huggingface/checkpoint
    ```

!!! success "Our Pretraining Results"

    We pretrained the `AICrossSim-CLM-60M` model on 2 NVIDIA H100 96GB GPUs for 1 hour.

    - Training config: [`experiments/llm-digital/pretrain/configs/aixsim-60M.yaml`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-digital/pretrain/configs/aixsim-60M.yaml)
    - Wandb logs: [link](https://wandb.ai/cz98/torchtitan/runs/7kttp3qt?nw=nwusercz98)
    - HuggingFace checkpoint: [AICrossSim/clm-60m](https://huggingface.co/AICrossSim/clm-60m)

### AICrossSim-CLM-200M

### AICrossSim-CLM-400M

### AICrossSim-CLM-600M