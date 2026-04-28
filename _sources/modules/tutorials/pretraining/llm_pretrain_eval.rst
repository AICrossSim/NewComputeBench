LLM Pretraining & Evaluation
============================

This tutorial covers pretraining ``AICrossSim-CLM`` models and evaluating them on
language modeling benchmarks.

.. note::

   If you have not set up the environment yet, follow :doc:`../../getting_started/installation` first.

Overview
--------

- We pretrain ``AICrossSim-CLM`` (60M, 200M, 400M, 1.1B) on the
  `FineWeb-Edu <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>`_ dataset.
- We follow the `Chinchilla scaling law <https://arxiv.org/abs/2203.15556>`_ to determine
  the number of training tokens: ``num_tokens = 22 × num_params``.
- The entry point is ``experiments/llm-digital/pretrain/run.py``.

  - Run ``python run.py -h`` to see all subcommands.
  - Run ``python run.py <subcommand> -h`` for subcommand-specific help.

- We use ``torchrun`` for distributed training.
- Pretrained checkpoints are available on HuggingFace:
  `NewComputeBench-CLM-Digital <https://huggingface.co/collections/AICrossSim/newcomputebench-clm-digital-67d19e95ebacdbc3e5752be3>`_.


Pretraining
-----------

The workflow is the same for all model sizes: generate a config, then launch training.
We demonstrate with ``AICrossSim-CLM-60M``.

AICrossSim-CLM-60M
~~~~~~~~~~~~~~~~~~

1. Change to the pretraining directory and activate the environment:

   .. code-block:: bash

      cd experiments/llm-digital/pretrain

   **uv:**

   .. code-block:: bash

      source .venv/bin/activate

   **conda:**

   .. code-block:: bash

      conda activate new-compute

2. Generate the training config:

   .. admonition:: Fast development run

      Use these flags to reduce memory usage and shorten training for quick tests:

      - ``--batch_size`` — smaller batch size to avoid OOM.
      - ``--data_parallel_replicate_degree`` — number of data-parallel replicas (typically equal to the number of GPUs).
      - ``--data_parallel_shard_degree`` — shard model parameters across GPUs (FSDP). Default ``-1`` disables sharding.
      - ``--token_num_scale`` — controls training length via ``num_tokens = scale × num_params``. Set to ``1`` for a short run.

   .. code-block:: bash

      data_parallel="2"
      batch_size="48"
      token_num_scale="22"

      python run.py generate-cfg \
          --model_flavor 60M \
          --batch_size ${batch_size} \
          --data_parallel_replicate_degree ${data_parallel} \
          --compile true \
          --save_path ./configs/tutorial-60M.yaml

   This generates ``configs/tutorial-60M.yaml`` for pretraining on a FineWeb-Edu subset
   of ``22 × 60M`` tokens with per-device batch size 48 and 2-GPU data parallelism.
   The ``--compile`` flag enables ``torch.compile`` for faster training.

3. Launch pretraining:

   .. code-block:: bash

      num_gpus="2"
      cuda_devices="1,2"   # GPU indices to use, e.g. "0,1" for the first two GPUs

      CUDA_VISIBLE_DEVICES=${cuda_devices} \
      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" STREAM_HF_DATA="1" \
      torchrun --nproc_per_node=${num_gpus} --rdzv_backend c10d \
          --rdzv_endpoint="localhost:0" --local-ranks-filter 0 \
          --role rank --tee 3 \
          run.py pretrain --config configs/tutorial-60M.yaml \
          --metrics_args.enable_wandb false

   - ``STREAM_HF_DATA=1`` streams the FineWeb-Edu dataset instead of downloading it.
   - Checkpoints are saved to ``./outputs/checkpoints/aixsim-60M/<timestamp>/``.
   - Disable W&B logging with ``--metrics_args.enable_wandb false`` if you have not run
     ``wandb login``.

   .. admonition:: Troubleshooting: Fatal Python error: Aborted
      :class: warning

      After training finishes, ``torchrun`` may raise ``Fatal Python error: Aborted``
      when destroying the process group. **This does not affect the training results**
      as long as the error appears after the final checkpoint is saved — look for a log
      line similar to::

         [rank0]: Finished saving the checkpoint ... in 5.53 seconds.
         [rank0]: Training completed

4. (Optional) Convert the checkpoint to HuggingFace format:

   .. admonition:: Why convert?

      The training code uses custom distributed model classes. Converting to HuggingFace
      format lets you use the full HuggingFace ecosystem (generation, evaluation, etc.).

   .. code-block:: bash

      python run.py convert-ckpt aixsim 60M \
          ./outputs/checkpoints/aixsim-60M/<timestamp>/<step-xxx> \
          path/to/huggingface/checkpoint

.. tip::

   **Our 60M results** — pretrained on 2 × H100 96 GB for 1 hour.

   - Config: `experiments/llm-digital/pretrain/configs/aixsim-60M.yaml <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-digital/pretrain/configs/aixsim-60M.yaml>`_
   - W&B logs: `link <https://wandb.ai/cz98/torchtitan/runs/7kttp3qt>`_
   - HuggingFace checkpoint: `AICrossSim/clm-60m <https://huggingface.co/AICrossSim/clm-60m>`_

AICrossSim-CLM-200M
~~~~~~~~~~~~~~~~~~~~

The 200M model uses Fully Sharded Data Parallel (FSDP) to reduce per-GPU memory at the
cost of slightly longer training.

.. code-block:: bash

   batch_size="32"
   data_parallel_replicate="1"
   data_parallel_shard="2"

   python run.py generate-cfg \
       --model_flavor 200M \
       --batch_size ${batch_size} \
       --data_parallel_replicate_degree ${data_parallel_replicate} \
       --data_parallel_shard_degree ${data_parallel_shard} \
       --compile true \
       --save_path ./configs/tutorial-200M.yaml

   num_gpus="2"

   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" STREAM_HF_DATA="1" \
   torchrun --nproc_per_node=${num_gpus} --rdzv_backend c10d \
       --rdzv_endpoint="localhost:0" --local-ranks-filter 0 \
       --role rank --tee 3 \
       run.py pretrain --config configs/tutorial-200M.yaml \
       --metrics_args.enable_wandb false

.. tip::

   **Our 200M results** — pretrained on 2 × H100 96 GB for 6.5 hours.

   - W&B logs: `link <https://wandb.ai/cz98/torchtitan/runs/uhnlw6k8/overview>`_
   - HuggingFace checkpoint: `AICrossSim/clm-200m <https://huggingface.co/AICrossSim/clm-200m>`_

AICrossSim-CLM-400M
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   batch_size="12"
   data_parallel_replicate="1"
   data_parallel_shard="8"

   python run.py generate-cfg \
       --model_flavor 400M \
       --batch_size ${batch_size} \
       --data_parallel_replicate_degree ${data_parallel_replicate} \
       --data_parallel_shard_degree ${data_parallel_shard} \
       --compile true \
       --save_path ./configs/tutorial-400M.yaml

   num_gpus="8"

   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" STREAM_HF_DATA="1" \
   torchrun --nproc_per_node=${num_gpus} --rdzv_backend c10d \
       --rdzv_endpoint="localhost:0" --local-ranks-filter 0 \
       --role rank --tee 3 \
       run.py pretrain --config configs/tutorial-400M.yaml \
       --metrics_args.enable_wandb false

.. tip::

   **Our 400M results** — pretrained on 8 × A6000 for 21 hours.

   - W&B logs: `link <https://wandb.ai/cz98/torchtitan/runs/cic7m3cx/overview>`_
   - HuggingFace checkpoint: `AICrossSim/clm-400m <https://huggingface.co/AICrossSim/clm-400m>`_

AICrossSim-CLM-1.1B
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   batch_size="24"
   data_parallel_replicate="1"
   data_parallel_shard="8"

   python run.py generate-cfg \
       --model_flavor 1.1B \
       --batch_size ${batch_size} \
       --data_parallel_replicate_degree ${data_parallel_replicate} \
       --data_parallel_shard_degree ${data_parallel_shard} \
       --compile true \
       --save_path ./configs/tutorial-1.1B.yaml

   num_gpus="8"

   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" STREAM_HF_DATA="1" \
   torchrun --nproc_per_node=${num_gpus} --rdzv_backend c10d \
       --rdzv_endpoint="localhost:0" --local-ranks-filter 0 \
       --role rank --tee 3 \
       run.py pretrain --config configs/tutorial-1.1B.yaml \
       --metrics_args.enable_wandb false

.. tip::

   **Our 1.1B results** — pretrained on 8 × H100 96 GB for 33 hours.

   - W&B logs: `link <https://wandb.ai/cz98/torchtitan/runs/8mcf8ay1/overview>`_
   - HuggingFace checkpoint: `AICrossSim/clm-1.1b <https://huggingface.co/AICrossSim/clm-1.1b>`_
   - Raw torchrun checkpoints (for resuming): `AICrossSim/clm-1.1b-torch-ckpt <https://huggingface.co/AICrossSim/clm-1.1b-torch-ckpt>`_


Evaluation
----------

Pretraining dataset perplexity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate a checkpoint on the pretraining dataset:

.. code-block:: bash

   # torchrun checkpoint
   python run.py eval pt-ppl \
       aixsim 60M \
       ./outputs/checkpoints/aixsim-60M/<timestamp>/<step-xxx>

   # HuggingFace checkpoint
   python run.py eval hf-ppl \
       AICrossSim/clm-60m

Downstream tasks (lm-eval-harness)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We integrate `lm-eval-harness <https://github.com/EleutherAI/lm-evaluation-harness>`_
for downstream evaluation:

.. code-block:: bash

   model_name="AICrossSim/clm-60m"

   python run.py eval hf-lm-eval \
       ${model_name} \
       --tasks ['wikitext'] \
       --dtype float16

Run ``python run.py hf-lm-eval -h`` for all available arguments.

.. note::

   Under the hood ``hf-lm-eval`` calls ``lm-eval-harness``'s ``simple_evaluate``.
   Key arguments:

   - ``--tasks`` — list of task names (same naming as lm-eval-harness).
   - ``--num_fewshot`` — few-shot count; ``None`` uses the task default.
   - ``--limit`` — if > 1, maximum number of examples; if ≤ 1, fraction of the dataset.

Simple text generation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   prompt="London is"

   python run.py hf-gen \
       --model_name AICrossSim/clm-60m \
       --prompt "${prompt}" \
       --max_new_tokens 100 \
       --do_sample true \
       --temperature 0.6 \
       --top_k 50 \
       --top_p 0.9
