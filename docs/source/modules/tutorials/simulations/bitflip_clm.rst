Random Bitflip on CLM
=====================

This tutorial covers two workflows:

1. **Post-training bitflip evaluation** — load a pretrained checkpoint, inject random
   bitflips, and evaluate the transformed model.
2. **Bitflip-aware pretraining** — pretrain a model from scratch with bitflip noise
   injected during every forward pass.

.. note::

   If you have not set up the environment yet, follow :doc:`../../getting_started/installation` first.


Overview
--------

- **Post-training evaluation** entry point:
  `experiments/llm-bitflip/transform/minimal.py <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/transform/minimal.py>`_
- **Bitflip-aware pretraining** entry point:
  `experiments/llm-bitflip/pretrain/run.py <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/run.py>`_
- Random bitflip kernels are implemented in
  `mase-triton <https://pypi.org/project/mase-triton/>`_.
  The core function is ``mase_triton.random_bitflip.core.random_bitflip_fn``, which
  supports independent bitflip probabilities for sign-exponent bits and mantissa bits,
  and can zero out outliers / NaN values via a threshold.

.. note::

   The bitflip probability must be a power of 0.5 (e.g., ``0.5``, ``0.5²``, ``0.5³``, …).
   The kernel snaps to the nearest valid value automatically.
   The minimum supported probability is ``0.5²⁴ ≈ 5.96 × 10⁻⁸`` due to the Philox
   pseudo-random number generator used internally.


Post-Training Bitflip Evaluation
---------------------------------

We provide minimal scripts that apply a bitflip transform to all linear layers
(>90% of FLOPs in Transformers) in a HuggingFace model, then evaluate with
``lm-eval-harness``.

Transform and evaluate
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd experiments/llm-bitflip/transform

   model_name="unsloth/Meta-Llama-3.1-8B-Instruct"
   x_p_exp=null
   w_p_exp=null
   x_zero_out_t="100"
   w_zero_out_t="1.25"
   x_p_frac=$(bc <<< "scale=10; 0.5^10")
   w_p_frac=$(bc <<< "scale=10; 0.5^10")

   python minimal.py eval-bitflip \
       --model_name ${model_name} \
       --bitflip_config "default" \
       --default_bitflip_config.x_p_exp=${x_p_exp} \
       --default_bitflip_config.x_p_frac=${x_p_frac} \
       --default_bitflip_config.x_zero_out_t=${x_zero_out_t} \
       --default_bitflip_config.w_p_exp=${w_p_exp} \
       --default_bitflip_config.w_p_frac=${w_p_frac} \
       --default_bitflip_config.w_zero_out_t=${w_zero_out_t} \
       --tasks ['wikitext']

.. note::

   ``eval-bitflip`` uses ``lm-eval-harness``'s ``simple_evaluate``.
   See the evaluation section of :doc:`../pretraining/llm_pretrain_eval` for argument details.

Evaluate the original model (clean baseline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python minimal.py eval-ori \
       --model_name ${model_name} \
       --tasks ['wikitext']

Text generation with bitflip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   prompt="London is"
   max_new_tokens="100"

   python minimal.py hf-gen \
       AICrossSim/clm-60m \
       ${prompt} \
       --max_new_tokens ${max_new_tokens} \
       --do_sample true \
       --temperature 0.6 \
       --top_k 50 \
       --top_p 0.9 \
       --bitflip_config "default" \
       --default_bitflip_config.x_p_exp=${x_p_exp} \
       --default_bitflip_config.x_p_frac=${x_p_frac} \
       --default_bitflip_config.x_zero_out_t=${x_zero_out_t} \
       --default_bitflip_config.w_p_exp=${w_p_exp} \
       --default_bitflip_config.w_p_frac=${w_p_frac} \
       --default_bitflip_config.w_zero_out_t=${w_zero_out_t}

.. tip::

   We swept ``x_p_frac`` and ``w_p_frac`` on ``AICrossSim/clm-1.1b`` and observed
   that when perplexity increases by only ~1%, generated text remains coherent with
   the clean model.
   Sample outputs: `Google Sheets <https://docs.google.com/spreadsheets/d/1N9_i3_YzKhDfI6H0EWO86zVMxiwsHcSUSbll2ws4zRA/edit?usp=sharing>`_


Bitflip-Aware Pretraining
--------------------------

The script ``experiments/llm-bitflip/pretrain/run.py`` extends the standard CLM
pretraining script (see :doc:`../pretraining/llm_pretrain_eval`) with an additional
argument for the bitflip transform configuration.

We demonstrate with ``AICrossSim-CLM-60M`` on 2 × H100 96 GB.

1. Generate a config with bitflip settings:

   .. code-block:: bash

      cd experiments/llm-bitflip/pretrain

      bitflip_transform_config="./configs/meta/fc-only-w-a-exp-frac.yaml"

      python run.py generate-cfg \
          ${bitflip_transform_config} \
          --model_arch "aixsim" \
          --model_flavor "60M" \
          --batch_size 48 \
          --data_parallel_replicate_degree 2 \
          --data_parallel_shard_degree -1 \
          --token_num_scale 22 \
          --compile "false" \
          --save_path "./configs/tutorial-60m.yaml"

2. Launch pretraining:

   .. code-block:: bash

      num_gpus="2"

      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" STREAM_HF_DATA="1" \
      torchrun --nproc_per_node=${num_gpus} --rdzv_backend c10d \
          --rdzv_endpoint="localhost:0" --local-ranks-filter 0 \
          --role rank --tee 3 \
          run.py pretrain \
          --config configs/tutorial-60m.yaml \
          --metrics_args.enable_wandb false

3. Convert the checkpoint to HuggingFace format:

   .. code-block:: bash

      python run.py convert-ckpt pt2hf \
          "aixsim" "60M" \
          path/to/torchrun/checkpoint \
          path/to/output/dir


Results Summary
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 20 15 25 15 35

   * - Model
     - Environment
     - Training time
     - Config
     - W&B
     - HuggingFace checkpoint
   * - 60M
     - 2× H100 96 GB
     - 2.5 hours
     - `configs/aixsim-60M.yaml <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/configs/aixsim-60M.yaml>`_
     - `link <https://wandb.ai/cz98/torchtitan/runs/bbyruxxh/overview>`_
     - `AICrossSim/bitflip-fc-clm-60m <https://huggingface.co/AICrossSim/bitflip-fc-clm-60m>`_
   * - 200M
     - 2× H100 96 GB
     - 14.3 hours
     - `configs/aixsim-200M.yaml <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/configs/aixsim-200M.yaml>`_
     - `link <https://wandb.ai/cz98/torchtitan/runs/iivbk9nr/overview>`_
     - `AICrossSim/bitflip-fc-clm-200m <https://huggingface.co/AICrossSim/bitflip-fc-clm-200m>`_
   * - 400M
     - 6× A6000 48 GB
     - 33 hours
     - `configs/aixsim-400M.yaml <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/configs/aixsim-400M.yaml>`_
     - `link <https://wandb.ai/cz98/torchtitan/runs/6mnsbo7e/overview>`_
     - `AICrossSim/bitflip-fc-clm-400m <https://huggingface.co/AICrossSim/bitflip-fc-clm-400m>`_
   * - 1.1B
     - 8× H200 141 GB
     - 51 hours
     - `configs/aixsim-1.1B.yaml <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/configs/aixsim-1.1B.yaml>`_
     - `link <https://wandb.ai/cz98/torchtitan/runs/5tbo5tkg>`_
     - `AICrossSim/bitflip-fc-clm-1.1b <https://huggingface.co/AICrossSim/bitflip-fc-clm-1.1b>`_
