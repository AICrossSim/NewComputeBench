Transform-Aware LLM Training Guidelines
=========================================

This guide explains how to implement and integrate a new compute paradigm simulation
into NewComputeBench. Both continual pretraining (starting from a pretrained checkpoint)
and pretraining from scratch are supported.

.. note::

   If the transform is highly lossy, pretraining from scratch may be ineffective —
   the model may fail to learn at all. Continual pretraining is recommended in
   those cases.


Continual Pretraining
----------------------

Continual pretraining adapts a pretrained HuggingFace checkpoint by injecting the
hardware simulation transform before training begins.

The Optical Transformer is used as a worked example below. The same pattern applies
to any new compute paradigm.

Step 1 — Implement the Simulated Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Place your implementation in ``src/aixsim_models/<paradigm>/``.
For the Optical Transformer, the key components are in
`src/aixsim_models/optical_compute/optical_transformer <https://github.com/AICrossSim/NewComputeBench/blob/master/src/aixsim_models/optical_compute/optical_transformer>`_:

**Simulated linear layer and matmul:**

- ``OpticalTransformerLinear`` (from `mase-triton <https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/optical_compute/layers.py>`_) — replaces all ``nn.Linear`` layers except ``lm_head``.
- ``HFOpticalTransformerLlamaAttention`` (in `layers.py <https://github.com/AICrossSim/NewComputeBench/blob/master/src/aixsim_models/optical_compute/optical_transformer/layers.py>`_) — wraps the attention module to intercept Q-K and Attention-V matmuls.

.. note::

   **When to use Triton vs. PyTorch built-ins:**
   The Optical Transformer uses Triton kernels (``ot_qlinear_fn``, ``ot_qmatmul_fn``)
   primarily to save GPU memory. If your simulation can be memory-efficiently implemented
   using PyTorch built-in operations, there is no need to write a Triton kernel.

   Also note: HuggingFace Trainer does not work with autotuned Triton kernels.
   Autotuning is therefore disabled in mase-triton by default.

Step 2 — Implement the Transform Pass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement a ``transform_hf_model`` function that iterates over the model's modules and
replaces them with your simulated equivalents.
See `transform.py <https://github.com/AICrossSim/NewComputeBench/blob/master/src/aixsim_models/optical_compute/optical_transformer/transform.py>`_
for the Optical Transformer example.

The typical pattern uses two loops:

1. Replace attention modules with your custom attention class (to intercept matmuls).
2. Replace remaining ``nn.Linear`` modules with your simulated linear layer.

A ``TransformConfigManager`` reads the YAML config and maps layer names to their
transform configurations.

Step 3 — Write a Transform Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a YAML file specifying the transform parameters layer-by-layer.
See ``experiments/llm-optical-transformer/continual_pretraining/configs/default.yaml``
for an example.

Step 4 — Adapt the Training Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy HuggingFace's
`run_clm.py <https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py>`_
and insert the following snippet just before training starts:

.. code-block:: python

   if model_args.transform_config is not None:
       with open(model_args.transform_config, "r") as f:
           transform_args = yaml.safe_load(f)
       config_manager = TransformConfigManager(**transform_args)
       transformed_layers = transform_hf_model(model, config_manager)
       transform_histogram = make_transform_histogram(transformed_layers=transformed_layers)
       logger.info(f"Transformed layers:\n{transform_histogram}")
   else:
       logger.info("No transform config provided. Using the original model.")

Replace ``transform_hf_model`` with your own transform pass.
The adapted script for the Optical Transformer is at
`experiments/llm-optical-transformer/continual_pretraining/run_clm.py <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-optical-transformer/continual_pretraining/run_clm.py>`_.

Step 5 — Launch Training
~~~~~~~~~~~~~~~~~~~~~~~~~

Example using the Optical Transformer:

.. code-block:: bash

   python run_clm.py \
       --model_name_or_path AICrossSim/clm-60m \
       --dataset_name HuggingFaceFW/fineweb-edu \
       --dataset_config_name "sample-10BT" \
       --per_device_train_batch_size 12 \
       --per_device_eval_batch_size 12 \
       --gradient_accumulation_steps 50 \
       --do_train \
       --report_to "wandb" \
       --learning_rate 5e-5 \
       --max_steps 100 \
       --save_strategy "steps" \
       --save_steps 500 \
       --save_total_limit 2 \
       --bf16 \
       --dataloader_num_workers 16 \
       --preprocessing_num_workers 32 \
       --tokenizer_name HuggingFaceTB/cosmo2-tokenizer \
       --output_dir ./output/test-clm-trainer \
       --transform_config ./configs/default.yaml \
       --seed 42


Pretraining from Scratch
-------------------------

We use `torchtitan <https://github.com/pytorch/torchtitan>`_ as the distributed training
backend for transform-aware pretraining from scratch.

See `experiments/llm-optical-transformer/pretrain/run.py <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-optical-transformer/pretrain/run.py>`_
for the Optical Transformer example.

The interface is identical to the standard CLM pretraining script
(see :doc:`../tutorials/pretraining/llm_pretrain_eval`), with an additional
transform config argument passed to ``generate-cfg``.
