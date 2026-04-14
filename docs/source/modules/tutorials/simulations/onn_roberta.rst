Optical Neural Networks on RoBERTa
====================================

This tutorial demonstrates how to apply optical transformer modifications to RoBERTa
for sequence classification. The implementation simulates photonic computing with
quantization-aware attention mechanisms and linear layers.

.. note::

   If you have not set up the environment yet, follow :doc:`../../getting_started/installation` first.


Overview
--------

- **Optical Transform** — replaces standard attention and linear layers with optical
  transformer equivalents in a RoBERTa model.

  - Entry point: `experiments/roberta-optical-transformer/run_glue.py <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/roberta-optical-transformer/run_glue.py>`_

- **Optical Attention** — custom attention with quantization-aware operations simulating
  optical matrix multiplication.

  - Implementation: `src/aixsim_models/optical_compute/optical_transformer/fine_tune/ot_roberta.py <https://github.com/AICrossSim/NewComputeBench/blob/master/src/aixsim_models/optical_compute/optical_transformer/fine_tune/ot_roberta.py>`_

- **GLUE Task Support** — fine-tune and evaluate on all GLUE benchmark tasks.

  - Multi-task script: `experiments/roberta-optical-transformer/finetune_base.sh <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/roberta-optical-transformer/finetune_base.sh>`_

The simulation uses custom Triton kernels from
`mase-triton <https://pypi.org/project/mase-triton/>`_ to accelerate quantization-aware
operations (see :doc:`mase_triton`).


Optical Transform Configuration
---------------------------------

The transform is controlled through a YAML config file.

Configuration parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``q_levels`` — number of quantization levels (default: 256)
- ``q_lut_min`` — minimum lookup-table value for quantization (default: 0.020040)
- ``q_quantiles`` — optional quantile-based range setting (default: null)
- ``q_smooth_factor`` — smoothing factor for statistics updates (default: 0.9)
- ``q_init_seed`` — random seed for initialization (default: 0)
- ``q_bypass`` — bypass the optical transform (default: false)

Default configuration (``experiments/roberta-optical-transformer/transform_cfg.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   "attn":
     q_levels: 256
     q_lut_min: 0.020040
     q_quantiles: null
     q_smooth_factor: 0.9
     q_init_seed: 0
     q_bypass: false
   "fc":
     q_levels: 256
     q_lut_min: 0.020040
     q_quantiles: null
     q_smooth_factor: 0.9
     q_init_seed: 0
     q_bypass: false


Fine-Tuning RoBERTa with Optical Transform
-------------------------------------------

Single task
~~~~~~~~~~~

.. code-block:: bash

   cd experiments/roberta-optical-transformer

   TASK_NAME="mrpc"
   MODEL_NAME="FacebookAI/roberta-base"
   LEARNING_RATE="2e-5"
   BATCH_SIZE="16"
   NUM_EPOCHS="3"
   TRANSFORM_CONFIG="transform_cfg.yaml"

   python run_glue.py \
       --model_name_or_path ${MODEL_NAME} \
       --task_name ${TASK_NAME} \
       --do_train \
       --do_eval \
       --max_seq_length 128 \
       --per_device_train_batch_size ${BATCH_SIZE} \
       --learning_rate ${LEARNING_RATE} \
       --num_train_epochs ${NUM_EPOCHS} \
       --output_dir ./output/${TASK_NAME}_optical \
       --overwrite_output_dir \
       --transform_config ${TRANSFORM_CONFIG} \
       --evaluation_strategy epoch \
       --save_strategy epoch \
       --logging_steps 50 \
       --seed 42

Multiple GLUE tasks
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd experiments/roberta-optical-transformer

   export USE_SINGLE_TASK=false
   export TASK_LIST="stsb mrpc cola"
   export LR_LIST="1e-3 2e-5 1e-5"
   export MODEL_NAME="FacebookAI/roberta-base"
   export BATCH_SIZE=16

   bash finetune_base.sh

Evaluation only
~~~~~~~~~~~~~~~

.. code-block:: bash

   python run_glue.py \
       --model_name_or_path ${MODEL_NAME} \
       --task_name ${TASK_NAME} \
       --do_eval \
       --max_seq_length 128 \
       --per_device_eval_batch_size ${BATCH_SIZE} \
       --output_dir ./output/${TASK_NAME}_eval \
       --transform_config ${TRANSFORM_CONFIG} \
       --overwrite_output_dir

Baseline comparison (no transform)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python run_glue.py \
       --model_name_or_path ${MODEL_NAME} \
       --task_name ${TASK_NAME} \
       --do_train \
       --do_eval \
       --max_seq_length 128 \
       --per_device_train_batch_size ${BATCH_SIZE} \
       --learning_rate ${LEARNING_RATE} \
       --num_train_epochs ${NUM_EPOCHS} \
       --output_dir ./output/${TASK_NAME}_baseline \
       --overwrite_output_dir \
       --evaluation_strategy epoch \
       --save_strategy epoch


Results
-------

Post-training transform (optical transform applied to a trained RoBERTa, no fine-tuning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 10 10 9 9 10 11 9 14 10

   * - Model
     - MNLI
     - QNLI
     - RTE
     - SST
     - MRPC
     - CoLA
     - QQP
     - STSB
     - Avg
   * - Original
     - 0.8728
     - 0.9244
     - 0.7978
     - 0.9357
     - 0.9019
     - 0.6232
     - 0.9153
     - 0.9089
     - 0.8600
   * - Random
     - 0.3266
     - 0.4946
     - 0.5271
     - 0.4908
     - 0.3162
     - 0.0000
     - 0.6318
     - 0.0332
     - 0.3525
   * - Optical Transformer
     - 0.8000
     - 0.7966
     - 0.4801
     - 0.8704
     - 0.7770
     - 0.2034
     - 0.9075
     - 0.8485
     - 0.7104
   * - SqueezeLight
     - 0.3200
     - 0.4961
     - 0.4404
     - 0.5126
     - 0.5025
     - 0.0213
     - 0.5890
     - −0.0543
     - 0.3582

Transform-aware fine-tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 10 10 9 9 10 11 9 14 10

   * - Model
     - MNLI
     - QNLI
     - RTE
     - SST
     - MRPC
     - CoLA
     - QQP
     - STSB
     - Avg
   * - Original
     - 0.8728
     - 0.9244
     - 0.7978
     - 0.9357
     - 0.9019
     - 0.6232
     - 0.9153
     - 0.9089
     - 0.8600
   * - Random
     - 0.3266
     - 0.4946
     - 0.5271
     - 0.4908
     - 0.3162
     - 0.0000
     - 0.6318
     - 0.0332
     - 0.3525
   * - Optical Transformer
     - 0.8510
     - 0.9032
     - 0.5813
     - 0.9140
     - 0.8677
     - 0.4441
     - 0.9060
     - 0.0332
     - 0.6876
   * - SqueezeLight
     - 0.3212
     - 0.4961
     - 0.4676
     - 0.5131
     - 0.5025
     - 0.0000
     - 0.5932
     - 0.0514
     - 0.3681

**Takeaways**

- The Optical Transformer significantly outperforms SqueezeLight in both evaluation modes.
  SqueezeLight was designed for convolutional networks and does not transfer well to Transformers.
- Transform-aware fine-tuning generally outperforms post-training transform, but the noisy
  forward pass with straight-through estimators can occasionally cause instability (e.g., STSB).
- We carry the Optical Transformer forward to large-scale CLM experiments.
  See :doc:`onn_clm` for the follow-up.
