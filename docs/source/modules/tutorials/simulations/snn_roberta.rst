Spiking Neural Networks on RoBERTa
====================================

This tutorial demonstrates how to convert a RoBERTa model to a Spiking Neural Network
(SNN) transformer for sequence classification on GLUE benchmark tasks.

.. note::

   If you have not set up the environment yet, follow :doc:`../../getting_started/installation` first.


Overview
--------

- **SNN Transform** — a two-stage conversion pipeline: (1) quantize with LSQ Integer
  layers, then (2) convert to SNN equivalents using ST-BIF spiking neurons.

  - Entry point: `experiments/roberta-snn-transformer/run_glue.py <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/roberta-snn-transformer/run_glue.py>`_

- **SNN RoBERTa** — custom SNN attention and linear layers replacing standard modules
  with spiking-neuron equivalents.

  - Implementation: `src/aixsim_models/snn/fine_tune/snn_roberta.py <https://github.com/AICrossSim/NewComputeBench/blob/master/src/aixsim_models/snn/fine_tune/snn_roberta.py>`_

- **Multi-task script**: `experiments/roberta-snn-transformer/finetune_base.sh <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/roberta-snn-transformer/finetune_base.sh>`_

The SNN transformation pipeline:

1. **Quantization stage** — replaces attention, output, intermediate, and classifier
   layers with LSQ Integer quantized equivalents.
2. **SNN conversion stage** — converts quantized layers to spiking equivalents:

   - ``zip_tf`` (SpikeZip-TF) for attention blocks, embeddings, and LayerNorm
   - ``unfold_bias`` linear layers with ST-BIF neurons for fully connected layers
   - ``identity`` replaces ReLU activations
   - ``st_bif`` spiking nodes replace LSQ Integer quantizers

.. note::

   **ReLU-based base model required.** The SNN conversion requires a RoBERTa model with
   ReLU activations (not GELU). Use ``JeffreyWong/roberta-base-relu-{task}`` checkpoints
   from HuggingFace.


SNN Transform Configuration
-----------------------------

Configuration structure
~~~~~~~~~~~~~~~~~~~~~~~~

The YAML config contains three top-level keys:

- ``quantization_config`` — applies LSQ Integer quantization by regex matching on module names.
- ``snn_transformer_config_attn`` — converts quantized attention layers to SNN by regex.
- ``snn_transformer_config_fc`` — converts remaining layer types (embedding, LayerNorm,
  linear, ReLU, LSQ Integer) to SNN equivalents by type.

Quantization parameters
~~~~~~~~~~~~~~~~~~~~~~~~

- ``name`` — quantization method (``lsqinteger``)
- ``level`` — number of quantization levels (default: 32)

SNN conversion parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``name`` — conversion type: ``zip_tf`` (attention/embedding/LayerNorm),
  ``unfold_bias`` (linear), ``st_bif`` (quantizer nodes), ``identity`` (ReLU)
- ``level`` — spike resolution (default: 32, used by ``unfold_bias``)
- ``neuron_type`` — spiking neuron model (default: ``ST-BIF``)

Default configuration (``experiments/roberta-snn-transformer/transform_cfg.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   quantization_config:
     by: regex
     'roberta\.encoder\.layer\.\d+\.attention\.self':
       config:
         name: lsqinteger
         level: 32
     'roberta\.encoder\.layer\.\d+\.attention\.output':
       config:
         name: lsqinteger
         level: 32
     'roberta\.encoder\.layer\.\d+\.output':
       config:
         name: lsqinteger
         level: 32
     'roberta\.encoder\.layer\.\d+\.intermediate':
       config:
         name: lsqinteger
         level: 32
     classifier:
       config:
         name: lsqinteger
         level: 32

   snn_transformer_config_attn:
     by: regex
     'roberta\.encoder\.layer\.\d+\.attention\.self':
       config:
         name: zip_tf
         level: 32
         neuron_type: ST-BIF

   snn_transformer_config_fc:
     by: type
     embedding:
       config:
         name: zip_tf
     layernorm:
       config:
         name: zip_tf
     linear:
       config:
         name: unfold_bias
         level: 32
         neuron_type: ST-BIF
     relu:
       manual_instantiate: true
       config:
         name: identity
     lsqinteger:
       manual_instantiate: true
       config:
         name: st_bif


Fine-Tuning RoBERTa with SNN Transform
----------------------------------------

Single task
~~~~~~~~~~~

.. code-block:: bash

   cd experiments/roberta-snn-transformer

   TASK_NAME="mrpc"
   MODEL_NAME="JeffreyWong/roberta-base-relu-${TASK_NAME}"
   LEARNING_RATE="2e-5"
   BATCH_SIZE="64"
   NUM_EPOCHS="10"
   TRANSFORM_CONFIG="./transform_cfg.yaml"

   python run_glue.py \
       --model_name_or_path ${MODEL_NAME} \
       --task_name ${TASK_NAME} \
       --do_train \
       --do_eval \
       --max_seq_length 128 \
       --per_device_train_batch_size ${BATCH_SIZE} \
       --learning_rate ${LEARNING_RATE} \
       --num_train_epochs ${NUM_EPOCHS} \
       --output_dir ./output/${TASK_NAME}_snn \
       --overwrite_output_dir \
       --transform_config ${TRANSFORM_CONFIG} \
       --evaluation_strategy epoch \
       --save_strategy epoch \
       --logging_steps 50 \
       --seed 42

Evaluation only (post-transform, no fine-tuning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Evaluation with fine-tuned SNN weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python run_glue.py \
       --model_name_or_path ${MODEL_NAME} \
       --task_name ${TASK_NAME} \
       --do_eval \
       --max_seq_length 128 \
       --per_device_eval_batch_size ${BATCH_SIZE} \
       --output_dir ./output/${TASK_NAME}_eval \
       --transform_config ${TRANSFORM_CONFIG} \
       --model_weights_path ./output/${TASK_NAME}_snn \
       --overwrite_output_dir

Convert to full spiking form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python run_glue.py \
       --model_name_or_path ${MODEL_NAME} \
       --task_name ${TASK_NAME} \
       --do_eval \
       --max_seq_length 128 \
       --per_device_eval_batch_size ${BATCH_SIZE} \
       --output_dir ./output/${TASK_NAME}_eval \
       --transform_config ${TRANSFORM_CONFIG} \
       --convert_to_snn \
       --overwrite_output_dir

Baseline (no SNN transform)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Results (MRPC)
--------------

.. list-table::
   :header-rows: 1
   :widths: 45 20 15 20

   * - Model
     - Accuracy
     - F1
     - Combined Score
   * - Post-transform (no fine-tune)
     - 0.5613
     - 0.6551
     - 0.6082
   * - SNN fine-tuned (10 epochs)
     - 0.7819
     - 0.8468
     - 0.8143

**Takeaways**

- Applying the SNN transform without fine-tuning causes a significant accuracy drop
  because the integer-quantized weights are not yet adapted to the spiking regime.
- SNN-aware fine-tuning (10 epochs) substantially recovers accuracy, approaching the
  performance of the quantized ANN baseline.
- The ST-BIF neuron enables effective gradient flow during fine-tuning via a
  straight-through estimator.
