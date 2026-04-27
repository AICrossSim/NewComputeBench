Processing-in-Memory on RoBERTa
================================

This tutorial demonstrates PIM-aware fine-tuning of RoBERTa for sequence classification.
The simulation supports SRAM (digital), RRAM (analogue), and PCM (analogue) technologies.

.. note::

   If you have not set up the environment yet, follow :doc:`../../getting_started/installation` first.


Overview
--------

- **PIM-aware fine-tuning** — applies PIM transformation (noise injection and
  quantization) to RoBERTa and fine-tunes on GLUE benchmark tasks.

  - Entry point: `experiments/roberta-pim/run_glue.py <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/roberta-pim/run_glue.py>`_

Supported technologies:

- **SRAM (Digital)** — simulates digital CIM with FP8 quantization.
- **RRAM (Analogue)** — models analog non-idealities such as device variation and noise.
- **PCM (Analogue)** — models programming noise, read noise, IR drop, and output noise.

.. note::

   The PIM simulation uses quantization primitives from the
   `MASE <https://github.com/DeepWok/mase>`_ submodule
   (e.g., ``scale_integer_quantizer`` for integer quantization with dynamic scaling).
   Ensure MASE is installed: ``pip install -e ./submodules/mase``.


PIM Configuration
-----------------

PIM behaviour is controlled through YAML configuration files.
Example configs are in ``experiments/roberta-pim/configs/``:

- ``sram.yaml`` — Digital PIM (FP8)
- ``reram.yaml`` — Analogue RRAM
- ``pcm.yaml`` — Analogue PCM

SRAM (digital, FP8) — example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # experiments/roberta-pim/configs/sram.yaml
   by: "type"
   linear:
     config:
       tile_type: "digital"
       core_size: 64
       rescale_dim: "vector"
       x_quant_type: "e4m3"
       weight_quant_type: "e4m3"

Configuration parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

Common:

- ``tile_type`` — compute tile type: ``"digital"`` (SRAM), ``"reram"``, or ``"pcm"``
- ``core_size`` — size of the simulation tile core (default: 64)

SRAM-specific:

- ``rescale_dim`` — granularity of the pre-alignment mechanism: ``"vector"`` or ``"matrix"``
- ``x_quant_type`` — quantization format for input activations (e.g., ``"e4m3"``, ``"e5m2"``)
- ``weight_quant_type`` — quantization format for weights (same convention as ``x_quant_type``)

ReRAM-specific:

- ``noise_magnitude`` — magnitude of the Gaussian distribution modeling analogue non-idealities
- ``num_bits`` — weight resolution (number of bits) in ReRAM cells

PCM-specific:

- ``programming_noise`` — simulates programming errors and device-to-device variability
- ``read_noise`` — models temporal read noise and 1/f noise during inference
- ``ir_drop`` — simulates voltage drop across the crossbar interconnects
- ``out_noise`` — models additive system noise in the peripheral circuits


Fine-Tuning RoBERTa with PIM
------------------------------

.. code-block:: bash

   TASK_NAME="mrpc"
   MODEL_NAME="FacebookAI/roberta-base"
   PIM_CONFIG="./experiments/roberta-pim/configs/sram.yaml"

   python experiments/roberta-pim/run_glue.py \
       --model_name_or_path ${MODEL_NAME} \
       --task_name ${TASK_NAME} \
       --do_train \
       --do_eval \
       --max_seq_length 128 \
       --per_device_train_batch_size 16 \
       --learning_rate 2e-5 \
       --num_train_epochs 3 \
       --output_dir ./output/${TASK_NAME}_pim \
       --pim_config_path ${PIM_CONFIG} \
       --pim


Results
-------

Post-training transform (PIM applied to a trained RoBERTa, no fine-tuning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 9 9 10 11 9 12 10

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
   * - ReRAM (analogue)
     - 0.3239
     - 0.4860
     - 0.5090
     - 0.4839
     - 0.4338
     - −0.0363
     - 0.5796
     - −0.0011
     - 0.3475
   * - PCM (analogue)
     - 0.3211
     - 0.5123
     - 0.5090
     - 0.5068
     - 0.5098
     - 0.0443
     - 0.6162
     - 0.0745
     - 0.3872
   * - SRAM (digital, FP8)
     - 0.7825
     - 0.4939
     - 0.5271
     - 0.5092
     - 0.3186
     - 0.0000
     - 0.6318
     - 0.0198
     - 0.3523

PIM-aware fine-tuning
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 9 9 10 11 9 12 10

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
   * - RRAM (analogue)
     - 0.8416
     - 0.8669
     - 0.5451
     - 0.8761
     - 0.6961
     - 0.0000
     - 0.9052
     - 0.3611
     - 0.7228
   * - PCM (analogue)
     - 0.6850
     - 0.7681
     - 0.4729
     - 0.8383
     - 0.6838
     - 0.0000
     - 0.8585
     - −0.1037
     - 0.6108
   * - SRAM (digital, FP8)
     - 0.8589
     - 0.9129
     - 0.6643
     - 0.9220
     - 0.8505
     - 0.5113
     - 0.9128
     - 0.8815
     - 0.8143
