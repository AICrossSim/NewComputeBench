Processing-in-Memory on Vision Transformer
===========================================

This tutorial demonstrates PIM-aware fine-tuning of Vision Transformer (ViT) models.

.. note::

   If you have not set up the environment yet, follow :doc:`../../getting_started/installation` first.

.. note::

   The PIM simulation uses quantization primitives from the
   `MASE <https://github.com/DeepWok/mase>`_ submodule.
   Ensure MASE is installed: ``pip install -e ./submodules/mase``.


Overview
--------

- **PIM-aware fine-tuning** — applies PIM transformation to a pretrained ViT and
  fine-tunes on downstream vision datasets.

  - Entry point: `experiments/vit-pim/run_vit.py <https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/vit-pim/run_vit.py>`_

Both digital (SRAM) and analogue (RRAM, PCM) PIM modes are supported.
See :doc:`pim_roberta` for a full description of PIM configuration parameters.


Supported Models and Datasets
------------------------------

Models
~~~~~~

Any ViT model from HuggingFace, e.g., ``google/vit-base-patch16-224``.

Datasets
~~~~~~~~

- **ImageNet** — 1000-class image classification (requires local path).


PIM Configuration
-----------------

SRAM (digital, FP8) example for ViT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # experiments/llm-pim/configs/sram.yaml
   by: "type"
   conv2d:
     config:
       tile_type: "digital"
       core_size: 16
       rescale_dim: "vector"
       x_quant_type: "e4m3"
       weight_quant_type: "e4m3"
   linear:
     config:
       tile_type: "digital"
       core_size: 64
       rescale_dim: "vector"
       x_quant_type: "e4m3"
       weight_quant_type: "e4m3"

The ViT config covers both ``conv2d`` (patch embedding) and ``linear`` (attention / MLP)
layers. See ``experiments/llm-pim/configs/`` for RRAM and PCM variants.


PIM-Aware Fine-Tuning
---------------------

.. code-block:: bash

   model_name="google/vit-base-patch16-224"
   dataset_name="imagenet"
   pim_config_path="./experiments/llm-pim/configs/sram.yaml"
   output_dir="./log_eval_results"

   python experiments/vit-pim/run_vit.py \
       --model_name_or_path ${model_name} \
       --dataset_name ${dataset_name} \
       --pim_config_path ${pim_config_path} \
       --output_dir ${output_dir} \
       --per_device_eval_batch_size 32 \
       --enable_pim_transform \
       --do_eval


Performance Metrics
-------------------

The evaluation reports:

- **Top-1 accuracy**
- **Top-5 accuracy**
