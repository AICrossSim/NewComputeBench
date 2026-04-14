Supported Models
================

The table below lists all models currently targeted by NewComputeBench.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Task
     - Model
     - Sizes
     - Notes
   * - Text classification
     - ``RoBERTa``
     - ``roberta-base``
     - Encoder-only model included as a sanity-check baseline.
   * - Causal language modeling
     - ``AICrossSim-CLM``
     - 60M, 200M, 400M, 1.1B
     - Custom family using the `Llama-3.1 architecture <https://arxiv.org/abs/2407.21783>`_.
       Trained with `cosmo2-tokenizer <https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer>`_
       on `FineWeb-Edu <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>`_.
       Checkpoints: `AICrossSim collection <https://huggingface.co/collections/AICrossSim/newcomputebench-clm-digital-67d19e95ebacdbc3e5752be3>`_.
   * - Causal language modeling
     - ``Llama-3``
     - 1B, 3B, 8B, 70B
     - Meta's `Llama-3 <https://arxiv.org/abs/2407.21783>`_ family.
   * - Image classification
     - ``ViT-Base``
     - 86M
     - ``google/vit-base-patch16-224`` from HuggingFace.
   * - Causal language modeling
     - TBD
     - TBD
     -
   * - Image generation
     - TBD
     - TBD
     -


Training Support
----------------

Pretraining from scratch
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Model
     - Supported
   * - ``RoBERTa``
     - ✅
   * - ``AICrossSim-CLM``, ``Llama-3``
     - ✅

Fine-tuning
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Model
     - Supported
   * - ``RoBERTa``
     - ✅
   * - ``AICrossSim-CLM``, ``Llama-3``
     - ⏹️

Evaluation
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Task
     - Model
     - Supported
   * - Text classification (GLUE)
     - ``RoBERTa``
     - ✅
   * - Causal language modeling
     - ``AICrossSim-CLM``, ``Llama-3``
     - ✅
   * - `lm-eval-harness <https://github.com/EleutherAI/lm-evaluation-harness>`_ benchmarks
     - ``AICrossSim-CLM``, ``Llama-3``
     - ✅


Model Behaviour-Level Simulation
----------------------------------

Transform-aware pretraining from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Transform
     - Model
     - Supported
   * - Random Bitflip
     - ``AICrossSim-CLM``, ``Llama-3``
     - ✅
   * - Optical Compute
     - ``AICrossSim-CLM``, ``Llama-3``
     - ⏹️
   * - In-Memory Compute
     - ``AICrossSim-CLM``, ``Llama-3``
     - ⏹️
   * - Spiking Neural Networks
     - ``AICrossSim-CLM``, ``Llama-3``
     - ⏹️

Post-transform evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Transform
     - Task
     - Model
     - Supported
   * - Random Bitflip
     - lm-eval-harness
     - ``AICrossSim-CLM``, ``Llama-3``
     - ⏹️
   * - Optical Compute
     - lm-eval-harness
     - ``AICrossSim-CLM``, ``Llama-3``
     - ⏹️
   * - In-Memory Compute
     - lm-eval-harness
     - ``AICrossSim-CLM``, ``Llama-3``
     - ⏹️
   * - Spiking Neural Networks
     - lm-eval-harness
     - ``AICrossSim-CLM``, ``Llama-3``
     - ⏹️
