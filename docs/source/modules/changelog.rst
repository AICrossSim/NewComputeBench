Changelog
=========

4 February 2026
---------------

**Bitflip-aware LoRA fine-tuning of Llama-3.1-8B** (:doc:`tutorials/simulations/bitflip_lora`)

LoRA adapters with only 1.2% trainable parameters effectively mitigate random bitflip
noise, reducing validation perplexity from 1008.95 to 11.01 (clean baseline: 7.91).

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Item
     - Link
   * - Llama-3.1-8B with random bitflip noise
     - :doc:`tutorials/simulations/bitflip_lora`


4 October 2025
--------------

**Optical Transformer fine-tuning on CLM models (60M – 1.1B)** (:doc:`tutorials/simulations/onn_clm`)

Full fine-tuning of pretrained CLM models with optical transformer simulation.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Item
     - Link
   * - Optical Transformer on CLM
     - :doc:`tutorials/simulations/onn_clm`


1 October 2025
--------------

**Optical Transformer, Spiking Transformer, and PIM on RoBERTa**

Initial experiments on RoBERTa with three new compute paradigms.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Item
     - Link
   * - Optical Transformer on RoBERTa
     - :doc:`tutorials/simulations/onn_roberta`
   * - Spiking Transformer on RoBERTa
     - :doc:`tutorials/simulations/snn_roberta`
   * - Processing in Memory on RoBERTa
     - :doc:`tutorials/simulations/pim_roberta`


9 June 2025
-----------

**Mase-triton released on PyPI** (:doc:`tutorials/simulations/mase_triton`)

Our software-emulation and acceleration backend is now publicly available:

.. code-block:: bash

   pip install mase-triton

See :doc:`tutorials/simulations/mase_triton` for full documentation.


15 April 2025
-------------

**System and model-level training simulation for Small Language Models**

Initial release of the scaling framework and bitflip-aware pretraining pipeline.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Item
     - Link
   * - Environment setup
     - :doc:`getting_started/installation`
   * - Pretraining AICrossSim-CLM (60M – 1.1B) and evaluation
     - :doc:`tutorials/pretraining/llm_pretrain_eval`
   * - Bitflip-aware pretraining and evaluation
     - :doc:`tutorials/simulations/bitflip_clm`
