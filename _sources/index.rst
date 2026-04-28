NewComputeBench
===============

.. figure:: _static/images/logo.png
   :width: 200px
   :align: center
   :alt: NewComputeBench

**NewComputeBench** (`GitHub <https://github.com/AICrossSim/NewComputeBench>`_) is a
benchmark suite for new compute paradigms — Spiking Neural Networks, Optical computation,
Processing-in-Memory, and more — via software emulation.
The project aims to predict the scaling law of neural networks trained with new compute
paradigms by running small- and medium-scale experiments and extrapolating observed trends.

The project is led by `Dr. Yiren Zhao <https://aaron-zhao123.github.io/>`_ (Imperial College
London), `Dr. Luo Mai <https://luomai.github.io/>`_ (University of Edinburgh), and
`Prof. Robert Mullins <https://www.cl.cam.ac.uk/~rdm34/>`_ (University of Cambridge),
and is funded by the `Advanced Research + Invention Agency (ARIA) <https://www.aria.org.uk/>`_.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   modules/getting_started/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   modules/tutorials/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Developer Guide

   modules/developer/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Changelog

   modules/changelog


Project Overview
----------------

NewComputeBench is structured around three phases:

1. Build a scaling framework to support pretraining of language models up to 1.1B parameters
   (the AICrossSim-CLM series).
2. Implement software emulation of new compute paradigms (optical compute, spiking neural
   networks, in-memory compute, etc.).
3. Filter out promising paradigms through small- and medium-scale experiments, then scale up.

Current status:

- ✅ Scaling framework for CLM pretraining (60M – 1.1B)
- ✅ Software emulation of Random Bitflip, Optical Compute, Spiking Neural Networks, PIM
- ✅ RoBERTa experiments on GLUE (sanity checks)
- ✅ CLM bitflip-aware pretraining and LoRA fine-tuning of Llama-3.1-8B
- ⏹️ Full CLM scaling for Optical Compute, SNN, PIM


Roadmap
-------

**Model Training & Evaluation**

- ✅ Pretraining of CLM models (60M, 200M, 400M, 1.1B) using the Llama-3 architecture
- ✅ ``lm-eval-harness`` evaluation of pretrained CLMs
- ✅ Parameter-efficient fine-tuning (LoRA)

**Model Behaviour-Level Simulation**

- ✅ Random Bitflip

  - ✅ Post-training bitflip transform
  - ✅ Bitflip-aware pretraining (60M – 1.1B)
  - ✅ Bitflip-aware LoRA fine-tuning (Llama-3.1-8B)

- ✅ Optical Compute

  - ✅ RoBERTa fine-tuning (125M)
  - ✅ CLM full fine-tuning (60M – 1.1B)
  - ✅ CLM parameter-efficient fine-tuning (60M – 1.1B)

- ✅ Spiking Neural Networks

  - ✅ RoBERTa fine-tuning (125M)

- ✅ Processing in Memory

  - ✅ RoBERTa fine-tuning (125M)
  - ✅ ViT-Base fine-tuning (86M)


.. _changelog-index:

What's New
----------

**4 Feb 2026** — Bitflip-aware LoRA fine-tuning of Llama-3.1-8B.
LoRA adapters with only 1.2% trainable parameters reduce perplexity from 1008.95 to 11.01
(clean baseline: 7.91). See :doc:`modules/tutorials/simulations/bitflip_lora`.

**4 Oct 2025** — Optical Transformer fine-tuning on CLM models (60M – 1.1B).
See :doc:`modules/tutorials/simulations/onn_clm`.

**1 Oct 2025** — Optical Transformer, Spiking Transformer, and PIM experiments on RoBERTa.
See :doc:`modules/tutorials/simulations/onn_roberta`,
:doc:`modules/tutorials/simulations/snn_roberta`,
:doc:`modules/tutorials/simulations/pim_roberta`.

**9 Jun 2025** — Mase-triton released on PyPI (``pip install mase-triton``).
See :doc:`modules/tutorials/simulations/mase_triton`.

**15 Apr 2025** — System and model-level training simulation (Small Language Models).
Environment setup, pretraining of AICrossSim-CLM (60M – 1.1B), and bitflip-aware pretraining.
See :doc:`modules/getting_started/installation` and
:doc:`modules/tutorials/pretraining/llm_pretrain_eval`.
