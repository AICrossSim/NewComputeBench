Quick Start
===========

This guide gets you from zero to running inference with a pretrained language model in
a few minutes. No training required.

Prerequisites
-------------

- Completed :doc:`installation`
- At least one CUDA-enabled GPU with ≥ 8 GB VRAM

Step 1 — Activate the environment
----------------------------------

.. code-block:: bash

   conda activate new-compute

Step 2 — Generate text with a pretrained CLM
---------------------------------------------

We publish pretrained ``AICrossSim-CLM`` checkpoints on HuggingFace.
The command below downloads ``AICrossSim/clm-60m`` automatically and runs text generation —
no local training needed.

.. code-block:: bash

   cd experiments/llm-digital/pretrain

   python run.py hf-gen \
       --model_name AICrossSim/clm-60m \
       --prompt "London is" \
       --max_new_tokens 100 \
       --do_sample true \
       --temperature 0.6 \
       --top_k 50 \
       --top_p 0.9

You should see generated text printed to stdout within a few seconds.

.. tip::

   Swap ``AICrossSim/clm-60m`` for ``AICrossSim/clm-200m``,
   ``AICrossSim/clm-400m``, or ``AICrossSim/clm-1.1b`` to use a larger model.
   Larger models require more VRAM.

Step 3 — Run evaluation on a downstream task
---------------------------------------------

Evaluate the same checkpoint on ``wikitext`` using ``lm-eval-harness``:

.. code-block:: bash

   python run.py eval hf-lm-eval \
       AICrossSim/clm-60m \
       --tasks ['wikitext'] \
       --dtype float16

What's Next
-----------

- :doc:`../tutorials/pretraining/llm_pretrain_eval` — pretrain your own CLM from scratch
- :doc:`../tutorials/simulations/bitflip_clm` — simulate random bitflip noise during pretraining
- :doc:`../tutorials/simulations/bitflip_lora` — LoRA fine-tuning of Llama-3.1-8B with bitflip noise
- :doc:`../tutorials/simulations/onn_roberta` — optical neural network experiments on RoBERTa
- :doc:`../tutorials/simulations/pim_roberta` — processing-in-memory simulation on RoBERTa
