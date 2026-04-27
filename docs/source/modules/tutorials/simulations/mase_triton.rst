Mase-Triton
===========

`PyPI <https://pypi.org/project/mase-triton/>`_ | `GitHub <https://github.com/DeepWok/mase-triton>`_

Mase-triton is a PyTorch extension library providing efficient Triton-based implementations
of operations used in simulating new compute paradigms.
It underpins NewComputeBench's bitflip, optical transformer, and PIM simulations.


Installation
------------

.. code-block:: bash

   pip install mase-triton


Functionality
-------------

Random Bitflip
~~~~~~~~~~~~~~

Simulate random bit flips in neural network computations.

**Functional APIs** (`source <https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/random_bitflip/functional.py>`_):

- ``random_bitflip_fn`` — perform random bit flipping on tensors with configurable
  exponent and mantissa bit-flip probabilities (supports backward pass).
- ``calculate_flip_probability`` — calculate flip probability from a number of halves.
- ``find_nearest_prob_n_halves`` — snap a probability to the nearest valid power-of-0.5 value.

**Layers** (`source <https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/random_bitflip/layers.py>`_):

- ``RandomBitFlipDropout`` — random bit-flip layer with dropout functionality.
- ``RandomBitFlipLinear`` — linear layer with random bit flipping.

.. note::

   The bitflip probability must be a power of 0.5 (e.g., ``0.5^10 ≈ 9.77e-04``).
   The kernel snaps to the nearest valid value automatically.
   The minimum supported probability is ``0.5^24 ≈ 5.96e-08`` due to the Philox PRNG.

Optical Transformer
~~~~~~~~~~~~~~~~~~~

Simulate optical computing for transformer models
(`paper <https://arxiv.org/abs/2302.10360>`_).

**Functional APIs** (`source <https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/optical_compute/functional.py>`_):

- ``ot_quantize_fn`` — quantize tensors for optical transformer operations.
- ``ot_qlinear_fn`` — quantized linear transformation for optical computing.
- ``ot_qmatmul_fn`` — quantized matrix multiplication for optical computing.

**Layers** (`source <https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/optical_compute/layers.py>`_):

- ``OpticalTransformerLinear`` — linear layer with optical transformer quantization.

MXFP (Microscaling Formats)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulate `MXFP <https://arxiv.org/abs/2310.10537>`_ on CPU and GPU using PyTorch and Triton.

**Functional** (`source <https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/mxfp/functional/__init__.py>`_):

- ``extract_mxfp_components`` — extract MXFP components (shared exponent and elements).
- ``compose_mxfp_tensor`` — compose MXFP components back to standard floats.
- ``quantize_dequantize`` — quantize and dequantize tensors using MXFP format.
- ``mxfp_linear`` / ``mxfp_matmul`` — linear and matmul with MXFP support.

**Layers** (`source <https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/mxfp/layers.py>`_):

- ``MXFPLinearPTQ`` — linear layer with MXFP for post-training quantization
  (no backpropagation support).

Minifloat
~~~~~~~~~

Simulate minifloat formats on CPU and GPU.

**Functional** (`source <https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/minifloat/functional/__init__.py>`_):

- ``extract_minifloat_component`` / ``compose_minifloat_component`` — component extraction and composition.
- ``quantize_dequantize`` — quantize and dequantize using minifloat format.
- ``minifloat_linear`` / ``minifloat_matmul`` — linear and matmul with minifloat support.

**Layers** (`source <https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/minifloat/layers.py>`_):

- ``MinifloatLinearPTQ`` — linear layer with minifloat for post-training quantization.

Utilities
~~~~~~~~~

- `manager.py <https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/manager.py>`_ —
  ``KernelManager``: enable/disable Triton kernel autotuning.
  HuggingFace Trainer does not work with autotuned Triton kernels; autotuning is
  therefore disabled by default in mase-triton.
- `utils/ <https://github.com/DeepWok/mase-triton/tree/master/src/mase_triton/utils>`_ —
  utility functions for PyTorch modules, debugging, and training.
