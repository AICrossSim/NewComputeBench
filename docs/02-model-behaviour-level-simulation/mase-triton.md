# MASE-Triton

[PyPI Link](https://pypi.org/project/mase-triton/) | [GitHub Link](https://github.com/DeepWok/mase-triton)

Mase-triton is a PyTorch extension library that provides efficient implementations of various operations used in simulating new compute paradigms and our [PLENA project](https://arxiv.org/abs/2509.09505),
including random bitflip, optical transformer, MXFP (Microscaling Formats), and minifloat. It leverages the Triton language to enable faster simulations on CUDA-enabled GPUs.


## Functionality
- **Random Bitflip**: Simulate random bit flips in neural network computations
    - [`functional APIs`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/random_bitflip/functional.py): Random bitflip functions with backward support.
        - `random_bitflip_fn`: Perform random bit flipping on tensors with configurable exponent and fraction bit flip probabilities
        - `calculate_flip_probability`: Calculate flip probability from number of halves
        - `find_nearest_prob_n_halves`: Find nearest probability in terms of halves
    - [`layers`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/random_bitflip/layers.py): PyTorch modules for random bitflip operations.
        - `RandomBitFlipDropout`: Random bit flip layer with dropout functionality
        - `RandomBitFlipLinear`: Linear layer with random bit flipping

- **[Optical Transformer](https://arxiv.org/abs/2302.10360)**: Simulate optical computing for transformers
    - [`functional APIs`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/optical_compute/functional.py): Optical transformer functions with backward support.
        - `ot_quantize_fn`: Quantize tensors for optical transformer operations
        - `ot_qlinear_fn`: Quantized linear transformation for optical computing
        - `ot_qmatmul_fn`: Quantized matrix multiplication for optical computing
    - [`layers`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/optical_compute/layers.py): PyTorch modules for optical computing.
        - `OpticalTransformerLinear`: Linear layer with optical transformer quantization

- **[MXFP](https://arxiv.org/abs/2310.10537)**: Simulate MXFP (Microscaling Formats) on CPU & GPU using PyTorch & Triton
    - [`functional`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/mxfp/functional/__init__.py): MXFP format conversion and operations.
        - `extract_mxfp_components`: Extract MXFP components (shared exponent and elements) from tensors
        - `compose_mxfp_tensor`: Compose MXFP components back to standard floating-point tensors
        - `quantize_dequantize`: Quantize and dequantize tensors using MXFP format
        - `flatten_for_quantize`: Flatten tensors for quantization operations
        - `permute_for_dequantize`: Permute tensors for dequantization operations
        - `mxfp_linear`: Linear operation with MXFP support
        - `mxfp_matmul`: Matrix multiplication with MXFP support
        - `parse_mxfp_linear_type`: Parse MXFP linear layer types
    - [`layers`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/mxfp/layers.py): PyTorch modules with MXFP support.
        - `MXFPLinearPTQ`: Linear layer with MXFP support for post-training quantization (no backpropagation support)

- **Minifloat**: Simulate minifloat formats on CPU & GPU using PyTorch & Triton
    - [`functional`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/minifloat/functional/__init__.py): Minifloat format operations.
        - `extract_minifloat_component`: Extract minifloat components from tensors
        - `compose_minifloat_component`: Compose minifloat components back to tensors
        - `quantize_dequantize`: Quantize and dequantize tensors using minifloat format
        - `minifloat_linear`: Linear operation with minifloat support
        - `minifloat_matmul`: Matrix multiplication with minifloat support
    - [`layers`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/minifloat/layers.py): PyTorch modules with minifloat support.
        - `MinifloatLinearPTQ`: Linear layer with minifloat support for post-training quantization (no backpropagation support)

- **Utilities & Management**
    - [`manager.py`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/manager.py): Kernel management and autotune control.
        - `KernelManager`: Enable/disable autotune for Triton kernels
    - [`utils/`](https://github.com/DeepWok/mase-triton/tree/master/src/mase_triton/utils): Various utility functions for PyTorch modules, debugging, and training.
