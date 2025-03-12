# Random BitFlip Aware Pretraining

## Overview

This is bitflip-aware training adapted from [llm-digital](/experiments/llm-digital/pretrain/).

## Emulated Random BitFlip

- We wrote random bitflip triton kernels to support
    - forward & backward propagation
    - separate bitflip probabilities for sign-exp part and frac part
    - BF16, FP16, FP32 dtypes
    - Zero out by threshold: If the absolute value of the flipped number of larger than a predefined threshold (for NaN as well), the number is set to 0.
    - For zero-outed values, it's gradient is also 0, otherwise it's identity. (The behaviour is like a combination of QAT and Dropout without scaling)
- This kernel can be inserted into any layer of the model, but it breaks `torch.compile` in this pretraining codebase.






