# Post-Training Transformation and Evaluation

`minimal.py` provides a minimal example of replacing linear layers in a HuggingFace model
with `RandomBitFlipLinear` layers and the corresponding evaluation using `lm-eval`.

## Extra Dependencies

- [`mase-triton`](https://github.com/DeepWok/mase-triton)