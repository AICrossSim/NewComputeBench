# Random BitFlip Aware Pretraining

## Model Arch and Datasets

- Model architecture is based on Llama-3.1 with smaller vocab, hidden sizes and number of layers RMSNorm
    - Grouped Query Attention
    - RoPE positional encoding
    - MLP: `up_proj, gate_proj, down_proj`
    - tokenizer/vocab: [`HuggingFaceTB/cosmo2-tokenizer`](https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer)
- Pretraining is performed on [`HuggingFaceFW/fneweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

## Emulated Random BitFlip

- We wrote random bitflip triton kernels to support
    - forward & backward propagation
    - separate bitflip probabilities for sign-exp part and frac part
    - BF16, FP16, FP32 dtypes
    - Zero out by threshold: If the absolute value of the flipped number of larger than a predefined threshold (for NaN as well), the number is set to 0.
    - For zero-outed values, it's gradient is also 0, otherwise it's identity. (The behaviour is like a combination of QAT and Dropout without scaling)
- This kernel can be inserted into any layer of the model, but it breaks `torch.compile` in this pretraining codebase.






