# Vision Transformer PIM Simulation

This tutorial demonstrates how to apply Processing-in-Memory (PIM) transformations to Vision Transformer (ViT) models for PIM-aware fine-tuning.

## Overview

- **PIM-aware fine-tuning** takes a pretrained ViT model, applies PIM transformation, and fine-tunes the model on downstream vision datasets.
    - The entry point is at [`experiments/vit-pim/run_vit.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/vit-pim/run_vit.py).

The PIM transformation simulates the effect of processing-in-memory architectures, with both digital and analog support.

## Evaluation of PIM-aware Fine-tuning

!!! info "Environment Setup"
    If you have not set up environments, please follow the guidelines in [Environment Setup](../env-setup.md).

We provide scripts to apply PIM-aware fine-tuning on Vision Transformer models and evaluate their performance on standard vision benchmarks.

### PIM-aware Fine-tuning & Evaluate on Vision Tasks

```bash
git clone https://github.com/AICrossSim/NewComputeBench.git
cd NewComputeBench

model_name="google/vit-base-patch16-224"    # HuggingFace ViT model
dataset_name="imagenet"                      # Vision dataset for evaluation
pim_config_path="./experiments/llm-pim/configs/sram.yaml" # PIM transformation configuration
output_dir="./log_eval_results"             # Output directory for results

python experiments/vit-pim/run_vit.py \
    --model_name_or_path ${model_name} \
    --dataset_name ${dataset_name} \
    --pim_config_path ${pim_config_path} \
    --output_dir ${output_dir} \
    --per_device_eval_batch_size 32 \
    --enable_pim_transform \
    --do_eval
```

!!! info "PIM Configuration"
    The PIM configuration file defines the noise characteristics, quantization levels, and other parameters that simulate the analog processing-in-memory effects. See [`experiments/llm-pim/configs/`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-pim/configs/) for example configurations.

## PIM Configuration Examples

### Typical SRAM PIM Configuration

```yaml
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
```

## Supported Models and Datasets

### Models
Support ViT model family from huggingface:
e.g.`google/vit-base-patch16-224`

### Datasets
- **ImageNet**: 1000-class image classification (requires custom path)

## Performance Metrics

The evaluation provides comprehensive metrics:

- **Accuracy**: Top-1 and Top-5 accuracy
