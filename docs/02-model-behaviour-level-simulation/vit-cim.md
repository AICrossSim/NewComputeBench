# Vision Transformer CIM Simulation

This tutorial demonstrates how to apply Compute-in-Memory (CIM) transformations to Vision Transformer (ViT) models for CIM-aware fine-tuning.

## Overview

- This tutorial takes a pretrained ViT model, applies CIM transformation, and fine-tunes the model on ImageNet dataset.
    - The entry point is at [`experiments/vit-cim/run_vit.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/vit-cim/run_vit.py).

The CIM transformation simulates the effect of compute-in-memory architectures, with both digital and analog support.

## Environment Setup
!!! info "Environment Setup"
    If you have not set up environments, please follow the guidelines in [Environment Setup](../env-setup.md).

If you want to run the compute in memory transformation, you need to install the mase with compute in memory support by running the following command:
```bash
pip install --force-reinstall mase-tools@git+https://github.com/DeepWok/mase@cx/comput_in_memory
```

## Evaluation of CIM-aware Fine-tuning

We provide scripts to apply CIM-aware transformation on Vision Transformer models and evaluate their performance on ImageNet dataset.

### CIM-aware Fine-tuning & Evaluate on ImageNet

```bash
git clone https://github.com/AICrossSim/NewComputeBench.git
cd NewComputeBench

model_name="google/vit-base-patch16-224"    # HuggingFace ViT model
dataset_name="imagenet"                      # Vision dataset for evaluation
cim_config_path="./experiments/llm-cim/configs/sram.yaml" # CIM transformation configuration
output_dir="./log_eval_results"             # Output directory for results

python experiments/vit-cim/run_vit.py \
    --model_name_or_path ${model_name} \
    --dataset_name ${dataset_name} \
    --cim_config_path ${cim_config_path} \
    --output_dir ${output_dir} \
    --per_device_eval_batch_size 32 \
    --enable_cim_transform \
    --do_eval
```

!!! info "CIM Configuration"
    The CIM configuration file defines the noise characteristics, quantization levels, and other parameters that simulate the analog compute-in-memory effects. See [`experiments/llm-cim/configs/`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-cim/configs/) for example configurations.

## CIM Configuration Examples

### Typical SRAM CIM Configuration

```yaml
# experiments/llm-cim/configs/sram.yaml
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
