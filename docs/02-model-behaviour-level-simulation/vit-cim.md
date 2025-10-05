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
pip uninstall mase-tools
git submodule add https://github.com/DeepWok/mase.git submodules/mase
cd submodules/mase
git checkout 248b69e2b073c19e41d21a9bd71027cf604fad97
pip install -e .
cd ../../
```

## Evaluation of CIM-aware Transformation

We provide scripts to apply CIM-aware transformation on Vision Transformer models and evaluate their performance on ImageNet dataset.

### CIM-aware Transformation & Evaluate on ImageNet

```bash
git clone https://github.com/AICrossSim/NewComputeBench.git
cd NewComputeBench

model_name="google/vit-base-patch16-224"    # HuggingFace ViT model
dataset_name="imagenet"                      # Vision dataset for evaluation
cim_config_path="./experiments/llm-cim/configs/sram.yaml" # CIM transformation configuration
output_dir="./log_eval_results"             # Output directory for results
custom_path="/data/datasets/imagenet_pytorch/" # Custom path for dataset

CUDA_VISIBLE_DEVICES=1,2 python experiments/vit-cim/run_vit.py \
    --model_name_or_path ${model_name} \
    --dataset_name ${dataset_name} \
    --cim_config_path ${cim_config_path} \
    --do_eval \
    --output_dir ${output_dir} \
    --per_device_eval_batch_size 64 \
    --custom_path ${custom_path}
```

## CIM Configuration Examples
The CIM configuration file defines the specific simulation parameters that simulate the compute-in-memory effects. See [`experiments/llm-cim/configs/`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-cim/configs/) for example configurations.

### Typical SRAM CIM Configuration

Here is an example of the SRAM CIM configuration file, the parameters settings are originally from the paper [A 28nm 192.3TFLOPS/W Accurate/Approximate Dual-Mode-Transpose Digital 6T-SRAM CIM Macro for Floating-Point Edge Training and Inference](https://ieeexplore.ieee.org/abstract/document/10904659).
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

## Evaluation

- **Models**:
Support ViT model family from huggingface:
e.g.`google/vit-base-patch16-224`.

- **Datasets**:
Support ImageNet dataset (requires custom path).

- **Evaluation Metrics**:
The evaluation provides comprehensive metric is Top-1 accuracy.
