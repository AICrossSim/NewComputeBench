# RoBERTa PIM Simulation

This tutorial demonstrates how to apply Processing-in-Memory (PIM) transformations to RoBERTa models for sequence classification tasks. The PIM simulation supports various technologies including SRAM (digital), RRAM (analogue), and PCM (analogue). The evaluation framework similar to the [RoBERTa Optical Transformer](roberta-onn.md). This tutorial will focus on introducing the PIM simulation framework.

## Overview

- **PIM-aware Fine-tuning**: Applies PIM transformation (noise injection and quantization) to RoBERTa models and fine-tunes them on GLUE benchmark tasks.
    - Supported Technologies: SRAM (Digital), RRAM/PCM (Analogue)
        - SRAM (Digital): Simulates digital PIM follows the digital circuit-aware quantization (e.g., FP8).
        - RRAM/PCM (Analogue): Simulates analogue PIM with non-ideal effects like noise and variation.
    - The entry point is at [`experiments/roberta-pim/run_glue.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/roberta-pim/run_glue.py).

## Environment Setup

!!! info "Environment Setup"
    If you have not set up environments, please follow the guidelines in [Environment Setup](../env-setup.md).

## PIM Configuration

PIM behavior is controlled through YAML configuration files that specify technology-specific parameters.

### Configuration Examples

Example configurations can be found in `experiments/llm-pim/configs/`:
- `sram.yaml`: Digital PIM (FP8)
- `reram.yaml`: Analogue RRAM
- `pcm.yaml`: Analogue PCM

### Typical SRAM PIM Configuration (FP8)

```yaml
# experiments/llm-pim/configs/sram.yaml
by: "type"
linear:
  config:
    tile_type: "digital"
    core_size: 64
    rescale_dim: "vector"
    x_quant_type: "e4m3"
    weight_quant_type: "e4m3"
```
The transform configuration includes the following parameters:

- `tile_type`: The type of tile to use for the matrix multiplication operation ("digital", "reram", "pcm")
- `core_size`: The size of the core of a simulation tile (default: 64)

The other parameters are related to the specific PIM technology:
- **SRAM (Digital)**:
    - `rescale_dim`: Specifies the granularity of the pre-alignment mechanism ("vector" or "matrix"). This simulates the vector-wise pre-alignment scheme used in digital CIM macros to support diverse precision formats.
    - `x_quant_type`: Specifies the quantization format for input activations (e.g., `"e4m3"`, `"e5m2"`). This models the signed fixed-point mantissa encoding for formats like FP8 and BF16.
    - `weight_quant_type`: Specifies the quantization format for model weights, following the same convention as `x_quant_type`.
- **ReRAM (Analogue)**:
    - `noise_magnitude`: Defines the magnitude of the Gaussian distribution used to model analog nonidealities, such as device variation and noise, within the ReRAM crossbar arrays where weights are stored as conductance values.
    - `num_bits`: The resolution (number of bits) used for weight representation in the ReRAM cells.
- **PCM (Analogue)**:
    - `programming_noise`: Simulates programming errors and device-to-device variability during the weight-setting process.
    - `read_noise`: Models temporal read noise and 1/f noise during the inference phase.
    - `ir_drop`: Simulates the voltage drop across the crossbar interconnects, which introduces spatial nonidealities in the MAC operations.
    - `out_noise`: Models additive system noise and nonlinearities in the peripheral circuits, including limited input/output dynamic range.

## Fine-tuning RoBERTa with PIM

### Run PIM-aware Fine-tuning on GLUE

```bash
# Set task parameters
TASK_NAME="mrpc"                                    # GLUE task (mrpc, sst2, cola, etc.)
MODEL_NAME="FacebookAI/roberta-base"                # Base model
PIM_CONFIG="./experiments/llm-pim/configs/sram.yaml" # PIM config (sram, reram, or pcm)

python experiments/roberta-pim/run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir ./output/${TASK_NAME}_pim \
    --pim_config_path ${PIM_CONFIG} \
    --pim
```

## Results

Below are the results for RoBERTa-base on the GLUE benchmark across different PIM technology types.

### Post-training Transform Results
(Applying PIM transformation to a trained RoBERTa model without further fine-tuning)

| Model | MNLI (Acc) | QNLI (Acc) | RTE (Acc) | SST (Acc) | MRPC (Acc) | CoLA (Matt) | QQP (Acc) | STSB (P/S) | Avg |
|-------|-----------:|-----------:|----------:|----------:|-----------:|------------:|----------:|-----------:|----:|
| Original | 0.8728 | 0.9244 | 0.7978 | 0.9357 | 0.9019 | 0.6232 | 0.9153 | 0.9089 | 0.8600 |
| Random | 0.3266 | 0.4946 | 0.5271 | 0.4908 | 0.3162 | 0.0000 | 0.6318 | 0.0332 | 0.3525 |
| Optical Transformer | 0.8000 | 0.7966 | 0.4801 | 0.8704 | 0.7770 | 0.2034 | 0.9075 | 0.8485 | 0.7104 |
| SqueezeLight | 0.3200 | 0.4961 | 0.4404 | 0.5126 | 0.5025 | 0.0213 | 0.5890 | -0.0543 | 0.3582 |
| **ReRAM - analogue** | 0.3239 | 0.4860 | 0.5090 | 0.4839 | 0.4338 | -0.0363 | 0.5796 | -0.0011 | 0.3475 |
| **PCM - analogue** | 0.3211 | 0.5123 | 0.5090 | 0.5068 | 0.5098 | 0.0443 | 0.6162 | 0.0745 | 0.3872 |
| **SRAM - digital (fp8)** | 0.7825 | 0.4939 | 0.5271 | 0.5092 | 0.3186 | 0.0000 | 0.6318 | 0.0198 | 0.3523 |

### Fine-tuning Results
(PIM-aware fine-tuning on a trained RoBERTa model)

| Model | MNLI (Acc) | QNLI (Acc) | RTE (Acc) | SST (Acc) | MRPC (Acc) | CoLA (Matt) | QQP (Acc) | STSB (P/S) | Avg |
|-------|-----------:|-----------:|----------:|----------:|-----------:|------------:|----------:|-----------:|----:|
| Original | 0.8728 | 0.9244 | 0.7978 | 0.9357 | 0.9019 | 0.6232 | 0.9153 | 0.9089 | 0.8600 |
| Random | 0.3266 | 0.4946 | 0.5271 | 0.4908 | 0.3162 | 0.0000 | 0.6318 | 0.0332 | 0.3525 |
| Optical Transformer | 0.8510 | 0.9032 | 0.5813 | 0.9140 | 0.8677 | 0.4441 | 0.9060 | 0.0332 | 0.6876 |
| SqueezeLight | 0.3212 | 0.4961 | 0.4676 | 0.5131 | 0.5025 | 0.0000 | 0.5932 | 0.0514 | 0.3681 |
| **RRAM - analogue** | 0.8416 | 0.8669 | 0.5451 | 0.8761 | 0.6961 | 0.0000 | 0.9052 | 0.3611 | 0.7228 |
| **PCM - analogue** | 0.6850 | 0.7681 | 0.4729 | 0.8383 | 0.6838 | 0.0000 | 0.8585 | -0.1037 | 0.6108 |
| **SRAM - digital (fp8)** | 0.8589 | 0.9129 | 0.6643 | 0.9220 | 0.8505 | 0.5113 | 0.9128 | 0.8815 | 0.8143 |
