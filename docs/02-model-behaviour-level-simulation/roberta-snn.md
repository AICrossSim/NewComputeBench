# RoBERTa Spiking Neural Network Transformer

This tutorial demonstrates how to apply spiking neural network (SNN) transformer modifications to RoBERTa models for sequence classification tasks. The SNN transformation converts a quantization-aware RoBERTa model into a spiking equivalent by replacing activation functions with ST-BIF spiking neurons and wrapping linear/attention layers with spike-compatible operations.

## Overview

- **SNN Transform**: Applies a two-stage conversion to RoBERTa models — first quantizing with LSQ Integer layers, then converting to SNN equivalents with ST-BIF spiking neurons.
    - The entry point is at [`experiments/roberta-snn-transformer/run_glue.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/roberta-snn-transformer/run_glue.py).
- **SNN Roberta**: Custom SNN attention and linear layers that replace standard modules with spiking neuron-based equivalents.
    - Implemented in [`src/aixsim_models/snn/fine_tune/snn_roberta.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/src/aixsim_models/snn/fine_tune/snn_roberta.py).
- **GLUE Task Support**: Fine-tune and evaluate SNN RoBERTa models on GLUE benchmark tasks.
    - Configuration scripts available at [`experiments/roberta-snn-transformer/finetune_base.sh`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/roberta-snn-transformer/finetune_base.sh).

The SNN transformation uses a two-stage pipeline:

1. **Quantization stage**: Replaces attention, output, intermediate, and classifier layers with LSQ Integer quantized equivalents.
2. **SNN conversion stage**: Converts quantized layers to spiking equivalents using `zip_tf` (SpikeZip-TF) transforms for attention blocks, and `unfold_bias` linear layers with ST-BIF neurons for fully connected layers. Embeddings and LayerNorm layers are wrapped with SpikeZip-TF passthrough. ReLU activations are replaced with identity, and LSQ Integer quantizers are replaced with ST-BIF spiking nodes.

!!! note "ReLU-based Base Model"
    The SNN conversion requires a RoBERTa model with **ReLU activations** (not GELU). Pre-trained ReLU-RoBERTa checkpoints are available at `JeffreyWong/roberta-base-relu-{task}` on HuggingFace.

## Environment Setup

!!! info "Environment Setup?"

    If you have not set up environments, please follow the guidelines in [Environment Setup](../env-setup.md).

## SNN Transform Configuration

The SNN transformer behaviour is controlled through a YAML configuration file that specifies two transformation passes: a quantization pass and an SNN conversion pass.

### Configuration Structure

The transform configuration contains three top-level keys:

- `quantization_config`: Applies LSQ Integer quantization layer-by-layer using regex matching on module names.
- `snn_transformer_config_attn`: Converts quantized attention layers to SNN equivalents using regex matching.
- `snn_transformer_config_fc`: Converts all remaining layer types (embedding, LayerNorm, linear, ReLU, LSQ Integer) to SNN equivalents by type.

### Quantization Parameters

Each quantized layer is configured with:

- `name`: Quantization method (`lsqinteger`)
- `level`: Number of quantization levels (default: 32)

### SNN Conversion Parameters

Each SNN layer is configured with:

- `name`: Conversion type — `zip_tf` for attention/embedding/LayerNorm, `unfold_bias` for linear layers, `st_bif` for quantizer nodes, `identity` for ReLU
- `level`: Spike resolution level (default: 32, used by `unfold_bias`)
- `neuron_type`: Spiking neuron model (default: `ST-BIF`)

### Default Configuration

```yaml
# experiments/roberta-snn-transformer/transform_cfg.yaml
quantization_config:
  by: regex
  'roberta\.encoder\.layer\.\d+\.attention\.self':
    config:
      name: lsqinteger
      level: 32
  'roberta\.encoder\.layer\.\d+\.attention\.output':
    config:
      name: lsqinteger
      level: 32
  'roberta\.encoder\.layer\.\d+\.output':
    config:
      name: lsqinteger
      level: 32
  'roberta\.encoder\.layer\.\d+\.intermediate':
    config:
      name: lsqinteger
      level: 32
  classifier:
    config:
      name: lsqinteger
      level: 32

snn_transformer_config_attn:
  by: regex
  'roberta\.encoder\.layer\.\d+\.attention\.self':
    config:
      name: zip_tf
      level: 32
      neuron_type: ST-BIF

snn_transformer_config_fc:
  by: type
  embedding:
    config:
      name: zip_tf
  layernorm:
    config:
      name: zip_tf
  linear:
    config:
      name: unfold_bias
      level: 32
      neuron_type: ST-BIF
  relu:
    manual_instantiate: true
    config:
      name: identity
  lsqinteger:
    manual_instantiate: true
    config:
      name: st_bif
```

## Fine-tuning RoBERTa with SNN Transform

### Single Task Fine-tuning

Fine-tune an SNN RoBERTa model on a specific GLUE task:

```bash
cd experiments/roberta-snn-transformer

# Set task parameters
TASK_NAME="mrpc"                                              # GLUE task (mrpc, sst2, cola, etc.)
MODEL_NAME="JeffreyWong/roberta-base-relu-${TASK_NAME}"      # ReLU-based RoBERTa model
LEARNING_RATE="2e-5"                                         # Learning rate
BATCH_SIZE="64"                                              # Batch size
NUM_EPOCHS="10"                                              # Training epochs
TRANSFORM_CONFIG="./transform_cfg.yaml"                      # SNN transform config

python run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --output_dir ./output/${TASK_NAME}_snn \
    --overwrite_output_dir \
    --transform_config ${TRANSFORM_CONFIG} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --logging_steps 50 \
    --seed 42
```

### Evaluation Only

Evaluate a pre-trained or fine-tuned SNN RoBERTa model without training.

**Evaluate the baseline ANN model** (transform applied, no fine-tuned weights):

```bash
python run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_eval \
    --max_seq_length 128 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --output_dir ./output/${TASK_NAME}_eval \
    --transform_config ${TRANSFORM_CONFIG} \
    --overwrite_output_dir
```

**Evaluate with fine-tuned SNN weights**:

```bash
python run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_eval \
    --max_seq_length 128 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --output_dir ./output/${TASK_NAME}_eval \
    --transform_config ${TRANSFORM_CONFIG} \
    --model_weights_path ./output/${TASK_NAME}_snn \
    --overwrite_output_dir
```

### Converting to Spiking Equivalent

To convert the transformed model to its full spiking form and save it:

```bash
python run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_eval \
    --max_seq_length 128 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --output_dir ./output/${TASK_NAME}_eval \
    --transform_config ${TRANSFORM_CONFIG} \
    --convert_to_snn \
    --overwrite_output_dir
```

## Baseline Comparison

To evaluate the original ReLU-RoBERTa model without any SNN transformation:

```bash
python run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --output_dir ./output/${TASK_NAME}_baseline \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch
```

## Results

**MRPC results** (post-transform evaluation vs. SNN-aware fine-tuning):

| Model                          | Accuracy | F1     | Combined Score |
|-------------------------------|----------:|--------:|---------------:|
| Post-transform (no fine-tune)  | 0.5613   | 0.6551 | 0.6082         |
| SNN fine-tuned (10 epochs)     | 0.7819   | 0.8468 | 0.8143         |

**Takeaways**:
- Applying the SNN transform directly without fine-tuning causes a significant accuracy drop, as the integer-quantized weights are not yet adapted to the spiking regime.
- SNN-aware fine-tuning (10 epochs) substantially recovers accuracy, approaching the performance of the quantized ANN baseline.
- The ST-BIF neuron model enables effective gradient flow during fine-tuning via a straight-through estimator.
