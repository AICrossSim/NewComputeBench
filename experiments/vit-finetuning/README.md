# ViT Fine-tuning Experiment Scripts

This directory contains Python scripts for running ViT evaluation and fine-tuning experiments using HuggingFace Trainer and structured argument parsing.

## Scripts Overview

### 1. `run_vit.py` - Main ViT Training and Evaluation Script
The main script that handles both training and evaluation using HuggingFace's structured argument parsing.

**Core Features:**
- Uses HfArgumentParser with structured dataclasses
- HuggingFace Trainer integration for robust training/evaluation
- Support for CIM transformations
- Comprehensive metrics and logging
- Compatible with all ViT model variants

### 2. `run_evaluation.sh` - Evaluation Shell Script
Shell script that demonstrates evaluation usage with proper HuggingFace-style arguments.

**Current Configuration:**
```bash
python experiments/vit-finetuning/run_vit.py \
    --model_name_or_path google/vit-base-patch16-224 \
    --dataset_name imagenet \
    --do_eval \
    --output_dir ./eval_results \
    --per_device_eval_batch_size 64 \
    --custom_path /data/datasets/imagenet_pytorch/
```

**Key Arguments:**
- `--model_name_or_path`: HuggingFace model identifier or local path
- `--dataset_name`: Dataset to use (cifar10, cifar100, imagenet, etc.)
- `--do_eval`: Enable evaluation mode
- `--do_train`: Enable training mode  
- `--do_predict`: Enable prediction mode
- `--output_dir`: Output directory for results
- `--per_device_eval_batch_size`: Batch size per device for evaluation
- `--per_device_train_batch_size`: Batch size per device for training
- `--custom_path`: Custom dataset path (for imagenet)

### 2. `run_finetuning.sh` - Single Model Fine-tuning
Fine-tunes a ViT model on a dataset.

**Usage:**
```bash
./run_finetuning.sh [model_name] [dataset_name] [num_epochs] [batch_size] [learning_rate] [output_dir]
```

**Parameters:**
- `model_name` (default: "vit-base"): Model to fine-tune
- `dataset_name` (default: "cifar10"): Dataset to use
- `num_epochs` (default: 5): Number of training epochs
- `batch_size` (default: 32): Training batch size
- `learning_rate` (default: 5e-5): Learning rate
- `output_dir` (default: "./finetune_outputs"): Output directory

**Examples:**
```bash
# Basic fine-tuning
./run_finetuning.sh

# Custom configuration
./run_finetuning.sh deit-small cifar100 10 16 1e-4 ./my_experiments
```

### 3. `run_quick_test.sh` - Development Testing
Runs quick tests to verify the ViT module is working correctly.

**Usage:**
```bash
./run_quick_test.sh
```

This script:
- Tests evaluation with limited samples
- Tests fine-tuning with 1 epoch and minimal data
- Tests backward compatibility with the original API
- Ideal for development and CI testing

### 4. `run_experiments.sh` - Comprehensive Experiments
Runs experiments across multiple models and datasets.

**Usage:**
```bash
./run_experiments.sh
```

**Default Configuration:**
- Models: vit-base, deit-small, deit-base
- Datasets: cifar10, cifar100
- Epochs: 10 per experiment
- Batch size: 32

This script:
- Runs all model-dataset combinations
- Saves detailed results for each experiment
- Generates comprehensive reports
- Creates timestamped output directories

## Supported Models

The scripts support various ViT-family models:

**ViT Models:**
- `vit-base` - ViT Base (86M parameters)
- `vit-large` - ViT Large (307M parameters)
- `vit-base-384` - ViT Base with 384x384 input

**DeiT Models:**
- `deit-tiny` - DeiT Tiny (5M parameters)
- `deit-small` - DeiT Small (22M parameters)
- `deit-base` - DeiT Base (86M parameters)

**Other Models:**
- `beit-base` - BEiT Base
- `swin-tiny` - Swin Transformer Tiny
- `swin-small` - Swin Transformer Small

## Supported Datasets

- `cifar10` - CIFAR-10 (10 classes, 32x32 images)
- `cifar100` - CIFAR-100 (100 classes, 32x32 images)
- `imagenet` - ImageNet (1000 classes, 224x224 images)

## Requirements

1. **Environment Setup:**
   ```bash
   # Ensure you're in the project root
   cd /path/to/NewComputeBench
   
   # Make scripts executable (if needed)
   chmod +x experiments/vit-finetuning/*.sh
   ```

2. **Python Dependencies:**
   - PyTorch
   - Transformers
   - chop (for datasets)
   - All dependencies from the main project

3. **Hardware:**
   - GPU recommended for fine-tuning
   - CPU-only works but will be slower

## Output Structure

Each script creates organized output directories:

```
output_directory/
├── experiment_summary.json    # High-level results
├── training_results.json      # Detailed training metrics
├── pytorch_model.bin          # Fine-tuned model weights
├── config.json               # Model configuration
├── experiment.log            # Detailed logs
└── runs/                     # TensorBoard logs
```

## Monitoring Training

The scripts enable TensorBoard logging by default:

```bash
# View training progress
tensorboard --logdir ./outputs/your_experiment/runs
```

## Quick Start

1. **Test the setup:**
   ```bash
   ./run_quick_test.sh
   ```

2. **Run a simple evaluation:**
   ```bash
   ./run_evaluation.sh vit-base cifar10
   ```

3. **Run a quick fine-tuning:**
   ```bash
   ./run_finetuning.sh deit-small cifar10 3
   ```

4. **Run comprehensive experiments:**
   ```bash
   ./run_experiments.sh
   ```

## Troubleshooting

**Common Issues:**

1. **Import errors:** Ensure you're running from the project root
2. **CUDA errors:** Check GPU availability and memory
3. **Dataset errors:** First run may download datasets automatically
4. **Permission errors:** Make scripts executable with `chmod +x *.sh`

**Debug Mode:**
Add `set -x` to any script for verbose debugging output.

## Customization

You can modify the scripts to:
- Add new models or datasets
- Change default hyperparameters
- Add custom evaluation metrics
- Integrate with experiment tracking systems (Weights & Biases)

## Performance Tips

- Use `fp16=True` for faster training on modern GPUs
- Increase `num_workers` for faster data loading
- Use smaller models (deit-tiny, deit-small) for quick experimentation
- Limit samples with `max_train_samples` and `max_eval_samples` for testing
