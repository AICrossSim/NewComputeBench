#!/usr/bin/env python3
"""
Comprehensive ViT Training and Evaluation Script - HuggingFace Style

This script provides comprehensive training and evaluation capabilities for Vision Transformer models,
following HuggingFace patterns similar to run_glue.py or run_image_classification.py.

"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import yaml
from transformers import (
    ViTForImageClassification, 
    ViTImageProcessor, 
    Trainer, 
    TrainingArguments,
    EvalPrediction,
    HfArgumentParser
)

from .utils import compute_accuracy, load_checkpoint, get_model_size
from .model_flavors import get_huggingface_model_name
from .arg_manager import ArgJob, ArgModel, ArgData, ArgTraining, ArgEvaluation, ArgOptimizer, ArgMetrics, TaskArguments
from .data import load_dataset
from chop.dataset import get_dataset_info

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments using HfArgumentParser and arg_manager dataclasses."""
    parser = HfArgumentParser((
        TaskArguments,
        ArgModel, 
        ArgData, 
        ArgTraining, 
        ArgOptimizer, 
        ArgEvaluation, 
        ArgMetrics,
        ArgJob
    ))
    
    # Parse arguments from command line
    task_args, model_args, data_args, training_args, optimizer_args, eval_args, metrics_args, job_args = parser.parse_args_into_dataclasses()
    
    return task_args, model_args, data_args, training_args, optimizer_args, eval_args, metrics_args, job_args

def setup_logging(job_args: ArgJob):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    if job_args.print_args:
        logger.info("=" * 50)
        logger.info("Arguments:")
        logger.info("=" * 50)
        # Print all arguments
        logger.info("=" * 50)
    
    return logger


def compute_metrics(eval_pred: EvalPrediction):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Compute accuracy
    accuracy = (predictions == labels).mean()
    
    # Compute per-class accuracy if needed
    unique_labels = np.unique(labels)
    class_accuracies = {}
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 0:
            class_acc = (predictions[mask] == labels[mask]).mean()
            class_accuracies[f'class_{label}_accuracy'] = class_acc
    
    metrics = {
        'accuracy': accuracy,
        'eval_accuracy': accuracy,  # Also provide eval_accuracy for consistency
        **class_accuracies
    }
    
    return metrics


def load_model_and_processor(task_args: TaskArguments, model_args: ArgModel, data_args: ArgData, training_args: ArgTraining, logger):
    """Load model and image processor."""
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    
    # Determine device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if training_args.bf16:
        dtype = torch.bfloat16
    elif training_args.fp16:
        dtype = torch.float16
    
    # Get dataset info to determine number of classes
    dataset_info = get_dataset_info(data_args.dataset_name)
    num_classes = model_args.num_labels or dataset_info.num_classes
    
    # Load model
    if os.path.isdir(model_args.model_name_or_path):
        # Local checkpoint
        logger.info("Loading from local checkpoint")
        model = ViTForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=dtype,
            num_labels=num_classes,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes
        )
        processor = ViTImageProcessor.from_pretrained(model_args.model_name_or_path)
    else:
        # Try to get HuggingFace model name
        try:
            hf_model_name = get_huggingface_model_name(model_args.model_name_or_path)
            logger.info(f"Loading from HuggingFace Hub: {hf_model_name}")
            model = ViTForImageClassification.from_pretrained(
                hf_model_name,
                cache_dir=model_args.cache_dir,
                torch_dtype=dtype,
                num_labels=num_classes,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes
            )
            processor = ViTImageProcessor.from_pretrained(hf_model_name)
        except:
            # Fall back to direct model name
            logger.info(f"Loading from HuggingFace Hub: {model_args.model_name_or_path}")
            model = ViTForImageClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                torch_dtype=dtype,
                num_labels=num_classes,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes
            )
            processor = ViTImageProcessor.from_pretrained(model_args.model_name_or_path)
    
    # Apply model modifications
    if model_args.freeze_backbone:
        # Freeze all parameters except classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        logger.info("Backbone frozen, only classifier trainable")
    elif model_args.freeze_embeddings:
        # Freeze only embedding layers
        for name, param in model.named_parameters():
            if 'embeddings' in name:
                param.requires_grad = False
        logger.info("Embedding layers frozen")
    
    model = model.to(device)
    
    # Apply CIM transform if requested
    if task_args.enable_cim_transform:
        if task_args.cim_config_path is None:
            raise ValueError("--cim_config_path is required when --enable_cim_transform is set")
        
        logger.info(f"Applying CIM transform with config: {task_args.cim_config_path}")
        with open(task_args.cim_config_path, 'r') as f:
            cim_config = yaml.safe_load(f)
        
        # Import and apply CIM transform
        try:
            from aixsim_models.cim.module_level_transform import module_level_transform
            model = module_level_transform(model, cim_config)
        except ImportError:
            logger.warning("CIM transform not available, skipping...")
    
    logger.info(f"Model loaded on {device} with dtype {dtype}")
    
    # Print model info
    total_params, trainable_params = get_model_size(model)
    logger.info(f"Model size: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    return model, processor, device




def create_trainer(model, train_dataset, eval_dataset, training_args, compute_metrics_fn=None, task_args: TaskArguments = None):
    """Create and configure HuggingFace Trainer."""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if task_args.do_eval is not None else None,
        compute_metrics=compute_metrics_fn if task_args.do_eval is not None else None,
    )
    
    return trainer


def compute_additional_metrics(model, eval_dataloader, device, predictions, labels, eval_args: ArgEvaluation, logger):
    """Compute additional evaluation metrics."""
    additional_results = {}
    
    # Per-class accuracies
    if eval_args.compute_class_accuracies:
        logger.info("Computing per-class accuracies...")
        # Convert predictions and labels to tensors
        pred_tensor = torch.tensor(predictions)
        label_tensor = torch.tensor(labels)
        
        # Compute per-class accuracy
        num_classes = len(np.unique(labels))
        class_accuracies = {}
        
        for class_idx in range(num_classes):
            class_mask = (label_tensor == class_idx)
            if class_mask.sum() > 0:
                class_pred = pred_tensor[class_mask]
                class_correct = (class_pred == class_idx).sum().item()
                class_total = class_mask.sum().item()
                class_accuracy = class_correct / class_total * 100
                class_accuracies[f'class_{class_idx}_accuracy'] = class_accuracy
        
        additional_results.update(class_accuracies)
    
    # Inference speed benchmark
    if eval_args.benchmark_inference:
        logger.info("Benchmarking inference speed...")
        
        # Create dummy input for benchmarking
        dummy_input = torch.randn(
            eval_args.benchmark_batch_size, 3, 224, 224, 
            device=device
        )
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):  # Fixed warmup runs
                _ = model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(eval_args.benchmark_num_runs):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / eval_args.benchmark_num_runs * 1000  # ms
        throughput = (eval_args.benchmark_batch_size * 1000) / avg_inference_time  # samples per second
        
        additional_results.update({
            'avg_inference_time_ms': avg_inference_time,
            'throughput_samples_per_sec': throughput
        })
        
        logger.info(f"Average inference time: {avg_inference_time:.2f}ms")
        logger.info(f"Throughput: {throughput:.2f} samples/sec")
    
    return additional_results


def save_results(results, training_args: ArgTraining, logger):
    """Save evaluation results."""
    # Create output directory
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    results_file = output_dir / "eval_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")


def save_results(results, training_args: ArgTraining, logger):
    """Save evaluation results."""
    # Create output directory
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    results_file = output_dir / "eval_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")

def compute_class_accuracies(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute per-class accuracies.
    
    Args:
        model: ViT model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
        class_names: Optional class names
        
    Returns:
        Dictionary with per-class accuracies
    """
    
    model.eval()
    
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Computing class accuracies"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            _, predicted = torch.max(logits, 1)
            
            # Update class counts
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == targets[i]:
                    class_correct[label] += 1
    
    # Compute per-class accuracies
    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100.0 * class_correct[i] / class_total[i]
            class_name = class_names[i] if class_names else f"Class_{i}"
            class_accuracies[class_name] = accuracy.item()
        else:
            class_name = class_names[i] if class_names else f"Class_{i}"
            class_accuracies[class_name] = 0.0
    
    # Add summary statistics
    accuracies_list = list(class_accuracies.values())
    class_accuracies['mean_accuracy'] = np.mean(accuracies_list)
    class_accuracies['std_accuracy'] = np.std(accuracies_list)
    class_accuracies['min_accuracy'] = np.min(accuracies_list)
    class_accuracies['max_accuracy'] = np.max(accuracies_list)
    
    return class_accuracies

def benchmark_inference_speed(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: torch.device = torch.device("cuda"),
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, float]:
    """
    Benchmark inference speed of ViT model.
    
    Args:
        model: ViT model to benchmark
        input_shape: Input tensor shape (batch_size, channels, height, width)
        device: Device to run on
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        
    Returns:
        Benchmark results
    """
    
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == "cuda" else None
    
    times = []
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc="Benchmarking"):
            start_time = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Compute statistics
    times = np.array(times)
    batch_size = input_shape[0]
    
    results = {
        'mean_latency_ms': np.mean(times) * 1000,
        'std_latency_ms': np.std(times) * 1000,
        'min_latency_ms': np.min(times) * 1000,
        'max_latency_ms': np.max(times) * 1000,
        'throughput_images_per_second': batch_size / np.mean(times),
        'batch_size': batch_size,
        'num_runs': num_runs,
    }
    
    logger.info("Inference Benchmark Results:")
    for key, value in results.items():
        if 'latency' in key:
            logger.info(f"  {key}: {value:.2f}")
        elif 'throughput' in key:
            logger.info(f"  {key}: {value:.1f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return results


