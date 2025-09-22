"""
Utility functions for ViT fine-tuning and evaluation.
"""

import logging
from typing import Tuple, Dict, Any, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import transforms

logger = logging.getLogger(__name__)


def get_model_size(model: nn.Module) -> Tuple[int, int]:
    """
    Get the total and trainable parameter count of a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Tuple[float, ...]:
    """
    Compute top-k accuracy.
    
    Args:
        logits: Model logits [batch_size, num_classes]
        targets: Target labels [batch_size]
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        Tuple of top-k accuracies as percentages
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return tuple(res)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model weights from checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict matching
        
    Returns:
        Dictionary with checkpoint metadata
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict if checkpoint contains other information
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            metadata = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
        else:
            # Assume the entire checkpoint is the state dict
            state_dict = checkpoint
            metadata = {}
    else:
        state_dict = checkpoint
        metadata = {}
    
    # Load state dict into model
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
        logger.info("Checkpoint loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise
    
    # Add loading info to metadata
    metadata.update({
        'checkpoint_path': str(checkpoint_path),
        'missing_keys': missing_keys if not strict else [],
        'unexpected_keys': unexpected_keys if not strict else [],
    })
    
    return metadata


def build_vision_transform(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    is_training: bool = False,
    augment_prob: float = 0.5,
) -> transforms.Compose:
    """
    Build vision transforms for ViT models.
    
    Args:
        image_size: Target image size
        mean: Normalization mean values
        std: Normalization std values
        is_training: Whether this is for training (adds augmentations)
        augment_prob: Probability of applying augmentations
        
    Returns:
        Composed transforms
    """
    transform_list = []
    
    if is_training:
        # Training augmentations
        transform_list.extend([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=augment_prob),
            transforms.RandomRotation(degrees=10, p=augment_prob),
            transforms.ColorJitter(
                brightness=0.1, 
                contrast=0.1, 
                saturation=0.1, 
                hue=0.05,
                p=augment_prob
            ),
        ])
    else:
        # Validation/test transforms
        transform_list.extend([
            transforms.Resize((image_size, image_size)),
        ])
    
    # Common transforms
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return transforms.Compose(transform_list)


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Denormalize a tensor using ImageNet statistics.
    
    Args:
        tensor: Normalized tensor [C, H, W] or [B, C, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized tensor
    """
    if tensor.dim() == 3:
        # Single image [C, H, W]
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
    elif tensor.dim() == 4:
        # Batch of images [B, C, H, W]
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
    
    # Move to same device as tensor
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    
    return tensor * std + mean


def log_model_info(model: nn.Module, logger: Optional[logging.Logger] = None) -> None:
    """
    Log detailed model information.
    
    Args:
        model: PyTorch model
        logger: Logger instance (uses module logger if None)
    """
    if logger is None:
        logger = globals()['logger']
    
    total_params, trainable_params = get_model_size(model)
    
    logger.info("Model Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    logger.info(f"  Trainable ratio: {trainable_params / total_params * 100:.2f}%")
    
    # Log model structure summary
    logger.info("Model Structure:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f"  {name}: {module_params:,} params ({module_trainable:,} trainable)")


def freeze_parameters(model: nn.Module, patterns: Union[str, Tuple[str, ...]]) -> int:
    """
    Freeze model parameters matching given patterns.
    
    Args:
        model: PyTorch model
        patterns: String or tuple of strings to match parameter names
        
    Returns:
        Number of parameters frozen
    """
    if isinstance(patterns, str):
        patterns = (patterns,)
    
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = False
            frozen_count += param.numel()
            logger.debug(f"Frozen parameter: {name}")
    
    logger.info(f"Frozen {frozen_count:,} parameters")
    return frozen_count


def unfreeze_parameters(model: nn.Module, patterns: Union[str, Tuple[str, ...]]) -> int:
    """
    Unfreeze model parameters matching given patterns.
    
    Args:
        model: PyTorch model
        patterns: String or tuple of strings to match parameter names
        
    Returns:
        Number of parameters unfrozen
    """
    if isinstance(patterns, str):
        patterns = (patterns,)
    
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = True
            unfrozen_count += param.numel()
            logger.debug(f"Unfrozen parameter: {name}")
    
    logger.info(f"Unfrozen {unfrozen_count:,} parameters")
    return unfrozen_count
