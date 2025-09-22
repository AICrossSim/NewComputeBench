"""
ViT model flavor definitions and utilities.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Mapping of model names to HuggingFace model identifiers
HUGGINGFACE_MODEL_MAP = {
    # ViT Base models
    "vit-base": "google/vit-base-patch16-224",
    "vit-base-224": "google/vit-base-patch16-224",
    "vit-base-384": "google/vit-base-patch16-384",
    "vit-base-patch32": "google/vit-base-patch32-224",
    "vit-base-patch32-224": "google/vit-base-patch32-224",
    "vit-base-patch32-384": "google/vit-base-patch32-384",
    
    # ViT Large models
    "vit-large": "google/vit-large-patch16-224",
    "vit-large-224": "google/vit-large-patch16-224",
    "vit-large-384": "google/vit-large-patch16-384",
    "vit-large-patch32": "google/vit-large-patch32-224",
    "vit-large-patch32-224": "google/vit-large-patch32-224",
    "vit-large-patch32-384": "google/vit-large-patch32-384",
    
    # ViT Huge models
    "vit-huge": "google/vit-huge-patch14-224",
    "vit-huge-224": "google/vit-huge-patch14-224",
    
    # DeiT models (Data-efficient Image Transformers)
    "deit-tiny": "facebook/deit-tiny-patch16-224",
    "deit-small": "facebook/deit-small-patch16-224", 
    "deit-base": "facebook/deit-base-patch16-224",
    "deit-base-384": "facebook/deit-base-patch16-384",
    
    # DeiT distilled models
    "deit-tiny-distilled": "facebook/deit-tiny-distilled-patch16-224",
    "deit-small-distilled": "facebook/deit-small-distilled-patch16-224",
    "deit-base-distilled": "facebook/deit-base-distilled-patch16-224",
    "deit-base-distilled-384": "facebook/deit-base-distilled-patch16-384",
    
    # BEiT models (BERT Pre-Training of Image Transformers)
    "beit-base": "microsoft/beit-base-patch16-224",
    "beit-base-384": "microsoft/beit-base-patch16-384",
    "beit-large": "microsoft/beit-large-patch16-224",
    "beit-large-384": "microsoft/beit-large-patch16-384",
    
    # DiNO ViT models
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base", 
    "dinov2-large": "facebook/dinov2-large",
    "dinov2-giant": "facebook/dinov2-giant",
    
    # Swin Transformer models
    "swin-tiny": "microsoft/swin-tiny-patch4-window7-224",
    "swin-small": "microsoft/swin-small-patch4-window7-224",
    "swin-base": "microsoft/swin-base-patch4-window7-224",
    "swin-base-384": "microsoft/swin-base-patch4-window12-384",
    "swin-large": "microsoft/swin-large-patch4-window7-224",
    "swin-large-384": "microsoft/swin-large-patch4-window12-384",
}


# Model configuration details
MODEL_CONFIGS = {
    "vit-base": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "patch_size": 16,
        "image_size": 224,
        "num_parameters": "86M",
    },
    "vit-large": {
        "hidden_size": 1024,
        "num_hidden_layers": 24, 
        "num_attention_heads": 16,
        "patch_size": 16,
        "image_size": 224,
        "num_parameters": "307M",
    },
    "vit-huge": {
        "hidden_size": 1280,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "patch_size": 14,
        "image_size": 224,
        "num_parameters": "632M",
    },
    "deit-tiny": {
        "hidden_size": 192,
        "num_hidden_layers": 12,
        "num_attention_heads": 3,
        "patch_size": 16,
        "image_size": 224,
        "num_parameters": "5M",
    },
    "deit-small": {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "patch_size": 16,
        "image_size": 224,
        "num_parameters": "22M",
    },
    "deit-base": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "patch_size": 16,
        "image_size": 224,
        "num_parameters": "86M",
    },
}


def get_huggingface_model_name(model_name: str) -> str:
    """
    Get the HuggingFace model identifier for a given model name.
    
    Args:
        model_name: Local model name (e.g., 'vit-base', 'deit-small')
        
    Returns:
        HuggingFace model identifier
        
    Raises:
        KeyError: If model name is not found
    """
    if model_name in HUGGINGFACE_MODEL_MAP:
        hf_name = HUGGINGFACE_MODEL_MAP[model_name]
        logger.info(f"Mapped model '{model_name}' to HuggingFace model '{hf_name}'")
        return hf_name
    else:
        # Check if it's already a HuggingFace model name
        if "/" in model_name:
            logger.info(f"Using HuggingFace model name as-is: '{model_name}'")
            return model_name
        else:
            available_models = list(HUGGINGFACE_MODEL_MAP.keys())
            raise KeyError(
                f"Unknown model name: '{model_name}'. "
                f"Available models: {available_models}"
            )


def get_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get model configuration details.
    
    Args:
        model_name: Model name
        
    Returns:
        Model configuration dictionary or None if not found
    """
    base_name = model_name.split("-")[0] + "-" + model_name.split("-")[1] if "-" in model_name else model_name
    return MODEL_CONFIGS.get(base_name)


def list_available_models() -> Dict[str, str]:
    """
    List all available model names and their HuggingFace identifiers.
    
    Returns:
        Dictionary mapping model names to HuggingFace identifiers
    """
    return HUGGINGFACE_MODEL_MAP.copy()


def get_model_family(model_name: str) -> str:
    """
    Get the model family for a given model name.
    
    Args:
        model_name: Model name
        
    Returns:
        Model family (e.g., 'vit', 'deit', 'swin')
    """
    if model_name.startswith("vit"):
        return "vit"
    elif model_name.startswith("deit"):
        return "deit"
    elif model_name.startswith("beit"):
        return "beit"
    elif model_name.startswith("dinov2"):
        return "dinov2"
    elif model_name.startswith("swin"):
        return "swin"
    else:
        return "unknown"


def get_recommended_batch_size(model_name: str, image_size: int = 224) -> Dict[str, int]:
    """
    Get recommended batch sizes for different scenarios.
    
    Args:
        model_name: Model name
        image_size: Image size
        
    Returns:
        Dictionary with recommended batch sizes for different memory configurations
    """
    family = get_model_family(model_name)
    size = "small" if "tiny" in model_name or "small" in model_name else \
           "large" if "large" in model_name or "huge" in model_name else "base"
    
    # Base recommendations for 224x224 images
    base_sizes = {
        ("vit", "small"): {"8gb": 64, "16gb": 128, "24gb": 256},
        ("vit", "base"): {"8gb": 32, "16gb": 64, "24gb": 128},
        ("vit", "large"): {"8gb": 16, "16gb": 32, "24gb": 64},
        ("deit", "small"): {"8gb": 128, "16gb": 256, "24gb": 512},
        ("deit", "base"): {"8gb": 64, "16gb": 128, "24gb": 256},
        ("swin", "small"): {"8gb": 32, "16gb": 64, "24gb": 128},
        ("swin", "base"): {"8gb": 16, "16gb": 32, "24gb": 64},
        ("swin", "large"): {"8gb": 8, "16gb": 16, "24gb": 32},
    }
    
    recommended = base_sizes.get((family, size), {"8gb": 32, "16gb": 64, "24gb": 128})
    
    # Adjust for image size
    if image_size > 224:
        scale_factor = (224 / image_size) ** 2
        recommended = {k: max(1, int(v * scale_factor)) for k, v in recommended.items()}
    
    return recommended


def validate_model_name(model_name: str) -> bool:
    """
    Validate if a model name is supported.
    
    Args:
        model_name: Model name to validate
        
    Returns:
        True if model is supported, False otherwise
    """
    try:
        get_huggingface_model_name(model_name)
        return True
    except KeyError:
        return False


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a model.
    
    Args:
        model_name: Model name
        
    Returns:
        Dictionary with model information
    """
    try:
        hf_name = get_huggingface_model_name(model_name)
        config = get_model_config(model_name)
        family = get_model_family(model_name)
        batch_sizes = get_recommended_batch_size(model_name)
        
        return {
            "model_name": model_name,
            "huggingface_name": hf_name,
            "family": family,
            "config": config,
            "recommended_batch_sizes": batch_sizes,
            "is_supported": True,
        }
    except KeyError:
        return {
            "model_name": model_name,
            "is_supported": False,
            "error": f"Model '{model_name}' is not supported",
        }
