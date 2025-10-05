import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from .arg_manager import ArgData, ArgModel, ArgTraining, TaskArguments


# Dataset info mapping
DATASET_INFO = {
    'imagenet': {'num_classes': 1000, 'image_size': 224},
    'cifar10': {'num_classes': 10, 'image_size': 32},
    'cifar100': {'num_classes': 100, 'image_size': 32},
}


class DatasetInfo:
    """Simple dataset info class."""
    def __init__(self, num_classes: int, image_size: int):
        self.num_classes = num_classes
        self.image_size = image_size


class VisionDataset(Dataset):
    """Wrapper dataset for HuggingFace Trainer compatibility with processor support."""
    
    def __init__(self, dataset, processor):
        """
        Args:
            dataset: Raw dataset (torchvision dataset)
            processor: HuggingFace image processor for preprocessing
        """
        self.dataset = dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, (list, tuple)) and len(item) == 2:
            image, label = item
        else:
            # Handle different dataset formats
            image = item['image'] if isinstance(item, dict) else item
            label = item['label'] if isinstance(item, dict) else 0
        
        # Use processor to preprocess the image
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }
def load_dataset(task_args: TaskArguments, model_args: ArgModel, data_args: ArgData, training_args: ArgTraining, logger, processor):
    """Load and prepare dataset using torchvision and HuggingFace processor.
    
    Args:
        processor: HuggingFace image processor for preprocessing images.
    """
    if processor is None:
        raise ValueError("processor is required for dataset loading")
    
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    
    # Get dataset info
    if data_args.dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {data_args.dataset_name}. Supported: {list(DATASET_INFO.keys())}")
    
    info = DATASET_INFO[data_args.dataset_name]
    num_classes = info['num_classes']
    dataset_info = DatasetInfo(num_classes=num_classes, image_size=info['image_size'])
    
    # Load raw datasets (no preprocessing - just convert to PIL)
    train_dataset = None
    eval_dataset = None
    
    if data_args.custom_path is None:
        raise ValueError("custom_path is required for ImageNet dataset")
    
    data_path = Path(data_args.custom_path)
    
    if task_args.do_train:
        train_dir = data_path / 'train'
        if not train_dir.exists():
            raise ValueError(f"Training directory not found: {train_dir}")
        
        raw_train_dataset = datasets.ImageFolder(
            root=str(train_dir),
            transform=transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224)])
        )
        
        if data_args.max_train_samples is not None:
            indices = torch.randperm(len(raw_train_dataset))[:data_args.max_train_samples]
            raw_train_dataset = torch.utils.data.Subset(raw_train_dataset, indices)
        
        train_dataset = VisionDataset(raw_train_dataset, processor=processor)
        logger.info(f"Training dataset: {len(train_dataset)} samples")
    
    if task_args.do_eval or task_args.do_predict:
        val_dir = data_path / 'val'
        if not val_dir.exists():
            raise ValueError(f"Validation directory not found: {val_dir}")
        
        raw_eval_dataset = datasets.ImageFolder(
            root=str(val_dir),
            transform=transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224)])
        )
        
        if data_args.max_eval_samples is not None:
            indices = torch.randperm(len(raw_eval_dataset))[:data_args.max_eval_samples]
            raw_eval_dataset = torch.utils.data.Subset(raw_eval_dataset, indices)
        
        eval_dataset = VisionDataset(raw_eval_dataset, processor=processor)
        logger.info(f"Evaluation dataset: {len(eval_dataset)} samples")
    
    logger.info(f"Dataset loaded: {num_classes} classes")
    return train_dataset, eval_dataset, num_classes, dataset_info
