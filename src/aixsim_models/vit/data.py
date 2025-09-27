from chop.dataset import MaseDataModule, get_dataset_info
import torch
from torch.utils.data import Dataset
from .arg_manager import ArgData, ArgModel, ArgTraining
from .arg_manager import TaskArguments


class VisionDataset(Dataset):
    """Wrapper dataset for HuggingFace Trainer compatibility."""
    
    def __init__(self, dataset, processor=None):
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
        
        # Convert to HuggingFace format
        # Check if image is already a preprocessed tensor
        if isinstance(image, torch.Tensor):
            # Image is already preprocessed by MaseDataModule
            pixel_values = image
        else:
            # Raw image - use processor
            if self.processor is not None:
                inputs = self.processor(image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].squeeze(0)
            else:
                # Convert PIL/numpy to tensor if needed
                if hasattr(image, 'shape'):  # numpy array
                    pixel_values = torch.from_numpy(image).float()
                else:  # PIL image
                    from torchvision import transforms
                    transform = transforms.ToTensor()
                    pixel_values = transform(image)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }
def load_dataset(task_args: TaskArguments, model_args: ArgModel, data_args: ArgData, training_args: ArgTraining, logger):
    """Load and prepare dataset."""
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    
    # Get dataset info
    dataset_info = get_dataset_info(data_args.dataset_name)
    num_classes = dataset_info.num_classes
    
    # Load dataset
    data_module = MaseDataModule(
        name=data_args.dataset_name,
        batch_size=training_args.per_device_train_batch_size if task_args.do_train else training_args.per_device_eval_batch_size,
        num_workers=data_args.num_workers,
        tokenizer=None,
        max_token_len=512,
        load_from_cache_file=True,
        model_name=model_args.model_name_or_path,
        custom_path=data_args.custom_path if data_args.dataset_name == "imagenet" else None,
    )
    data_module.prepare_data()
    data_module.setup()
    
    # Get train and eval datasets
    train_dataset = None
    eval_dataset = None
    
    if task_args.do_train:
        raw_train_dataset = data_module.train_dataset
        if data_args.max_train_samples is not None:
            indices = torch.randperm(len(raw_train_dataset))[:data_args.max_train_samples]
            raw_train_dataset = torch.utils.data.Subset(raw_train_dataset, indices)
        
        # Wrap in VisionDataset for HuggingFace compatibility
        # Don't pass processor since MaseDataModule already preprocesses
        train_dataset = VisionDataset(raw_train_dataset, processor=None)
        logger.info(f"Training dataset: {len(train_dataset)} samples")
    
    if task_args.do_eval or task_args.do_predict:
        raw_eval_dataset = data_module.val_dataset
        if data_args.max_eval_samples is not None:
            indices = torch.randperm(len(raw_eval_dataset))[:data_args.max_eval_samples]
            raw_eval_dataset = torch.utils.data.Subset(raw_eval_dataset, indices)
        
        # Wrap in VisionDataset for HuggingFace compatibility
        # Don't pass processor since MaseDataModule already preprocesses
        eval_dataset = VisionDataset(raw_eval_dataset, processor=None)
        logger.info(f"Evaluation dataset: {len(eval_dataset)} samples")
    
    logger.info(f"Dataset loaded: {num_classes} classes")
    return train_dataset, eval_dataset, num_classes, dataset_info
