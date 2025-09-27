from dataclasses import dataclass, field, asdict
from typing import Union, List, Optional, Literal
from transformers import TrainingArguments


@dataclass
class ArgJob:
    """
    Job level configurations.

    Args:
        dump_folder : str
            Folder to dump job outputs.
        description : str
            Description of the job.
        use_for_integration_test : bool
            Add this config to the integration test suite.
        print_args : bool
            Print the args to terminal.
    """

    dump_folder: str = "./vit_outputs"
    description: str = "default vit job"
    use_for_integration_test: bool = False
    print_args: bool = False


@dataclass
class ArgProfiling:
    """
    Profiling configurations.

    Args:
        enable_profiling : bool
            Whether to enable pytorch profiler.
        save_traces_folder : str
            Trace files location.
        profile_freq : int
            How often to collect profiler traces, in iterations.
        enable_memory_snapshot : bool
            Whether to dump memory snapshot.
        save_memory_snapshot_folder : str
            Memory snapshot files location.
    """

    enable_profiling: bool = False
    save_traces_folder: str = "profile_traces"
    profile_freq: int = 10
    enable_memory_snapshot: bool = False
    save_memory_snapshot_folder: str = "memory_snapshot"


@dataclass
class ArgMetrics:
    """
    Metrics configurations.

    Args:
        log_freq : int
            How often to log metrics to console, in steps.
        enable_tensorboard : bool
            Whether to log metrics to TensorBoard.
        disable_color_printing : bool
            Whether to disable color printing in logs.
        save_tb_folder : str
            Folder to dump TensorBoard states.
        rank_0_only : bool
            Whether to save TensorBoard metrics only for rank 0 or for all ranks.
        enable_wandb : bool
            Whether to log metrics to Weights & Biases.
        wandb_project : Optional[str]
            Weights & Biases project name.
        wandb_run_name : Optional[str]
            Weights & Biases run name.
    """

    log_freq: int = 100
    enable_tensorboard: bool = False
    disable_color_printing: bool = False
    save_tb_folder: str = "tb"
    rank_0_only: bool = True
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class ArgModel:
    """
    ViT Model configurations.

    Args:
        model_name_or_path : str
            HuggingFace model name or local path.
        cache_dir : Optional[str]
            Directory to store downloaded models.
        ignore_mismatched_sizes : bool
            Whether to ignore mismatched sizes when loading pretrained model.
        freeze_backbone : bool
            Whether to freeze the backbone (all layers except classifier).
        freeze_embeddings : bool
            Whether to freeze embedding layers only.
        num_labels : Optional[int]
            Number of output labels (auto-detected from dataset if None).
    """

    model_name_or_path: str = "google/vit-base-patch16-224"
    cache_dir: Optional[str] = None
    ignore_mismatched_sizes: bool = True
    freeze_backbone: bool = False
    freeze_embeddings: bool = False
    num_labels: Optional[int] = None


@dataclass
class ArgOptimizer:
    """
    Optimizer configurations.

    Args:
        name : str
            Optimizer to use (AdamW, SGD, etc.).
        learning_rate : float
            Learning rate to use.
        weight_decay : float
            Weight decay for regularization.
        adam_beta1 : float
            Beta1 parameter for Adam optimizers.
        adam_beta2 : float
            Beta2 parameter for Adam optimizers.
        adam_epsilon : float
            Epsilon parameter for Adam optimizers.
        max_grad_norm : float
            Maximum gradient norm for clipping.
    """

    name: str = "AdamW"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0


@dataclass
class ArgData:
    """
    Data configurations.

    Args:
        dataset_name : str
            Dataset to use for training/evaluation.
        data_dir : str
            Path to the dataset in the file system.
        custom_path : Optional[str]
            Custom path for dataset (for special datasets like ImageNet).
        max_train_samples : Optional[int]
            Maximum number of training samples to use.
        max_eval_samples : Optional[int]
            Maximum number of evaluation samples to use.
        num_workers : int
            Number of data loading workers.
        pin_memory : bool
            Whether to pin memory for data loading.
        image_size : Optional[int]
            Image size to resize to (auto-detected from dataset if None).
    """

    dataset_name: str = "cifar10"
    data_dir: str = "./data"
    custom_path: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    num_workers: int = 4
    pin_memory: bool = True
    image_size: Optional[int] = None


@dataclass
class ArgTraining:
    """
    Training configurations for ViT fine-tuning.

    Args:
        output_dir : str
            Output directory for training artifacts.
        num_train_epochs : int
            Number of training epochs.
        per_device_train_batch_size : int
            Training batch size per device.
        per_device_eval_batch_size : int
            Evaluation batch size per device.
        warmup_ratio : float
            Ratio of warmup steps to total training steps.
        lr_scheduler_type : str
            Learning rate scheduler type.
        save_strategy : str
            Model saving strategy ('epoch', 'steps', 'no').
        save_total_limit : int
            Maximum number of checkpoints to keep.
        evaluation_strategy : str
            Evaluation strategy ('epoch', 'steps', 'no').
        eval_steps : Optional[int]
            Number of steps between evaluations (if evaluation_strategy='steps').
        logging_strategy : str
            Logging strategy ('epoch', 'steps').
        logging_steps : int
            Number of steps between logging.
        load_best_model_at_end : bool
            Whether to load best model at end of training.
        metric_for_best_model : str
            Metric to use for best model selection.
        greater_is_better : bool
            Whether higher metric values are better.
        early_stopping_patience : Optional[int]
            Early stopping patience (None to disable).
        fp16 : bool
            Use FP16 precision.
        bf16 : bool
            Use BF16 precision.
        seed : int
            Random seed for reproducibility.
        resume_from_checkpoint : Optional[str]
            Path to checkpoint to resume from.
        remove_unused_columns : bool
            Whether to remove unused columns from dataset.
    """

    output_dir: str = "./vit_finetune_output"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    evaluation_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    logging_strategy: str = "steps"
    logging_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True
    early_stopping_patience: Optional[int] = 3
    fp16: bool = False
    bf16: bool = False
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    remove_unused_columns: bool = False


@dataclass
class ArgHuggingFace:
    """
    HuggingFace specific configurations.

    Args:
        push_to_hub : bool
            Whether to push model to HuggingFace Hub after training.
        hub_model_id : Optional[str]
            HuggingFace Hub model ID for pushing model.
        hub_strategy : str
            Strategy for pushing to hub ('end', 'every_save', 'checkpoint', 'all_checkpoints').
        hub_token : Optional[str]
            HuggingFace Hub token for authentication.
        hub_private_repo : bool
            Whether to create a private repository on the Hub.
    """

    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "every_save"
    hub_token: Optional[str] = None
    hub_private_repo: bool = False


@dataclass
class ArgEvaluation:
    """
    Evaluation specific configurations.

    Args:
        compute_class_accuracies : bool
            Whether to compute per-class accuracies.
        benchmark_inference : bool
            Whether to benchmark inference speed.
        benchmark_batch_size : int
            Batch size for inference benchmarking.
        benchmark_num_runs : int
            Number of runs for inference benchmarking.
        topk_accuracies : List[int]
            Top-k accuracies to compute (e.g., [1, 5]).
    """

    compute_class_accuracies: bool = False
    benchmark_inference: bool = False
    benchmark_batch_size: int = 1
    benchmark_num_runs: int = 100
    topk_accuracies: List[int] = field(default_factory=lambda: [1, 5])


@dataclass
class ViTFinetuneArgs:
    """
    ViT Fine-tuning arguments.

    Args:
        job : ArgJob
            Job level configurations.
        profiling : ArgProfiling
            Profiling configurations.
        metrics : ArgMetrics
            Metrics configurations.
        model : ArgModel
            Model configurations.
        optimizer : ArgOptimizer
            Optimizer configurations.
        data : ArgData
            Data configurations.
        training : ArgTraining
            Training configurations.
        huggingface : ArgHuggingFace
            HuggingFace specific configurations.
        evaluation : ArgEvaluation
            Evaluation specific configurations.
    """

    job: ArgJob = field(default_factory=ArgJob)
    profiling: ArgProfiling = field(default_factory=ArgProfiling)
    metrics: ArgMetrics = field(default_factory=ArgMetrics)
    model: ArgModel = field(default_factory=ArgModel)
    optimizer: ArgOptimizer = field(default_factory=ArgOptimizer)
    data: ArgData = field(default_factory=ArgData)
    training: ArgTraining = field(default_factory=ArgTraining)
    huggingface: ArgHuggingFace = field(default_factory=ArgHuggingFace)
    evaluation: ArgEvaluation = field(default_factory=ArgEvaluation)

    def to_dict(self) -> dict:
        """
        Convert the dataclass to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the dataclass.
        """
        return asdict(self)

@dataclass
class TaskArguments:
    """Task-specific arguments for training and evaluation."""
    
    do_train: bool = field(default=False, metadata={"help": "Whether to run training"})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run evaluation"})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run prediction"})
    
    # CIM Transform arguments
    enable_cim_transform: bool = field(default=False, metadata={"help": "Enable CIM transformation"})
    cim_config_path: str = field(default=None, metadata={"help": "Path to CIM configuration file"})


def create_training_arguments(task_args: TaskArguments, training_args: ArgTraining, optimizer_args: ArgOptimizer, data_args: ArgData, metrics_args: ArgMetrics):
    """Create TrainingArguments from structured arguments."""
    # Determine report_to list
    report_to = []
    if metrics_args.enable_tensorboard:
        report_to.append("tensorboard")
    if metrics_args.enable_wandb:
        report_to.append("wandb")
    
    hf_training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        overwrite_output_dir=True,  # Enable overwriting by default
        do_train=task_args.do_train,
        do_eval=task_args.do_eval,
        do_predict=task_args.do_predict,
        
        # Training hyperparameters
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        learning_rate=optimizer_args.learning_rate,
        weight_decay=optimizer_args.weight_decay,
        adam_beta1=optimizer_args.adam_beta1,
        adam_beta2=optimizer_args.adam_beta2,
        adam_epsilon=optimizer_args.adam_epsilon,
        max_grad_norm=optimizer_args.max_grad_norm,
        
        # Learning rate scheduler
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_ratio=training_args.warmup_ratio,
        
        # Data loading
        dataloader_num_workers=data_args.num_workers,
        dataloader_pin_memory=data_args.pin_memory,
        
        # Evaluation
        evaluation_strategy=training_args.evaluation_strategy,
        eval_steps=training_args.eval_steps,
        
        # Saving
        save_strategy=training_args.save_strategy,
        save_total_limit=training_args.save_total_limit,
        load_best_model_at_end=training_args.load_best_model_at_end,
        metric_for_best_model=training_args.metric_for_best_model,
        greater_is_better=training_args.greater_is_better,
        
        # Hardware
        seed=training_args.seed,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        
        # Logging
        logging_steps=training_args.logging_steps,
        report_to=report_to,
        
        # Other settings
        remove_unused_columns=False,  # Important for vision tasks
    )
    
    return hf_training_args