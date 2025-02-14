from dataclasses import dataclass, field, asdict
from typing import Union, List, Optional, Literal


@dataclass
class ArgJob:
    """
    Job level configurations.

    Attributes
    ----------
    dump_folder : str
        Folder to dump job outputs.
    description : str
        Description of the job.
    use_for_integration_test : bool
        Add this config to the integration test suite.
    print_args : bool
        Print the args to terminal.
    """

    dump_folder: str = "./torchtitan/outputs"
    description: str = "default job"
    use_for_integration_test: bool = False
    print_args: bool = False


@dataclass
class ArgProfiling:
    """
    Profiling configurations.

    Attributes
    ----------
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

    Attributes
    ----------
    log_freq : int
        How often to log metrics to TensorBoard, in iterations.
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
    """

    log_freq: int = 10
    enable_tensorboard: bool = False
    disable_color_printing: bool = False
    save_tb_folder: str = "tb"
    rank_0_only: bool = True
    enable_wandb: bool = False


@dataclass
class ArgModel:
    """
    Model configurations.

    Attributes
    ----------
    name : str
        Which model to train.
    flavor : str
        Which model config to train.
    norm_type : {'layernorm', 'np_layernorm', 'rmsnorm'}
        Type of layer normalization to use.
    tokenizer_path : str
        Tokenizer path.
    """

    name: str = "llama"
    flavor: str = "debugmodel"
    norm_type: Literal["layernorm", "np_layernorm", "rmsnorm"] = "rmsnorm"
    tokenizer_path: str = "./torchtitan/datasets/tokenizer/tokenizer.model"


@dataclass
class ArgOptimizer:
    """
    Optimizer configurations.

    Attributes
    ----------
    name : str
        Optimizer to use.
    lr : float
        Learning rate to use.
    fused : bool
        Whether the fused implementation (CUDA only) is used.
    early_step_in_backward : bool
        Whether to apply optimizer in the backward.
        Caution, optimizer_in_backward is not compatible with gradients clipping,
        users should not call register_post_accumulate_grad_hook after the optimizer is built.
    """

    name: str = "AdamW"
    lr: float = 8e-4
    fused: bool = False
    early_step_in_backward: bool = False


@dataclass
class ArgTraining:
    """
    Training configurations.

    Attributes
    ----------
    dataset : str
        Dataset to use.
    dataset_path : Optional[str]
        Path to the dataset in the file system.
    batch_size : int
        Batch size.
    seq_len : int
        Sequence length.
    warmup_steps : int
        Steps for lr scheduler warmup, normally 1/5 of training steps.
    max_norm : Union[float, int]
        Max norm for gradient clipping.
    steps : int
        How many train steps to run.
    data_parallel_replicate_degree : int
        The `data_parallel_replicate_degree` argument specifies the degree of data parallelism for weight replication.
        When this value is greater than 1, weights will be replicated across `data_parallel_replicate_degree` ranks.
        If `data_parallel_shard_degree` is also greater than 1, the parallelism method used is HSDP (Hybrid Sharded Data Parallelism).
        Otherwise, the parallelism method used is DDP (Distributed Data Parallelism). 1 means disabled.
    data_parallel_shard_degree : int
        The `data_parallel_shard_degree` argument specifies the degree of data
        parallelism for weight sharding. When this value is greater than 1, weights
        will be sharded across `data_parallel_shard_degree` ranks. If
        `data_parallel_replicate_degree` is also greater than 1, the parallelism
        method used is HSDP (Hybrid Sharded Data Parallelism).  Otherwise, the
        parallelism method used is FSDP (Fully Sharded Data Parallelism).
        -1 means leftover ranks will be used (After DP_REPLICATE/SP/PP). Note that
        only `data_parallel_shard_degree` can be negative. 1 means disabled.
    enable_cpu_offload : bool
        Whether to apply CPU offloading of parameters, gradients, and optimizer states in FSDP.
    tensor_parallel_degree : int
        Tensor Parallelism degree. 1 means disabled.
    disable_loss_parallel : bool
        Whether to apply loss parallel when sequence parallel is enabled.
    fsdp_reshard_after_forward : {'default', 'always', 'never'}
        `reshard_after_forward` specifies the policy for applying `reshard_after_forward`
        within an FSDP setup. `reshard_after_forward` controls parameter behavior after forward,
        trading off memory and communication. See torch's `fully_shard` API for more documentation
        on `reshard_after_forward`.
        The supported policies include "default", "always" and "never":
        - "default" applies default resharding behavior, implementing "smart defaults" for known optimal
          scenarios.
        - "always" will enable `reshard_after_forward` for all forward passes.
        - "never" will disable `reshard_after_forward` for all forward passes.
    mixed_precision_param : {'bfloat16', 'float32'}
        torch dtype to use for parameters when applying mixed precision via FSDP.
        This feature only takes effect when data_parallel_shard_degree > 1
    mixed_precision_reduce : {'float32'}
        torch dtype to use for reductions when applying mixed precision via FSDP.
        This feature only takes effect when data_parallel_shard_degree > 1
    compile : bool
        Whether to compile the model.
    gc_freq : int
        Python garbage control scheduling interval, in steps.
    seed : Optional[int]
        Choose the base RNG seed used for training.
    deterministic : bool
        Use deterministic algorithms wherever possible.
    """

    dataset: str = "c4_mini"
    dataset_path: Optional[str] = None
    batch_size: int = 8
    seq_len: int = 2048
    warmup_steps: int = 200
    max_norm: Union[float, int] = 1.0
    steps: int = 10000
    data_parallel_replicate_degree: int = 1
    data_parallel_shard_degree: int = -1
    enable_cpu_offload: bool = False
    tensor_parallel_degree: int = 1
    disable_loss_parallel: bool = False
    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    mixed_precision_param: Literal["bfloat16", "float32"] = "bfloat16"
    mixed_precision_reduce: Literal["float32"] = "float32"
    compile: bool = False
    gc_freq: int = 50
    seed: Optional[int] = None
    deterministic: bool = False


@dataclass
class ArgExperimental:
    """
    Experimental configurations.

    Attributes
    ----------
    enable_async_tensor_parallel : bool
        Whether to apply async tensor parallel.
    pipeline_parallel_degree : int
        Pipeline Parallelism degree.
    pipeline_parallel_split_points : List[str]
        Names of modules to use as the beginning of a split point.
    pipeline_parallel_schedule : str
        Pipeline Parallel schedule to use.
    pipeline_parallel_schedule_csv : str
        Path to the pipeline parallel schedule csv file to use.
    pipeline_parallel_microbatches : Optional[int]
        How many microbatches to split the global training batch into when using pipeline parallelism.
    enable_compiled_autograd : bool
        Enable CompiledAutograd to compile the backward.
    context_parallel_degree : int
        Context parallelism degree.
    context_parallel_rotate_method : str
        The collective to use in context parallel SDPA for kv shards exchange.
    """

    enable_async_tensor_parallel: bool = False
    pipeline_parallel_degree: int = 1
    pipeline_parallel_split_points: List[str] = field(default_factory=list)
    pipeline_parallel_schedule: str = "1F1B"
    pipeline_parallel_schedule_csv: str = ""
    pipeline_parallel_microbatches: Optional[int] = None
    enable_compiled_autograd: bool = False
    context_parallel_degree: int = 1
    context_parallel_rotate_method: str = "allgather"


@dataclass
class ArgCheckpoint:
    """
    Checkpointing configurations.

    Attributes
    ----------
    enable_checkpoint : bool
        Whether to enable checkpoint.
    folder : str
        The folder to store the checkpoints.
    interval_type : str
        Checkpointing interval unit of measurement.
    interval : int
        Checkpointing interval, in steps or seconds depending on interval_type.
    model_weights_only : bool
        Whether to save only model weights at the end of training.
    export_dtype : {'float16', 'bfloat16', 'float32'}
        Converts to the specified precision when training completes and model_weights_only=true.
    create_seed_checkpoint : bool
        Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
    async_mode : str
        Which async checkpoint mode to use.
    keep_latest_k : int
        Keeps only the latest k checkpoints, and purging older ones.
    load_step : int
        Load the checkpoint at the specified step.
    exclude_from_loading : List[str]
        Exclude specific keys from being loaded from the checkpoint.
    """

    enable_checkpoint: bool = False
    folder: str = "checkpoint"
    interval_type: str = "steps"
    interval: int = 500
    model_weights_only: bool = False
    export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    create_seed_checkpoint: bool = False
    async_mode: str = "disabled"
    keep_latest_k: int = 0
    load_step: int = -1
    exclude_from_loading: List[str] = field(default_factory=list)


@dataclass
class ArgActivationCheckpoint:
    """
    Activation checkpointing configurations.

    Attributes
    ----------
    mode : {'none', 'full', 'selective'}
        Type of activation checkpointing to use.
    selective_ac_option : str
        Selective activation checkpointing options.
    """

    mode: Literal["none", "full", "selective"] = "selective"
    selective_ac_option: str = "2"


@dataclass
class ArgFloat8:
    """
    Float8 configurations.

    Attributes
    ----------
    enable_float8_linear : bool
        If true, swaps `torch.nn.Linear` with `Float8Linear`.
    enable_fsdp_float8_all_gather : bool
        Whether enable float8 all-gather in FSDP.
    precompute_float8_dynamic_scale_for_fsdp : bool
        Whether precompute float8 scales dynamically for FSDP.
    """

    enable_float8_linear: bool = False
    enable_fsdp_float8_all_gather: bool = False
    precompute_float8_dynamic_scale_for_fsdp: bool = False


@dataclass
class ArgComm:
    """
    Communications library settings.

    Attributes
    ----------
    init_timeout_seconds : int
        Timeout for communication operations, during initialization and first train step.
    train_timeout_seconds : int
        Timeout for communication operations after the first train step.
    trace_buf_size : int
        Flight recorder ring buffer size.
    """

    init_timeout_seconds: int = 300
    train_timeout_seconds: int = 100
    trace_buf_size: int = 20000


@dataclass
class ArgMemoryEstimation:
    """
    Memory estimation settings.

    Attributes
    ----------
    enabled : bool
        Whether to estimate memory usage for FSDP.
    disable_fake_mode : bool
        Whether to estimate memory under FakeTensorMode.
    """

    enabled: bool = False
    disable_fake_mode: bool = False


@dataclass
class PreTrainArgs:
    """
    Pre-training arguments.

    Attributes
    ----------
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
    training : ArgTraining
        Training configurations.
    experimental : ArgExperimental
        Experimental configurations.
    checkpoint : ArgCheckpoint
        Checkpointing configurations.
    activation_checkpoint : ArgActivationCheckpoint
        Activation checkpointing configurations.
    float8 : ArgFloat8
        Float8 configurations.
    comm : ArgComm
        Communications library settings.
    memory_estimation : ArgMemoryEstimation
        Memory estimation settings.
    """

    job: ArgJob = field(default_factory=ArgJob)
    profiling: ArgProfiling = field(default_factory=ArgProfiling)
    metrics: ArgMetrics = field(default_factory=ArgMetrics)
    model: ArgModel = field(default_factory=ArgModel)
    optimizer: ArgOptimizer = field(default_factory=ArgOptimizer)
    training: ArgTraining = field(default_factory=ArgTraining)
    experimental: ArgExperimental = field(default_factory=ArgExperimental)
    checkpoint: ArgCheckpoint = field(default_factory=ArgCheckpoint)
    activation_checkpoint: ArgActivationCheckpoint = field(
        default_factory=ArgActivationCheckpoint
    )
    float8: ArgFloat8 = field(default_factory=ArgFloat8)
    comm: ArgComm = field(default_factory=ArgComm)
    memory_estimation: ArgMemoryEstimation = field(default_factory=ArgMemoryEstimation)

    def to_dict(self) -> dict:
        """
        Convert the dataclass to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the dataclass.
        """
        return asdict(self)
