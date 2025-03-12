from dataclasses import dataclass, field, asdict
from typing import Literal


from ..llm.arg_manager import (
    ArgJob,
    ArgProfiling,
    ArgMetrics,
    ArgModel,
    ArgOptimizer,
    ArgTraining,
    ArgExperimental,
    ArgCheckpoint,
    ArgActivationCheckpoint,
    ArgFloat8,
    ArgComm,
    ArgMemoryEstimation,
)


@dataclass
class ArgRandomBitFlipTransform:
    use_regex: bool = True
    transform_flavor: Literal["fc"] = "fc"
    layer_name_to_config: dict = field(default_factory=dict)


@dataclass
class PreTrainArgs:
    """
    Pre-training arguments.

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
        transform: ArgRandomBitFlipTransform
            Random bitflip transformation.
    """

    job: ArgJob = field(default_factory=ArgJob)
    profiling: ArgProfiling = field(default_factory=ArgProfiling)
    metrics: ArgMetrics = field(default_factory=ArgMetrics)
    model: ArgModel = field(default_factory=ArgModel)
    optimizer: ArgOptimizer = field(default_factory=ArgOptimizer)
    training: ArgTraining = field(default_factory=ArgTraining)
    experimental: ArgExperimental = field(default_factory=ArgExperimental)
    checkpoint: ArgCheckpoint = field(default_factory=ArgCheckpoint)
    activation_checkpoint: ArgActivationCheckpoint = field(default_factory=ArgActivationCheckpoint)
    float8: ArgFloat8 = field(default_factory=ArgFloat8)
    comm: ArgComm = field(default_factory=ArgComm)
    memory_estimation: ArgMemoryEstimation = field(default_factory=ArgMemoryEstimation)
    transform: ArgRandomBitFlipTransform = field(default_factory=ArgRandomBitFlipTransform)

    def to_dict(self) -> dict:
        """
        Convert the dataclass to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the dataclass.
        """
        return asdict(self)
