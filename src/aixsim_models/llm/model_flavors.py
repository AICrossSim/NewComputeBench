import logging

logger = logging.getLogger(__name__)


def register_model_configs():
    from torchtitan.models.llama import ModelArgs
    from torchtitan.models import (
        models_config,
        model_name_to_tokenizer,
        model_name_to_cls,
    )
    from torchtitan.parallelisms import models_parallelize_fns, models_pipelining_fns

    model_arch_name = "aixsim"
    aixsim_configs = {}
    models_config[model_arch_name] = aixsim_configs
    model_name_to_tokenizer[model_arch_name] = "hf"
    model_name_to_cls[model_arch_name] = model_name_to_cls["llama3"]
    models_parallelize_fns[model_arch_name] = models_parallelize_fns["llama3"]
    models_pipelining_fns[model_arch_name] = models_pipelining_fns["llama3"]

    aixsim_configs["debug"] = ModelArgs(
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=-1,
        multiple_of=128,
        ffn_dim_multiplier=1.3,
        rope_theta=10000,
    )

    aixsim_configs["60M"] = ModelArgs(
        dim=384,
        n_layers=22,
        n_heads=6,
        n_kv_heads=2,
        vocab_size=-1,
        multiple_of=128,
        ffn_dim_multiplier=1.3,
        rope_theta=10000,
    )

    aixsim_configs["200M"] = ModelArgs(
        dim=768,
        n_layers=24,
        n_heads=12,
        n_kv_heads=4,
        vocab_size=-1,
        multiple_of=128,
        ffn_dim_multiplier=1.3,
        rope_theta=10000,
    )

    aixsim_configs["400M"] = ModelArgs(
        dim=960,
        n_layers=30,
        n_heads=15,
        n_kv_heads=5,
        vocab_size=-1,
        multiple_of=128,
        ffn_dim_multiplier=1.3,
        rope_theta=10000,
    )

    aixsim_configs["600M"] = ModelArgs(
        dim=1152,
        n_layers=32,
        n_heads=18,
        n_kv_heads=6,
        vocab_size=-1,
        multiple_of=128,
        ffn_dim_multiplier=1.3,
        rope_theta=10000,
    )

    # aixsim_configs["900M"] = ModelArgs(
    #     dim=1344,
    #     n_layers=32,
    #     n_heads=21,
    #     n_kv_heads=7,
    #     vocab_size=-1,
    #     multiple_of=128,
    #     ffn_dim_multiplier=1.3,
    #     rope_theta=10000,
    # )

    aixsim_configs["1.1B"] = ModelArgs(
        dim=1536,
        n_layers=32,
        n_heads=24,
        n_kv_heads=8,
        vocab_size=-1,
        multiple_of=128,
        ffn_dim_multiplier=1.3,
        rope_theta=10000,
    )
    logger.info(f"Registered the following AIxSIM model configurations: {list(aixsim_configs.keys())}")
    logger.info(f"Registered `model_arch` 'aixsim' with `tokenizer_type` 'hf'")
    logger.info(f"Registered `model_arch` 'aixsim' with `parallelize_fn` 'llama3'")
