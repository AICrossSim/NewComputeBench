# Random Bitflip

This is tutorial on how to run post-bitflip evaluation on a pretrained checkpoint, and how to run a bitflip-aware pretraining from scratch.

## Overview

- **Post-bitflip evaluation** loads a pretrained checkpoint from HuggingFace, applies bitflip transformation to the model, and evaluates the model on downstream tasks.
    - The entry point is at [`experiments/llm-bitflip/transform/minimal.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/transform/minimal.py).
- **Bitflip-aware pretraining** creates a randomly initialized model, applies bitflip transformation to the model, and pretrains the model on FineWeb-Edu.
    - The entry point is at [`experiments/llm-bitflip/pretrain/run.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/run.py).
- To accelerate the emulation, we build custom triton kernels in [`mase-triton`](https://pypi.org/project/mase-triton/).
    - The random bitflip kernel is wrapped in function [`mase_triton.random_bitflip.core.random_bitflip_fn`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/random_bitflip/core.py), which supports unique bitflip probability for the sign-exp bits and the mantissa bits. `random_bitflip_fn` also supports zeroing out outliers (and "NaN" values) by assigning a threshold.
    - The random bitflip probability only supports a power of 0.5, e.g, `0.5`, `0.5^2`, `0.5^3`, etc. The kernel will automatically convert the probability to the nearest power of 0.5. Due to the limitation of the pseudo random number generation algorithm (Philox), the kernel only works for a random bitflip probability greater or equal to `0.5^-24=5.96-08`.

## Evaluation of Post-Training Bitflip Transform

!!! info "Environment Setup?"

    If you have not set up environments, please follow the guidelines in [Environment Setup](../env-setup.md).

We offer minimal scripts to apply post-training bitflip transform on all linear layers (contributing to over 90% FLOPS in Transformers) in a HuggingFace pretrained model and evaluate the transformed model with `lm-eval-harness`.

### Transform & Evaluate on Downstream Tasks

```bash
cd experiments/llm-bitflip/transform

model_name="unsloth/Meta-Llama-3.1-8B-Instruct" # HuggingFace model name
x_p_exp=null                                    # bitflip probability for the sign-exp bits of the activation. Null means no bitflip.
w_p_exp=null                                    # bitflip probability for the sign-exp bits of the weight. Null means no bitflip.
x_zero_out_t="100"                              # threshold for zeroing out outliers (and "NaN" values) of the activation
w_zero_out_t="1.25"                             # threshold for zeroing out outliers (and "NaN" values) of the weight
x_p_frac=$(bc <<< "scale=10; 0.5^10")           # bitflip probability for the mantissa bits of the activation
w_p_frac=$(bc <<< "scale=10; 0.5^10")           # bitflip probability for the mantissa bits of the weight
python minimal.py eval-bitflip \
    --model_name ${model_name} \
    --bitflip_config "default" \
    --default_bitflip_config.x_p_exp=${x_p_exp} \
    --default_bitflip_config.x_p_frac=${x_p_frac} \
    --default_bitflip_config.x_zero_out_t=${x_zero_out_t} \
    --default_bitflip_config.w_p_exp=${w_p_exp} \
    --default_bitflip_config.w_p_frac=${w_p_frac} \
    --default_bitflip_config.w_zero_out_t=${w_zero_out_t} \
    --tasks ['wikitext']
```

!!! info "eval-bitflip"
    This `eval-bitflip` subcommand also uses `lm-eval-harness`'s `simple_evaluate` function. Please refer to the evaluation section of [LLM Pretraining & Evaluation](../01-model-training/llm-pretrain-and-eval.md) for more details.

### Evaluate the Original Model

You may want to compare the evaluation results of the bitflip model with the original model. You can do this by running the following command:

```bash
python minimal.py eval-ori \
    --model_name ${model_name} \
    --tasks ['wikitext']
```
### Test the Generation

We also offer a `hf-gen` script to

### Simple Generation

We provide a simple generation subcommand `hf-gen` as well.

```bash
prompt="London is"
max_new_tokens="100"
do_sample="true"
temperature="0.6"
top_k="50"
top_p="0.9"

python minimal.py hf-gen \
    AICrossSim/clm-60m \
    ${prompt} \
    --max_new_tokens ${max_new_tokens} \
    --do_sample ${do_sample} \
    --temperature ${temperature} \
    --top_k ${top_k} \
    --top_p ${top_p} \
    --bitflip_config "default" \
    --default_bitflip_config.x_p_exp=${x_p_exp} \
    --default_bitflip_config.x_p_frac=${x_p_frac} \
    --default_bitflip_config.x_zero_out_t=${x_zero_out_t} \
    --default_bitflip_config.w_p_exp=${w_p_exp} \
    --default_bitflip_config.w_p_frac=${w_p_frac} \
    --default_bitflip_config.w_zero_out_t=${w_zero_out_t}
```

!!! success "Our Initial Experiments"
    For our `AICrossSim/clm-1.1b`, we sweep `x_p_frac` and `w_p_frac` and observe how perplexity and generated texts changes.

    Here are some samples of the generated texts: [link](https://docs.google.com/spreadsheets/d/1N9_i3_YzKhDfI6H0EWO86zVMxiwsHcSUSbll2ws4zRA/edit?usp=sharing)

    Notably, when the perplexity is increase by 1%, the generated text are consistent with the original text.


## Bitflip-Aware Pretraining

Similar to the [pretraining of script of AICrossSim-CLM](../01-model-training/llm-pretrain-and-eval.md) ([`experiments/llm-bitflip/pretrain/run.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/run.py)), we offer a [`experiments/llm-bitflip/pretrain/run.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/run.py) script to run bitflip-aware pretraining from scratch. The subcommands accepts the same arguments as [`experiments/llm-bitflip/pretrain/run.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/run.py), but with an additional argument for bitflip transform.

- For example, we can run the following command to run a bitflip-aware pretraining for `AICrossSim-CLM-60M` on 2 H100 96GB GPUs.
    1. Generate a training config with bitflip transform config.

        ```bash
        cd experiments/llm-bitflip/pretrain

        bitflip_transform_config="./configs/meta/fc-only-w-a-exp-frac.yaml"
        python run.py generate-cfg \
            ${bitflip_transform_config} \
            --model_arch "aixsim" \
            --model_flavor "60M" \
            --batch_size 48 \
            --data_parallel_replicate_degree 2\
            --data_parallel_shard_degree -1 \
            --token_num_scale 22 \
            --compile "false" \
            --save_path "./configs/tutorial-60m.yaml"
        ```

    2. Run the pretraining with the generated config.

        ```bash
        num_gpus="2"
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" STREAM_HF_DATA="1" \
        torchrun --nproc_per_node=${num_gpus} --rdzv_backend c10d \
            --rdzv_endpoint="localhost:0" --local-ranks-filter 0 \
            --role rank --tee 3 \
            run.py pretrain \
            --config configs/tutorial-60m.yaml \
            --metrics_args.enable_wandb false
        ```

    3. Convert the checkpoint to HuggingFace format.

        ```bash
        torchrun_ckpt_path="path/to/torchrun/checkpoint"
        output_dir="path/to/output/dir"
        python run.py convert-ckpt pt2hf \
            "aixsim" "60M" \
            ${torchrun_ckpt_path} \
            ${output_dir}
        ```

    !!! success "Our Bitflip-Aware Training Results of AICrossSim-CLM-60M"

        We performed bitflip-aware pretraining on `AICrossSim-CLM-60M` on 2 H100 96GB GPUs for 2.5 hours.

        - Train config: [`experiments/llm-bitflip/pretrain/configs/aixsim-60M.yaml`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/configs/aixsim-60M.yaml)
        - Wandb logs: [link](https://wandb.ai/cz98/torchtitan/runs/bbyruxxh/overview)
        - HuggingFace checkpoint: [AICrossSim/bitflip-fc-clm-60m](https://huggingface.co/AICrossSim/bitflip-fc-clm-60m)

    - Similarly, one can run bitflip-aware pretraining for other AICrossSim-CLM model sizes. Here we summarize our current bitflip-aware pretraining progress.

        | Model | Environment | Pretraining Time | Training Config | Wandb Logs | HuggingFace Checkpoint |
        |-------|-------------|------------------|------------------|------------|------------------------|
        | 60M | 2x H100 96GB | 2.5 hours | [`configs/aixsim-60M.yaml`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/configs/aixsim-60M.yaml) | [link](https://wandb.ai/cz98/torchtitan/runs/bbyruxxh/overview) | [AICrossSim/bitflip-fc-clm-60m](https://huggingface.co/AICrossSim/bitflip-fc-clm-60m) |
        | 200M | 2x H100 96GB | 14.3 hours | [`configs/aixsim-200M.yaml`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/configs/aixsim-200M.yaml) | [link](https://wandb.ai/cz98/torchtitan/runs/iivbk9nr/overview) | [AICrossSim/bitflip-fc-clm-200m](https://huggingface.co/AICrossSim/bitflip-fc-clm-200m) |
        | 400M | 6x A6000 48GB | 33 hours | [`configs/aixsim-400M.yaml`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/configs/aixsim-400M.yaml) | [link](https://wandb.ai/cz98/torchtitan/runs/6mnsbo7e/overview) | [AICrossSim/bitflip-fc-clm-400m](https://huggingface.co/AICrossSim/bitflip-fc-clm-400m) |
        | 1.1B | 8x H200 141GB | 51 hours | [`configs/aixsim-1.1B.yaml`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-bitflip/pretrain/configs/aixsim-1.1B.yaml) | [link](https://wandb.ai/cz98/torchtitan/runs/5tbo5tkg?nw=nwusercz98) | [AICrossSim/bitflip-fc-clm-1.1b](https://huggingface.co/AICrossSim/bitflip-fc-clm-1.1b) |