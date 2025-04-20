# Development Guidelines for Transform-Aware LLM Training

In this codebase, we can support both transform-ware continual pretraining and pretraining from scratch.
However, if the transform is too lossy, the model may not be able to learn effectively if trained from scratch.
continual pretraining is recommended in this case.

## Continual Pretraining

!!! info "Example: Continual Pretraining with Simulated Optical Compute"

    The example scripts can be found at [`experiments/llm-optical-transformer/continual_pretraining`](/experiments/llm-optical-transformer/continual_pretraining/)

HuggingFace `transformers`'s Trainer is used to perform continual pretraining on the converted/pretrained checkpoint on HuggingFace. Our pretrained AICrossSim/clm checkpoints can be found in [this collection](https://huggingface.co/collections/AICrossSim/newcomputebench-clm-digital-67d19e95ebacdbc3e5752be3)

Here we use optical compute in the [*Optical Transformers* (`OT`) paper](https://arxiv.org/abs/2302.10360) as an example. You may follow the following steps to implement other new compute paradigms. To implement `OT`, we have a few key components you can find in [`src/aixsim_models/optical_compute/optical_transformer`](https://github.com/AICrossSim/NewComputeBench/blob/master/src/aixsim_models/optical_compute/optical_transformer):

1. Simulated OT linear layer and matmul
    - class [`OpticalTransformerLinear`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/optical_compute/layers.py) to simulate the linear layer. All the linear layers in the pretrained model will be replace by this linear layer except for `lm_head`.
    - function [`OpticalTransformerFunctions.quantized_matmul_fn`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/optical_compute/core/optical_transformer/matmul.py) to simulate the matmul. The matmul is wrapped in [`HFOpticalTransformerLlamaAttention`](https://github.com/AICrossSim/NewComputeBench/blob/master/src/aixsim_models/optical_compute/optical_transformer/layers.py) to simulate the Query-Key matmul and Attention-Value matmul.

    !!! info "Kernels in `mase_triton.optical_compute`"

        - We use Triton instead of PyTorch API to implement `OpticalTransformerLinear` (essentially functional [`ot_qlinear_fn`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/optical_compute/core/optical_transformer/linear.py)) and [`OpticalTransformerFunctions.quantized_matmul_fn`](https://github.com/DeepWok/mase-triton/blob/master/src/mase_triton/optical_compute/core/optical_transformer/matmul.py), because for the method described in *Optical Transformers*, if we implement it using PyTorch built-in functions, the training will consume a lot of GPU memory and the training speed will be very slow. We implement Triton kernel mainly for **saving GPU memory**. If your simulation can be memory-effciently implemented using PyTorch built-in functions, you don't need to use Triton.

        - HuggingFace `transformers`'s Trainer may not work with autotuned Triton kernels. This is why in [`mase-triton`](https://github.com/DeepWok/mase-triton), the autotuning is disabled.

2. A pass to transform [`LlamaForCausalLM`](https://github.com/huggingface/transformers/blob/6daa3eeba582facb57cd71db8efb66998b12942f/src/transformers/models/llama/modeling_llama.py#L739).

    We implement the function [`transform_hf_model`](https://github.com/AICrossSim/NewComputeBench/blob/master/src/aixsim_models/optical_compute/optical_transformer/transform.py) to transform the model. Inside the function, there are two for loops, one for replacing attention layer with `HFOpticalTransformerLlamaAttention` to replace matmuls and the other for replacing linear layer with `OpticalTransformerLinear`.

3. Transform config

    We use a YAML file to specify the transform config ([`configs/default.yaml`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-optical-transformer/continual_pretraining/configs/default.yaml)). In `transform_hf_model`'s for loop, the `TransformConfigManager` uses the layer name to find the corresponding transform config.


With these two components, we can simply adapt HuggingFace's [`run_clm.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) such that the model is transformed before training starts. The **adapted `run_clm.py`** can be found [here](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-optical-transformer/continual_pretraining/run_clm.py).

- In the **adapted `run_clm.py`**, we insert the following code snippet to transform the model before training starts:

    ```python
        if model_args.transform_config is not None:
            with open(model_args.transform_config, "r") as f:
                transform_args = yaml.safe_load(f)
            config_manager = TransformConfigManager(**transform_args)
            transformed_layers = transform_hf_model(model, config_manager)
            transform_histogram = make_transform_histogram(transformed_layers=transformed_layers)
            logger.info(f"🔍 Transformed layers:\n{transform_histogram}")
        else:
            logger.info("⚠️ No transform config file provided. Using the original model.")
    ```

- You may copy the **adapted `run_clm.py`** and replace the `OT` transform pass `transform_hf_model` with your own transform pass.

Then as shown in the [`justfile`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-optical-transformer/continual_pretraining/justfile), we can launch the optical compute aware continual pretraining by:

```bash
# This run uses small batch size and training steps for demonstration purpose.
python run_clm.py \
    --model_name_or_path AICrossSim/clm-60m \
    --dataset_name HuggingFaceFW/fineweb-edu \
    --dataset_config_name "sample-10BT" \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 50 \
    --do_train \
    --report_to "wandb" \
    --learning_rate 5e-5 \
    --max_steps 100 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --bf16 \
    --dataloader_num_workers 16 \
    --preprocessing_num_workers 32 \
    --tokenizer_name HuggingFaceTB/cosmo2-tokenizer \
    --output_dir ./output/test-clm-trainer \
    --transform_config ./configs/default.yaml \
    --seed 42

```


## Pretraining from Scratch

We use torchtitan as the backend to pretrain transformed LLM from scratch. Please refer to [`experiments/llm-optical-transformer/pretrain/run.py`](https://github.com/AICrossSim/NewComputeBench/blob/master/experiments/llm-optical-transformer/pretrain/run.py).