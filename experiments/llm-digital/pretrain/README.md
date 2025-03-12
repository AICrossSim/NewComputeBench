# LLM PreTraining

- This folder includes the scripts for preTraining AICrossSim CLM models on HuggingFaceFW/fineweb-edu dataset.
- We leave `dev_run=true` in the justfile for quick testing.

## Model Arch and Datasets

- Model architecture is based on Llama-3.1 with smaller vocab, hidden sizes and number of layers RMSNorm
    - Grouped Query Attention
    - RoPE positional encoding
    - MLP: `up_proj, gate_proj, down_proj`
    - tokenizer/vocab: [`HuggingFaceTB/cosmo2-tokenizer`](https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer)
- Model Sizes: 60M, 200M, 400M, 600M, 1.1B
- Pretraining is performed on [`HuggingFaceFW/fneweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

## Commands

Run the following command to see the available subcommands. Most subcommands are also wrapped in the justfile.

```bash
$ python run.py -h
INFO:aixsim_models:Logging verbosity set to 20
usage: run.py [-h] [--config CONFIG] [--print_config[=flags]] {count-params,pretrain,eval,generate-hf,convert-ckpt,generate-cfg,download-dataset} ...

options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords separated by comma. The supported flags are: comments,
                        skip_default, skip_null.

subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  Available subcommands:
    count-params        Profiles the number of parameters in a specified model architecture and flavor.
    pretrain            Pretrain a model using the provided arguments.
    eval
    generate-hf         Generate text using a Hugging Face model.
    convert-ckpt
    generate-cfg        Generate a configuration for pre-training a language model.
    download-dataset    Download a dataset from HuggingFace Hub and optionally create a symlink.
```

## Pipeline

1. Generate PreTraining Config

    `run.py generate-cfg` generates a configuration template for pretraining. The transform config can also be included in it (Refer to [llm-bitflip](/experiments/llm-bitflip/pretrain/run.py)).

    ```bash
    python run.py generate-cfg -h
    ```

    Or, use the shortcut in justfile.
    ```bash
    just model_flavor=60M nnodes=1 ngpus=4 data_parallel_replicate_degree=4 batch_size=32 generate-cfg
    ```

2. PreTraining

    ```bash
    python run.py pretrain -h
    ```

    Or, use the shortcut in justfile.
    ```
    just model_flavor=60M dev_run=false pretrain
    ```

3. Convert to HuggingFace Checkpoint

    ```bash
    python run.py convert-ckpt -h
    ```

4. Evaluation

    ```bash
    python run.py eval pt-ppl # evaluate checkpoint in torchtitan format
    python run.py eval hf-ppl # evaluate checkpoint in HuggingFace format. The ppl of converted HF checkpoint should match pt-ppl.
    python run.py eval hf-lm-eval # evaluate HF checkpoint using `lm-evaluation-harness`.
    ```

5. Generate Text

    ```bash
    python run.py generate-hf -h
    ```
### Others

- Count-Params

Note that embedding layer is not counted

```bash
just model_favor=60M count-params
```