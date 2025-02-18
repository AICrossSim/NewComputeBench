# LLM PreTraining

## Dev Run Flag

We leave a dev run flag `dev_run` in the justfile. This can be triggered by setting the `dev_run` flag to `true`.

```bash
just dev_run=true pretrain
```

## Commands

Run the following command to see the available subcommands. Most subcommands are wrapped in the justfile.

```bash
$ python run.py -h
INFO:aixsim_models:Logging verbosity set to 20
usage: run.py [-h] [--config CONFIG] [--print_config[=flags]]
              {count-params,estimate-mem,pretrain,eval-ppl,generate-hf,check-hf-ppl,convert-ckpt,generate-cfg,download-dataset} ...

options:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords
                        separated by comma. The supported flags are: comments, skip_default, skip_null.

subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  Available subcommands:
    count-params        Profiles the number of parameters in a specified model architecture and flavor.
    estimate-mem        Estimate the memory usage of a model during training.
    pretrain            Pretrain a model using the provided arguments.
    eval-ppl            Evaluate the perplexity of a language model.
    generate-hf         Generate text using a Hugging Face model.
    check-hf-ppl        Calculate perplexity of a Hugging Face model on a given dataset.
    convert-ckpt
    generate-cfg        Generate a configuration for pre-training a language model.
    download-dataset    Download a dataset from HuggingFace Hub and optionally create a symlink.
```

### Dataset

- Dev run dataset: `Salesforce/wikitext`'s subset `wikitext-2-raw-v1`
- Pretraining dataset: `HuggingFaceFW/fineweb`'s subset `sample-100BT`

It's recommended to download the pretraining dataset before training and specify the dataset cache directory using `HF_HUB_CACHE` when launching the training script.

```bash
# For downloading dataset,
python run.py download-dataset -h
```

Launch the training script with `HF_HUB_CACHE` set to the dataset cache directory, or edit the `HF_HUB_CACHE` in the justfile.

```bash
just HF_HUB_CACHE=/path/to/dataset_cache_dir <command>
```

### Generate PreTraining Config

```bash
python run.py generate-config -h
```

We also wrap it as a command in the justfile.

```bash
just model_flavor=60M nnodes=1 ngpus=4 data_parallel_replicate_degree=4 batch_size=32 generate-cfg
```

### PreTraining

```bash
CUDA_VISIBLE_DEVICES=0,1 just model_flavor=60M dev_run=false pretrain
```

### Evaluating Perplexity

```bash
# this runs on a single GPU
just model_flavor=60M pt_ckpt=./outputs/checkpoints/aixsim-60M/20250216-191903/step-7075 eval-ppl
# ppl = 125.41
```

### Convert to HuggingFace Checkpoint

- Convert torchtitan checkpoint (pt_ckpt) to HuggingFace PreTrainedModel and PreTrainedTokenizer (hf_ckpt)
```bash
just model_flavor=60M pt_ckpt=./outputs/checkpoints/aixsim-60M/20250216-191903/step-7075/ hf_ckpt=./outputs/huggingface/aixsim-60M/ convert-ckpt
```

- Check HuggingFace perplexity
```bash
just model_flavor=60M hf_ckpt=./outputs/huggingface/aixsim-60M/ check-hf-ppl
# ppl = 125.90
```

### Others

#### Count-Params

Note that embedding layer is not counted

```bash
just model_favor=60M count-params
```
#### Estimate Memory for FSDP/HSDP

```bash
just model_flavor=60M nnodes=1 ngpus=4 batch_size=32 data_parallel_replicate_degree=1 data_parallel_shard_degree=4 estimate-mem
```

## References

- [microbatching](https://github.com/pytorch/torchtitan/issues/292)