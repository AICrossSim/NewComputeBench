# LLM PreTraining


## Dev Run Flag

We leave a dev run flag `dev_run` in the justfile. This can be triggered by setting the `dev_run` flag to `true`.

```bash
just dev_run=true pretrain
```

## Dataset

- Dev run dataset: `Salesforce/wikitext`'s subset `wikitext-2-raw-v1`
- Pretraining dataset: `HuggingFaceFW/fineweb`'s subset `sample-100BT`

It's recommended to download the pretraining dataset before training and specify the dataset cache directory using `HF_HUB_CACHE` when launching the training script.

```bash
# For downloading dataset,
python run.py download-dataset -h
```

Launch the training script with `HF_HUB_CACHE` set to the dataset cache directory.

```bash
just HF_HUB_DIR=/path/to/dataset_cache_dir <command>
```

## Generate PreTraining Config

```bash
python run.py generate-config -h
```

We also wrap it as a command in the justfile.

```bash
just model_flavor=60M nnodes=1 ngpus=4 data_parallel_replicate_degree=4 batch_size=32 generate-cfg
```

## PreTraining

```bash
CUDA_VISIBLE_DEVICES=0,1 just model_flavor=60M dev_run=false pretrain
```

## Evaluating Perplexity

```bash
# this runs on a single GPU
just model_flavor=60M pt_ckpt=./outputs/checkpoints/aixsim-60M/20250216-191903/step-7075 eval-ppl
# ppl = 125.41
```

## Convert to HuggingFace Checkpoint

- Convert torchtitan checkpoint (pt_ckpt) to HuggingFace PreTrainedModel and PreTrainedTokenizer (hf_ckpt)
```bash
just model_flavor=60M pt_ckpt=./outputs/checkpoints/aixsim-60M/20250216-191903/step-7075/ hf_ckpt=./outputs/huggingface/aixsim-60M/ convert-ckpt
```

- Check HuggingFace perplexity
```bash
just model_flavor=60M hf_ckpt=./outputs/huggingface/aixsim-60M/ check-hf-ppl
# ppl = 125.90
```

## Others

### Count-Params

Note that embedding layer is not counted

```bash
just model_favor=60M count-params
```
### Estimate Memory for FSDP/HSDP

```bash
just model_flavor=60M nnodes=1 ngpus=4 batch_size=32 data_parallel_replicate_degree=1 data_parallel_shard_degree=4 estimate-mem
```

### Checkpoint format

- convert dcp to torch pt/torch pt to dcp
- 🚧 convert torch pt to meta pt
- [convert meta pt to hf safetensors](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)


## References

- [microbatching](https://github.com/pytorch/torchtitan/issues/292)