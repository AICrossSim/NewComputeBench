# Verify ONN RoBERTa eval-only fix

Context: PR adds `--model_weights_path` to `experiments/roberta-optical-transformer/run_glue.py`
and rewrites the "Evaluation only" section of `docs/source/modules/tutorials/simulations/onn_roberta.rst`
into two blocks. The bug being fixed: re-loading a fine-tuned optical checkpoint via
`AutoModelForSequenceClassification.from_pretrained` silently drops the calibrated
`*_min_max` / `seed` buffers as "unexpected keys", so eval-only metrics come out
worse than the eval reported at the end of training.

Hardware needed: GPU whose compute capability is in `torch.cuda.get_arch_list()`
of the installed `torch` (e.g. A100 sm_80 or H100 sm_90 with `torch 2.6.0+cu124`).
The dev box is B200/sm_100 — incompatible with the current pinned torch wheel.

## 0. Sanity

```bash
cd /path/to/NewComputeBench
git status                # confirm the two modified files are present
.venv/bin/python -c "import torch; print(torch.cuda.get_arch_list()); \
  print([torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])"
# Expect each device's (major, minor) to appear (or have a lower PTX) in the arch list.

cd experiments/roberta-optical-transformer
python run_glue.py --help 2>&1 | grep -- --model_weights_path
# Expect one line describing the new flag.
```

## 1. Fine-tune to produce a checkpoint with calibrated buffers

Use the Single task block from the tutorial — keep epochs small for a quick test:

```bash
cd experiments/roberta-optical-transformer

TASK_NAME="mrpc"
MODEL_NAME="FacebookAI/roberta-base"
LEARNING_RATE="2e-5"
BATCH_SIZE="16"
NUM_EPOCHS="3"
TRANSFORM_CONFIG="transform_cfg.yaml"

CUDA_VISIBLE_DEVICES=0 python run_glue.py \
    --model_name_or_path "${MODEL_NAME}" \
    --task_name "${TASK_NAME}" \
    --do_train --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --output_dir "./output/${TASK_NAME}_optical" \
    --overwrite_output_dir \
    --transform_config "${TRANSFORM_CONFIG}" \
    --eval_strategy epoch \
    --save_strategy epoch \
    --logging_steps 50 \
    --seed 42
```

**Record:**
- `eval_accuracy` / `eval_f1` from the final epoch (in stdout and `output/mrpc_optical/eval_results.json`).
- Confirm `output/mrpc_optical/model.safetensors` exists.
- Optional: dump key list to confirm calibration buffers were saved:
  ```bash
  .venv/bin/python -c "
  from safetensors.torch import load_file
  sd = load_file('experiments/roberta-optical-transformer/output/mrpc_optical/model.safetensors')
  buf_keys = [k for k in sd if any(s in k for s in ('_min_max', '.seed'))]
  print(f'{len(buf_keys)} calibration-buffer keys, e.g.:'); print('\n'.join(buf_keys[:6]))
  "
  ```
  Expect dozens of `..._min_max` and `.seed` entries.

## 2. Verify the bug exists WITHOUT the new flag

Re-run eval on the saved checkpoint without `--model_weights_path` (this is what
the old tutorial would have produced if MODEL_NAME pointed at the fine-tune dir):

```bash
CUDA_VISIBLE_DEVICES=0 python run_glue.py \
    --model_name_or_path "./output/${TASK_NAME}_optical" \
    --task_name "${TASK_NAME}" \
    --do_eval \
    --max_seq_length 128 \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --output_dir "./output/${TASK_NAME}_eval_nobufs" \
    --transform_config "${TRANSFORM_CONFIG}" \
    --overwrite_output_dir
```

**Record** the metrics. **Expected:** stdout contains a HF warning of the form
`Some weights of the model checkpoint at … were not used` listing the
`*_min_max` / `seed` keys — that's the silent drop. Eval numbers should be
**lower** than step 1's final-epoch eval (this confirms the bug).

## 3. Verify the fix WITH the new flag

```bash
CUDA_VISIBLE_DEVICES=0 python run_glue.py \
    --model_name_or_path "${MODEL_NAME}" \
    --task_name "${TASK_NAME}" \
    --do_eval \
    --max_seq_length 128 \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --output_dir "./output/${TASK_NAME}_eval_withbufs" \
    --transform_config "${TRANSFORM_CONFIG}" \
    --model_weights_path "./output/${TASK_NAME}_optical" \
    --overwrite_output_dir
```

**Expected:**
- Stdout includes `✅ Loaded fine-tuned optical weights from …`.
- The "missing keys" / "unexpected keys" warnings from the new loader should be
  empty or near-empty (a few classifier head names are fine).
- Eval numbers should **match** step 1's final-epoch eval within rounding
  (≤ ~0.001 drift is OK; a meaningful gap means buffers still aren't loading).

## 4. Verify the post-transform / no-fine-tune block (other half of the doc)

Run the first new doc block as written, using a hub-fine-tuned model so the
classifier head is real:

```bash
TASK_NAME="mrpc"
MODEL_NAME="Intel/roberta-base-mrpc"
BATCH_SIZE="16"
TRANSFORM_CONFIG="transform_cfg.yaml"

CUDA_VISIBLE_DEVICES=0 python run_glue.py \
    --model_name_or_path "${MODEL_NAME}" \
    --task_name "${TASK_NAME}" \
    --do_eval \
    --max_seq_length 128 \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --output_dir "./output/${TASK_NAME}_eval_posttransform" \
    --transform_config "${TRANSFORM_CONFIG}" \
    --overwrite_output_dir
```

**Expected:** completes without error and produces eval numbers in the ballpark
of the "Optical Transformer" row of the "Post-training transform" table in the
tutorial (MRPC ≈ 0.78). It will be **lower** than the original `Intel/roberta-base-mrpc`
because calibration is being built up live during eval — that is the documented
behavior of this mode, not a regression.

## 5. Build the docs

```bash
cd docs
make html SPHINXOPTS="-W"   # -W turns RST warnings into errors
```

**Expected:** clean build, no warnings about the two new section headings.
Open `docs/build/html/modules/tutorials/simulations/onn_roberta.html` and eyeball
the two new code blocks.

## 6. Report back

Paste the four eval numbers (step 1 final-epoch eval, step 2 no-flag eval,
step 3 with-flag eval, step 4 post-transform eval) here so we can confirm
step 3 ≈ step 1 and step 2 < step 1. If step 3 still drifts from step 1,
the loader needs to also restore the dropped buffers from
`AutoModelForSequenceClassification.from_pretrained`'s "unexpected keys"
list — file an update.
