# Models

A list of models we aim to port to NewComputeBench.

| Task Type | Model Name | Model Sizes | Description |
| --- | :----------| ------------| :---------- |
| Text classification | `RoBERTa` | `roberta-base` | A classic encoder-only language model we include for sanity checks. |
| Causal language modeling | `AICrossSim-CLM` | 60M, 200M, 400M, 1.1B | A family of small language models using [Llama-3.1 architecture](https://arxiv.org/abs/2407.21783). <br> We use [`cosmo2-tokenizer`](https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer) and pretrain them on [Fineweb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). |
| Causal language modeling | `Llama-3`| 1B, 3B, 8B, 70B | Meta's [Llama-3](https://arxiv.org/abs/2407.21783) model family |
| Causal language modeling | TBD | TBD | TBD |
| Image generation | TBD | TBD | TBD |
| Image classification | TBD | TBD | TBD |


## Model Training

- Pretraining from scratch

    | Model Names | Supported? |
    |-----------| :--------:|
    | `RoBERTa` | ✅ |
    | `AICrossSim-CLM`, `Llama-3` | ✅ |

- Fine-tuning

    | Model Names | Supported? |
    |-----------| :--------:|
    | `RoBERTa` | ✅ |
    | `AICrossSim-CLM`, `Llama-3` | ⏹️ |

- Evaluation

    | Task | Model Name | Supported? |
    |------|------------| :--------:|
    | Text classification (GLUE) | `RoBERTa` | ✅ |
    | Causal language modeling | `AICrossSim-CLM`, `Llama-3` | ✅ |
    | Benchmarks in [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) | `AICrossSim-CLM`, `Llama-3` | ✅ |


## Model Behavior-Level Simulation

- Transform-aware pretraining from scratch

    | Transform | Model Name | Supported? |
    |---|-----------| :--------:|
    | Random Bitflip | `AICrossSim-CLM`, `Llama-3` | ✅ |
    | Optical Compute | `AICrossSim-CLM`, `Llama-3` | ⏹️ |
    | In-Memory Compute | `AICrossSim-CLM`, `Llama-3` | ⏹️ |
    | Spiking Neural Networks | `AICrossSim-CLM`, `Llama-3` | ⏹️ |

- Post-transform/training evaluation

    | Transform | Task | Model Name | Supported? |
    | --- | ------|------------| :--------:|
    | Random Bitflip | Benchmarks in [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) | `AICrossSim-CLM`, `Llama-3` | ⏹️ |
    | Optical Compute | Benchmarks in [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) | `AICrossSim-CLM`, `Llama-3` | ⏹️ |
    | In-Memory Compute | Benchmarks in [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) | `AICrossSim-CLM`, `Llama-3` | ⏹️ |
    | Spiking Neural Networks | Benchmarks in [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) | `AICrossSim-CLM`, `Llama-3` | ⏹️ |


