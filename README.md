# README

**NewComputeBench** is a project to develop a benchmark suite for the new compute paradigm (Spiking neural networks, Optical computation, In-Memory computation, etc). The project is divided into three main components:
- Model Training
- Model Behavior-Level Simulation
- Hardware-Performance Simulation

**🔖 For tutorials and examples, please refer to [this site](https://aicrosssim.github.io/NewComputeBench/)**.

## Model Training

### LLMs

We adopt Llama-3 architecture and aim to support the following features:

- Pretraining
- Generation (inference)
- `🚧 TODO`: Parameter-efficient fine-tuning;
- `🚧 TODO` `🐌 LowPriority`: Supervised-fine-tuning
- Evaluation

#### PreTraining

The LLM pretraining is built on top of [torchtitan](https://github.com/pytorch/torchtitan).

- Model architecture: [`Llama3`](/src/torchtitan/models/llama/model.py)
- Model configs: [`60M`, `200M`, `400M`, `1.1B`](src/aixsim_models/llm/model_flavors.py)
- Datasets: [`HuggingFaceFW/fineweb`](/src/aixsim_models/llm/pretrain_data.py)
- HuggingFace checkpoints: [AICrossSim](https://huggingface.co/AICrossSim)

#### Generation

We recommend using the HuggingFace Transformers library for generation tasks.
We provide a script to convert the torchtitan checkpoint to a HuggingFace checkpoint (See [this file](/experiments/llm-digital/pretrain/README.md)).


#### Parameter-Efficient Fine-tuning
- `🚧 TODO`: For models larger than 1.1B, we fine-tune pretrained checkpoints.
  - LoRA fine-tuning data
  - LoRA fine-tuning scripts

## Model Behavior Simulation

- [Random bitflip](/experiments/llm-bitflip/)
  - Post-training bitflip transform
  - Bitflip-aware pretraining
- Optical compute `🚧 TODO`
- Spiking neural networks `🚧 TODO`
- In-memory compute `🚧 TODO
`

## Hardware-Performance Simulation

`🚧 TODO`
