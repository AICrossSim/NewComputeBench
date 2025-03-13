# README

**NewComputeBench** is a project to develop a benchmark suite for the new compute paradigm (Spiking neural networks, Optical computation, In-Memory computation, etc). The project is divided into three main components:
- Model Training
- Model Behavior-Level Simulation
- Hardware-Performance Simulation

## Model Training

### LLMs

We adopt Llama-3 architecture and aim to support the following features:

- Pretraining
- Generation (inference)
- `🚧 TODO`: Parameter-efficient fine-tuning;
- `🚧 TODO` `🐌 LowPriority`: Supervised-fine-tuning
- `🚧 TODO` `🐌 LowPriority`: Evaluation

#### PreTraining

The LLM pretraining is built on top of [torchtitan](https://github.com/pytorch/torchtitan).

- Model architecture: [`Llama3`](/src/torchtitan/models/llama/model.py)
- Model configs: [`60M`, `200M`, `400M`, `600M`, `1.1B`](src/aixsim_models/llm/model_flavors.py)
- Datasets: [`HuggingFaceFW/fineweb`](/src/aixsim_models/llm/pretrain_data.py)

#### Generation

We recommend using the HuggingFace Transformers library for generation tasks.
We provide a script to convert the torchtitan checkpoint to a HuggingFace checkpoint (See [this file](/experiments/llm-digital/pretrain/README.md)).


#### Parameter-Efficient Fine-tuning
- `🚧 TODO`: For models larger than 1.1B, we fine-tune pretrained checkpoints.
  - LoRA fine-tuning data
  - LoRA fine-tuning scripts

## Model Behavior Simulation

`🚧 TODO`

## Hardware-Performance Simulation

`🚧 TODO`
