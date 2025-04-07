# Environment Setup

## Prerequisites

- Linux or WSL2
- CUDA-enabled GPU
- [MiniConda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) (for installing Cuda-Toolkit)

!!! info "Our Environment Setup for Reference"
    We run all the experiments on linux machines and did not test on Windows.

    Here are a few environment we have tested for reference:

    - NVIDIA A6000 48GBx8, Ubuntu 24.04, CUDA 12.4
    - NVIDIA H100 96GBx2, Red Hat Enterprise Linux 9.5, CUDA 12.6.
    - NVIDIA H100 80GBx8, Ubuntu 24.04, CUDA 12.4

## Environment Setup

1. Config SSH key for GitHub. One of the dependencies, [MASE](https://github.com/DeepWok/mase), requires SSH to clone and install. Please set up `~/.ssh/config` accordingly (refer to [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)).

2. Clone the project repository

    ```bash
    git clone https://github.com/AICrossSim/NewComputeBench.git
    cd NewComputeBench
    git submodule update --init
    ```

3. Create a new conda environment

    ```bash
    conda env create -f environment.yaml
    ```

4. Activate the new environment and install required packages

    ```bash
    conda activate new-compute
    ```

    We recommend check if the python and pip in `$PATH` are from the conda environment:
    ```bash
    which python
    which pip
    ```

    Then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

5. (Optional) You may want to log in [Wandb](https://wandb.ai/site/) to track the training logs.

    ```bash
    wandb login
    ```
