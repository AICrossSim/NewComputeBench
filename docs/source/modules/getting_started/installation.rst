Installation
============

Prerequisites
-------------

- Linux or WSL2
- A CUDA-enabled GPU
- `MiniConda <https://www.anaconda.com/docs/getting-started/miniconda/install>`_ or
  `Anaconda <https://www.anaconda.com/docs/getting-started/anaconda/install>`_
  (required to install the CUDA Toolkit)

.. admonition:: Tested environments

   All experiments are run on Linux. Windows is not tested.

   - NVIDIA A6000 48 GB × 8, Ubuntu 24.04, CUDA 12.4
   - NVIDIA H100 96 GB × 2, Red Hat Enterprise Linux 9.5, CUDA 12.6
   - NVIDIA H100 80 GB × 8, Ubuntu 24.04, CUDA 12.4
   - NVIDIA H200 141 GB × 8, Ubuntu 24.04, CUDA 12.4


Step-by-Step Setup
------------------

1. **Configure SSH access to GitHub.**
   One dependency — `MASE <https://github.com/DeepWok/mase>`_ — requires SSH to clone.
   Follow GitHub's guide on
   `Connecting to GitHub with SSH <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`_
   and ensure ``~/.ssh/config`` is set up correctly.

2. **Clone the repository.**

   .. code-block:: bash

      git clone https://github.com/AICrossSim/NewComputeBench.git
      cd NewComputeBench
      git submodule update --init

3. **Create the conda environment.**

   .. code-block:: bash

      conda env create -f environment.yaml

4. **Activate the environment and install dependencies.**

   .. code-block:: bash

      conda activate new-compute

   Install the required packages. Choose one option:

   **Option 1 — uv** (recommended, assumes CUDA is pre-installed on the system):

   .. code-block:: bash

      uv python install        # reads .python-version (Python 3.11)
      uv venv
      source .venv/bin/activate
      uv pip install -r requirements.txt
      uv pip install -e ./submodules/mase

   **Option 2 — conda + pip** (use this if CUDA is not pre-installed):

   .. code-block:: bash

      conda env create -f environment.yaml
      conda activate new-compute
      pip install -r requirements.txt
      pip install -e ./submodules/mase

   .. note::

      The `MASE <https://github.com/DeepWok/mase>`_ submodule provides the quantization
      backend used by PIM and other hardware simulation passes.

   .. note::

      ``uv`` can be installed with ``pip install uv`` or via the standalone installer:

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

5. **(Optional) Log in to Weights & Biases** to track experiment metrics.

   .. code-block:: bash

      wandb login
