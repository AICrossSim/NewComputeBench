conda activate new-compute
export CX_DATA_HOME="/data/models/cx922"
export CX_PROJECT_HOME="/home/cx922/NewComputeBench"

export HF_HOME="${CX_DATA_HOME}/hf_home"
export TRANFORMERS_CACHE="${CX_DATA_HOME}/hf_transformers"
export DATA_HOME="${CX_DATA_HOME}/data"
export HF_HUB_CACHE="${CX_DATA_HOME}/hf_home/hub"

export CUDA_DEVICE="2,3"
export PROC_NUM="2"
