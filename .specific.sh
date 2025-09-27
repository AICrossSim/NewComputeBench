conda activate new-compute
export CX_DATA_HOME="/data/cx922"
export CX_PROJECT_HOME="/home/jianyicheng/cx922/NewComputeBench"
export CUDA_DEVICE="0,1,2,3,6,7"
export PROC_NUM="2"

# HuggingFace environment variables
export HF_HOME="${CX_DATA_HOME}/hf_home"