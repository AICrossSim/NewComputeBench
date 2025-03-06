#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=18:mem=200gb:ngpus=2:gpu_type=A100
#PBS -N clm-60m

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE
module purge

local_rank="0"
config="../configs/aixsim-60M-hpc.yaml"
checkpoint_folder="$(date +'%Y-%m-%d_%H')"
nvidia-smi >> nvidia_smi.log
export PYTHONPATH="${PYTHONPATH}:/gpfs/home/zz7522/Projects/NewComputeBench/src"
conda run -n new-compute --no-capture-output \
    torchrun --nnodes=1 --nproc_per_node=2  \
    --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter "${local_rank}" \
    --role rank --tee 3 \
    run.py pretrain.py --config "${config}" --checkpoint_args.folder "${checkpoint_folder}"