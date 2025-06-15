#!/bin/bash
#SBATCH --account=PAS2956
#SBATCH --job-name=word_puzzle_sft
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL

module reset
module load cuda/12.4.1 

source /users/PAS2956/elijahmansur/open-r1/openr1/bin/activate

# Compute how many GPUs we actually have
# SLURM sets CUDA_VISIBLE_DEVICES for you; count those entries:
NUM_PROC=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# echo for debugging
echo "Launching $NUM_PROC processes on GPUs: $CUDA_VISIBLE_DEVICES"

# Configure HF to use PFS scratch 
SCRATCH=/fs/scratch/PAS2956/elijahmansur
HF_HUB_CACHE=$SCRATCH/hf_cache/hub
HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
HF_METRICS_CACHE=$SCRATCH/hf_cache/metrics

mkdir -p $HF_HUB_CACHE $HF_DATASETS_CACHE $HF_METRICS_CACHE

export HF_HUB_CACHE HF_DATASETS_CACHE HF_METRICS_CACHE

cd /users/PAS2956/elijahmansur/open-r1

# Launch training, overriding zero3.yaml’s num_processes
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  --num_processes $NUM_PROC \
  src/open_r1/sft.py \
  --config recipes/Qwen2.5-1.5B-Instruct/sft/config_wordpuzzle.yaml \
  --output_dir sft_output_3
