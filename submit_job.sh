#!/bin/bash
#SBATCH --job-name=geovit_run
#SBATCH --partition=LocalQ
#SBATCH --account=default
#SBATCH --gres=shard:16
#SBATCH --ntasks=1
#SBATCH --output=geovit_run_%j.out
#SBATCH --error=geovit_run_%j.err

# Navigate to the directory where you submitted the job
cd $SLURM_SUBMIT_DIR

# Load the required modules exactly as your HPC needs them
module load compilers/anaconda3-2024.06
module load libs/cuda-12.8

# Initialize and activate conda
source /apps/compilers/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate urban_env

# Run the Python pipeline (Train the model and then run Inference)
python main.py --step train
python main.py --step infer
