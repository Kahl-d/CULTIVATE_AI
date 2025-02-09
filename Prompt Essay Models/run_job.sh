#!/bin/bash
#SBATCH --job-name=tacit_train         # Job name
#SBATCH --output=logs/output_%j.log    # Standard output log (%j is replaced by the job ID)
#SBATCH --error=logs/error_%j.log      # Error log (%j is replaced by the job ID)
#SBATCH --partition=gpucluster         # Use GPU cluster for training
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task=4              # Request 4 CPU cores per GPU
#SBATCH --time=3:59:00                  # Must be under 4 hours
#SBATCH --mail-type=END,FAIL           # Mail notifications for job completion or failure
#SBATCH --mail-user=kkhan@sfsu.edu  # Replace with your actual email

# Activate Conda environment
source ../../../miniconda/bin/activate llm_env



# Run the Resistance model training script
python3 pc-model-t5-test.py
# Deactivate the environment (not necessary for Conda, but keeping it for cleanliness)
conda deactivate
