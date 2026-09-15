#!/bin/bash
#SBATCH --job-name=sc4001-partb
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs

source .venv/bin/activate

python partb.py