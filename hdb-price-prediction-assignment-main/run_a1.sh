#!/bin/bash
#SBATCH --job-name=hdb-a1
#SBATCH --output=outputs/slurm-%j.out
#SBATCH --error=outputs/slurm-%j.err
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p outputs

source .venv/bin/activate

python src/qa1.py