#!/bin/bash
#SBATCH --job-name=gibbs_fg
#SBATCH --output=logs/gibbs_LP%j.out
#SBATCH --error=logs/gibbs_LP%j.err
#SBATCH --partition=Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=12:00:00

# Usage: sbatch scripts/submit_gibbs.sh <n_modes> [extra run_gibbs.py args]
#
#   sbatch scripts/submit_gibbs.sh 6
#   sbatch scripts/submit_gibbs.sh 6 --n-samples 500 --seed 42
#
# On the (70, 45, 250) grid one sample takes ~4 s, so 500 samples is well
# inside the 12 h wall clock. Traces land in outputs/samples/ -- roughly 13 MB
# per sample for x_sample alone, so 500 samples is ~6.5 GB. Point --out at
# scratch rather than home if your quota is tight.

set -euo pipefail

N_MODES=${1:?"Usage: sbatch scripts/submit_gibbs.sh <n_modes> [args...]"}
shift

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# Activate your Python environment here, e.g.:
#   source /path/to/venv/bin/activate
#   module load python/3.12
#
# IMGIBBS_DATA must point at the directory holding L2021_polished_cube.npy if
# it is not in ./data -- see README, "Data".
# export IMGIBBS_DATA=/path/to/cubes

echo "Running Gibbs sampling with ${N_MODES} foreground modes"
python scripts/run_gibbs.py "${N_MODES}" "$@"
