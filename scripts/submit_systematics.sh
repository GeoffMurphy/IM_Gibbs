#!/bin/bash
#SBATCH --job-name=imgibbs_sys
#SBATCH --output=logs/sys_%x_%j.out
#SBATCH --error=logs/sys_%x_%j.err
#SBATCH --partition=Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=16:00:00
#
# One arm of a systematics injection run, as a Slurm job.
#
#   sbatch scripts/submit_systematics.sh <systematic> <arm> [extra args]
#
#   sbatch scripts/submit_systematics.sh onef clean --n-samples 2500
#   sbatch scripts/submit_systematics.sh onef on    --n-samples 2500
#   sbatch scripts/submit_systematics.sh leakage on --rm 1000
#
# Submit one job per arm rather than looping inside one job: the arms are
# independent, Slurm will run them in parallel across nodes, and a failure in
# one does not cost the others. Use the SAME --seed for every arm of a
# comparison or the difference between them is partly chain noise.
#
# On the (70, 45, 250) grid a contaminated arm runs ~6-7 s/sample with 8 CPUs,
# so 2500 samples is ~5 h -- inside the 16 h wall clock with room to spare.
# Signal cubes are thinned to every 10th sample by default (~800 MB per arm);
# Pk and g traces are written every sample and are negligible.
#
# RUN EVERYTHING THROUGH SLURM. The login node is throttled -- a validation
# that takes 287 s on a compute node took 28 minutes there.

set -euo pipefail

SYSTEMATIC=${1:?"Usage: sbatch scripts/submit_systematics.sh <systematic> <arm> [args...]"}
ARM=${2:?"Usage: sbatch scripts/submit_systematics.sh <systematic> <arm> [args...]"}
shift 2

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# --- Environment -----------------------------------------------------------
# IMGIBBS_VENV lets you point at a different interpreter without editing this
# file. The default is the SKA environment, which already carries numpy,
# scipy, pyccl and tqdm -- imgibbs needs nothing else. fastbox is optional:
# imgibbs.grid only imports it to CHECK that the cosmology has not drifted,
# and skips the check when it is absent.
VENV=${IMGIBBS_VENV:-$HOME/ska/.venv}
PYTHON="$VENV/bin/python"
[ -x "$PYTHON" ] || { echo "No interpreter at $PYTHON (set IMGIBBS_VENV)" >&2; exit 1; }

# The MeerKLASS cubes are not redistributed with the repository. On ilifu they
# already exist under the Sampling Nb archive; IMGIBBS_DATA points imgibbs.data
# at whatever directory holds them.
export IMGIBBS_DATA=${IMGIBBS_DATA:-$PWD/data}

# Keep BLAS threads equal to the allocation. Left unset, OpenBLAS grabs every
# core on the node and fights the other arms for them.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

OUT=${IMGIBBS_OUT:-outputs/${SYSTEMATIC}_run}

echo "host        : $(hostname)"
echo "job         : ${SLURM_JOB_ID:-none}  cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "python      : $PYTHON"
echo "data        : $IMGIBBS_DATA"
echo "systematic  : $SYSTEMATIC"
echo "arm         : $ARM"
echo "out         : $OUT"
echo

srun "$PYTHON" scripts/systematics_injection.py \
    --systematic "$SYSTEMATIC" --arm "$ARM" --out "$OUT" "$@"
