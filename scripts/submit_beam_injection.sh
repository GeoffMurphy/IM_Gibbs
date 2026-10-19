#!/bin/bash
#SBATCH --job-name=gibbs_beam_inj
#SBATCH --output=logs/beam_inj_%x_%j.out
#SBATCH --error=logs/beam_inj_%x_%j.err
#SBATCH --partition=Main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=3-00:00:00

# One arm of the beam known-answer test (scripts/beam_injection.py).
#
#   python scripts/beam_injection.py build          # once, writes the mocks
#   sbatch -J beam      scripts/submit_beam_injection.sh beam
#   sbatch -J nocollapse scripts/submit_beam_injection.sh nocollapse
#   sbatch -J nobeam    scripts/submit_beam_injection.sh nobeam
#   sbatch -J control   scripts/submit_beam_injection.sh control
#   python scripts/beam_injection.py summarise
#
# Arms:
#   beam        beamed mock, beam in the model, collapsed S step   (the method)
#   nocollapse  beamed mock, beam in the model, S from every mode  (old S step)
#   nobeam      beamed mock, NO beam in the model                  (misspecified)
#   control     unbeamed mock, no beam in the model                (baseline)
#
# Beamed arms run at ~45 s/sample on 12 cores (the foreground block of the
# preconditioner is the bottleneck, see docs/STATUS.md); unbeamed at ~9 s.
# Extra arguments go to run_gibbs.py, e.g. --n-samples 1500.
#
# PYTHON defaults to the validated ilifu environment; override to taste.

set -euo pipefail

ARM=${1:?"Usage: sbatch scripts/submit_beam_injection.sh <beam|nocollapse|nobeam|control> [args...]"}
shift

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

PYTHON=${PYTHON:-$HOME/ska/.venv/bin/python}
OUT=${OUT:-outputs/beam_injection}
TABLE=${TABLE:-data/beam_uhf_table.npz}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

case "$ARM" in
  beam)       DATA=mock_beamed;  EXTRA=(--beam-table "$TABLE") ;;
  nocollapse) DATA=mock_beamed;  EXTRA=(--beam-table "$TABLE" --no-collapse) ;;
  nobeam)     DATA=mock_beamed;  EXTRA=() ;;
  control)    DATA=mock_control; EXTRA=() ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

echo "arm $ARM on $(hostname), $(date)"
"$PYTHON" -u scripts/run_gibbs.py 6 --data "$OUT/$DATA.npy" --out "$OUT/$ARM" \
    --seed 1 --save-x-every 50 "${EXTRA[@]}" "$@"
echo "done $(date)"
