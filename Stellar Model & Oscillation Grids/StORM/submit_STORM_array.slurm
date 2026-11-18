#!/bin/bash -l
#SBATCH --account=intro_vsc39211
#SBATCH --clusters=wice
#SBATCH --partition=batch_sapphirerapids
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=kian.tedeskino@student.kuleuven.be
#SBATCH --job-name=storm_b1
#SBATCH --array=1-675%120
#SBATCH --error=std/storm_b1_%A_%a.err
#SBATCH --output=std/storm_b1_%A_%a.out
#
# One array task = one model = one line of the worklist CSV.
#
# SUBMIT:
#   mkdir -p std
#   sbatch submit_STORM_array.slurm
#
# Array bound must equal the data rows in the worklist:
#   echo $(( $(wc -l < worklists/worklist_storm_${BATCH_TAG}.csv) - 1 ))
#
# 750 rows is within the 500-job cap only because %125 limits concurrency;
# if sbatch rejects it, split:  --array=1-375%125  then  --array=376-750%125
#
# PER BATCH, change: BATCH_TAG, --array bound, --job-name, both log names.

set -uo pipefail

export BATCH_TAG="batch1_M1.8-M2.0"

export PROJECT_DATA="$VSC_DATA/HD35929/grid_run_storm"
export PROJECT_SCRATCH="$VSC_SCRATCH/HD35929/StORM_grid"

export STORM_BIN="$VSC_DATA/StORM/storm"
export STORM_RUNNER="$PROJECT_DATA/run_storm_model.py"
export STORM_PROFILES="$PROJECT_DATA/worklists/storm_profiles_${BATCH_TAG}.csv"

export PROFILE_ROOT="$VSC_SCRATCH/HD35929/MESA_grid/extracted/${BATCH_TAG}"
export OUTPUT_ROOT="$PROJECT_SCRATCH/${BATCH_TAG}"

# Converged scan window (10-30 is off by 0.01 c/d at omega/omega_crit = 0.55).
export STORM_FREQ_MIN=4
export STORM_FREQ_MAX=38
export STORM_N_SCAN=850
export STORM_TIMEOUT=900
export STORM_MAX_OMEGA_CRIT=0.75

export STORM_SET_ROTATION=0        # fatal on models that already rotate
export STORM_SURFACE_ROTATION=1    # deform reads the model's own surface rate
export STORM_DISABLE_SYMMETRIC=0   # handled automatically by newer StORM
export STORM_FAMILIES=even         # even and odd are options,note for P2 StORM only couples even and odd families

export STORM_PROPERTIES="frequency,radial-order,degree,azimuthal-order,deformed-frequency,deformed-eigenvector"

export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORKLIST="$PROJECT_DATA/worklists/worklist_storm_${BATCH_TAG}.csv"

module load Python/3.11.3-GCCcore-12.3.0
module load zstd/1.5.5-GCCcore-12.3.0
command -v zstd >/dev/null || { echo "FATAL: zstd not on PATH" >&2; exit 1; }
[[ -x "$STORM_BIN" ]] || { echo "FATAL: storm not executable at $STORM_BIN" >&2; exit 1; }
[[ -f "$WORKLIST" ]] || { echo "FATAL: worklist not found: $WORKLIST" >&2; exit 1; }

# Line 1 is the header, so task N reads line N+1.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$WORKLIST")
if [[ -z "$LINE" ]]; then
    echo "FATAL: no row for array index ${SLURM_ARRAY_TASK_ID} in $WORKLIST" >&2
    exit 1
fi

model=$(echo "$LINE" | cut -d, -f1)
n_profiles=$(echo "$LINE" | cut -d, -f2)

echo "array task : ${SLURM_ARRAY_TASK_ID}"
echo "model      : ${model}"
echo "profiles   : ${n_profiles}"
echo "host       : $(hostname)"
echo "threads    : ${RAYON_NUM_THREADS}"

bash "$PROJECT_DATA/run_STORM.sh" "$model" "$n_profiles"
