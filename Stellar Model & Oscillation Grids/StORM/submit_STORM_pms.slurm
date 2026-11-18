#!/bin/bash -l
#SBATCH --account=intro_vsc39211
#SBATCH --clusters=wice
#SBATCH --partition=batch_sapphirerapids
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=kian.tedeskino@student.kuleuven.be
#SBATCH --job-name=storm_pms_box_3sig
#SBATCH --error=std/storm_pms_box_3sig_%j.err
#SBATCH --output=std/storm_pms_box_3sig_%j.out
#
# ---------------------------------------------------------------------
# StORM on the PMS profiles of a Teff / L box (omega = 0.00 models), with
# a constant rotation imposed at each MS-grid label 0.00, 0.05 ... 0.75
# (see LABEL_TO_DYNAMICAL in run_storm_pms_model.py).
#
# box 3sig: 340 models, 7060 profiles x 9 rotations = 63540 runs at
# ~5 s each on one thread (~90 core-h, ~1 h wall on 96 cores). Mode
# profiles are switched off, so ~0.65 MB per run, ~40 GB in total
# (with xi_r, xi_h it would be ~10 MB per run, ~600 GB).
#
# ONE job, not an array:
#   1. run every model of the worklist, NPAR at a time, into a temp
#      folder  $VSC_SCRATCH/HD35929/StORM_grid/tmp_<tag>/<tag>/<model>/
#   2. if every model succeeded, tar the temp folder into
#      StORM_grid/tarballs/<tag>_results.tar (+ .sha256), check the
#      archive, and delete the temp folder.
#   If any model failed, nothing is tarred or deleted: resubmit and the
#   runner skips every run whose HDF5 output already exists.
#
# SUBMIT (from grid_run_storm/):
#   mkdir -p std
#   sbatch submit_STORM_pms.slurm
#
# FOR ANOTHER BOX: change BATCH_TAG and --job-name/--error/--output.
# ---------------------------------------------------------------------

set -uo pipefail

export BATCH_TAG="pms_box_3sig"

export PROJECT_DATA="$VSC_DATA/HD35929/grid_run_storm"
export PROJECT_SCRATCH="$VSC_SCRATCH/HD35929"

export STORM_BIN="$VSC_DATA/Codes/StORM/storm"
export RUNNER="$PROJECT_DATA/run_storm_pms_model.py"

export PROFILE_ROOT="$PROJECT_SCRATCH/MESA_grid/extracted/${BATCH_TAG}"
WORKLIST="$PROJECT_DATA/worklists/worklist_storm_${BATCH_TAG}.csv"

# Same scan as the MS StORM grid.
export STORM_PMS_LABELS=0.00,0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75
export STORM_FREQ_MIN=4
export STORM_FREQ_MAX=38
export STORM_N_SCAN=850
export STORM_FAMILIES=even
export STORM_TIMEOUT=900
# No mode profiles (xi_r, xi_h): they are ~90% of each file. Frequencies,
# deformed frequencies and deformed eigenvectors are still written.
export STORM_PROFILES_OUTPUT=""

TMP_ROOT="$PROJECT_SCRATCH/StORM_grid/tmp_${BATCH_TAG}"
export OUTPUT_ROOT="$TMP_ROOT/${BATCH_TAG}"
TARBALL="$PROJECT_SCRATCH/StORM_grid/tarballs/${BATCH_TAG}_results.tar"

# StORM barely speeds up past 2 threads (5.0 s on 1, 2.7 s on 8), so run
# single-threaded and put the cores into concurrent models instead.
export RAYON_NUM_THREADS=1
NPAR=$(( SLURM_CPUS_PER_TASK / RAYON_NUM_THREADS ))

# ---------------------------------------------------------------------
# ENVIRONMENT -- the job starts clean, so load everything here.
# ---------------------------------------------------------------------
module load Python/3.11.3-GCCcore-12.3.0

for f in "$WORKLIST" "$STORM_BIN" "$RUNNER"; do
    [[ -e "$f" ]] || { echo "FATAL: not found: $f" >&2; exit 1; }
done
[[ -d "$PROFILE_ROOT" ]] || { echo "FATAL: no grid at $PROFILE_ROOT" >&2; exit 1; }
[[ -e "$TARBALL" ]] && { echo "FATAL: $TARBALL already exists" >&2; exit 1; }

export LOG_DIR="$OUTPUT_ROOT/job_logs"
export ERR_DIR="$OUTPUT_ROOT/job_errs"
mkdir -p "$LOG_DIR" "$ERR_DIR"
export FAILED_LIST="$TMP_ROOT/failed_models.txt"
rm -f "$FAILED_LIST"

NMODELS=$(( $(wc -l < "$WORKLIST") - 1 ))
echo "batch      : ${BATCH_TAG}"
echo "models     : ${NMODELS}, ${NPAR} at a time x ${RAYON_NUM_THREADS} thread(s)"
echo "rotations  : ${STORM_PMS_LABELS}"
echo "grid       : ${PROFILE_ROOT}"
echo "temp out   : ${OUTPUT_ROOT}"
echo "start      : $(date)"

# ---------------------------------------------------------------------
# 1. RUN -- one runner per model, NPAR concurrently
# ---------------------------------------------------------------------
tail -n +2 "$WORKLIST" | cut -d, -f1 | \
    xargs -P "$NPAR" -I{} bash -c '
        if python3 "$RUNNER" --model "$1" > "$LOG_DIR/$1.log" 2> "$ERR_DIR/$1.err"; then
            echo "  done   $1"
        else
            echo "  FAILED $1"
            echo "$1" >> "$FAILED_LIST"
        fi' _ {}

echo "runs done  : $(date)"

# Per-run tally over all models (status is column 5 of run_log.csv).
# 'diverged' = deformation blew up near critical rotation; not a failure.
cat "$OUTPUT_ROOT"/M*/run_log.csv 2>/dev/null | grep -v '^profile,' | \
    cut -d, -f5 | sort | uniq -c

if [[ -s "$FAILED_LIST" ]]; then
    echo "ERROR: $(wc -l < "$FAILED_LIST") model(s) failed, see $FAILED_LIST" >&2
    echo "Temp folder kept; resubmit to finish (completed runs are skipped)." >&2
    exit 1
fi

# ---------------------------------------------------------------------
# 2. ARCHIVE -- tar, verify, then delete the temp folder
# HDF5 output only compresses ~14% with zstd, so the tar is uncompressed.
# ---------------------------------------------------------------------
mkdir -p "$(dirname "$TARBALL")"
tar -cf "$TARBALL" -C "$TMP_ROOT" "$BATCH_TAG" \
    || { echo "ERROR: tar failed, temp folder kept" >&2; rm -f "$TARBALL"; exit 1; }

N_DISK=$(find "$OUTPUT_ROOT" | wc -l)
N_TAR=$(tar -tf "$TARBALL" | wc -l)
if [[ "$N_DISK" -ne "$N_TAR" ]]; then
    echo "ERROR: tarball has $N_TAR entries, disk has $N_DISK; temp folder kept" >&2
    exit 1
fi
( cd "$(dirname "$TARBALL")" && sha256sum "$(basename "$TARBALL")" > "$(basename "$TARBALL").sha256" )

rm -rf "$TMP_ROOT"
echo "archived   : ${TARBALL} (${N_TAR} entries, $(du -h "$TARBALL" | cut -f1))"
echo "removed    : ${TMP_ROOT}"
echo "stop       : $(date)"
