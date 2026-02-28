#!/bin/bash
# Slurm job array for the first 5 FLAN tasks (one task per array index).
# Submit with: sbatch Scripts/train_flan.sh

#SBATCH --job-name=flan_lora
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --account=Soltoggio2025a
#SBATCH --array=0-29
#SBATCH --output=Scripts/logs/%x.%A_%a.out
#SBATCH --error=Scripts/logs/%x.%A_%a.err

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

module purge
module load CUDA/12.4.0
source ~/.bashrc
conda activate ADS-moe

# ── Resolve absolute paths so the script works from any working directory ────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # …/Scripts
# Prefer shared filesystem repo path when running on HPC; fallback to script-relative.
DEFAULT_REPO_ROOT="/data/home/co/coimd/Auxloss-For-Advancing-Expert-Specialization"
if [ -d "$DEFAULT_REPO_ROOT" ]; then
    REPO_ROOT="$DEFAULT_REPO_ROOT"
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                # …/Auxloss-For-…
fi

# ── Configurable defaults (override via environment variables) ───────────────
LORA_RANK="${LORA_RANK:-16}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$REPO_ROOT/checkpoints_flan}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/deepseek-moe-16b-chat}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
FLAN_TASK_DIR="${FLAN_TASK_DIR:-$REPO_ROOT/flan_task}"
# Control how many tasks (files) this job array covers.


# Example: NUM_TASKS=10 sbatch --array=0-9 Scripts/train_flan.sh
NUM_TASKS="${NUM_TASKS:-30}"
# Optionally stage flan_task to node-local scratch for faster I/O on HPC.
if [ -n "$SLURM_TMPDIR" ] && [ -d "$FLAN_TASK_DIR" ]; then
    STAGE_FLAN_TASK="${STAGE_FLAN_TASK:-1}"
    if [ "$STAGE_FLAN_TASK" = "1" ]; then
        STAGED_FLAN_TASK_DIR="$SLURM_TMPDIR/flan_task"
        if [ ! -d "$STAGED_FLAN_TASK_DIR" ]; then
            cp -a "$FLAN_TASK_DIR" "$STAGED_FLAN_TASK_DIR"
        fi
        FLAN_TASK_DIR="$STAGED_FLAN_TASK_DIR"
    fi
fi

echo "SCRIPT_DIR    : $SCRIPT_DIR"
echo "REPO_ROOT     : $REPO_ROOT"
echo "FLAN_TASK_DIR : $FLAN_TASK_DIR"
echo "OUTPUT_PREFIX : $OUTPUT_PREFIX"

# ── Verify flan_task directory exists and has at least NUM_TASKS files ───────
if [ ! -d "$FLAN_TASK_DIR" ]; then
    echo "ERROR: flan_task directory not found: $FLAN_TASK_DIR" >&2
    exit 1
fi

mapfile -t ALL_TASK_FILES < <(find "$FLAN_TASK_DIR" -maxdepth 1 -type f | sort)
if [ "${#ALL_TASK_FILES[@]}" -lt "$NUM_TASKS" ]; then
    echo "ERROR: Expected at least $NUM_TASKS files in $FLAN_TASK_DIR, found ${#ALL_TASK_FILES[@]}" >&2
    exit 1
fi

TASK_FILES=("${ALL_TASK_FILES[@]:0:$NUM_TASKS}")

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. This script must be run via sbatch." >&2
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_TASKS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID out of range (0-$((NUM_TASKS-1))): $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

FILE="${TASK_FILES[$SLURM_ARRAY_TASK_ID]}"
FILENAME="$(basename "$FILE")"

# Derive task prefix (part before ":" if present, otherwise strip extension)
if [[ "$FILENAME" == *:* ]]; then
    PREFIX="${FILENAME%%:*}"
else
    PREFIX="${FILENAME%%.*}"
fi

TASK_OUTPUT_DIR="$OUTPUT_PREFIX/$PREFIX"
echo "Selected task file: $FILE"
echo "Task prefix       : $PREFIX"
echo "Output dir        : $TASK_OUTPUT_DIR"

if [ -d "$TASK_OUTPUT_DIR" ]; then
    echo "Already trained, skipping: $PREFIX"
    exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

conda run -n ADS-moe python "$REPO_ROOT/Scripts/Train/batch_train.py" \
    --model_name_or_path         "$MODEL_NAME" \
    --dataset_name               "$FILE" \
    --output_prefix              "$OUTPUT_PREFIX" \
    --lora_r                     "$LORA_RANK" \
    --max_seq_length             "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --num_train_epochs           "$NUM_EPOCHS" \
    --learning_rate              "$LEARNING_RATE" \
    --warmup_steps               0 \
    --logging_steps              10 \
    --save_total_limit           5 \
    --ref                        "flan"

echo "=========================================="
echo "Task completed at: $(date)"
echo "=========================================="
