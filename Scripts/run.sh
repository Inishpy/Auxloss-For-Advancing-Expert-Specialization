#!/bin/bash
# run.sh – loop through all flan_task files and fine-tune one LoRA adapter per
# task, spreading jobs across all available GPUs in parallel.

# ── Resolve absolute paths so the script works from any working directory ────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # …/Scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                    # …/Auxloss-For-…

# ── Sanity checks ────────────────────────────────────────────────────────────
if [ -z "${CONDA_EXE}" ]; then
    echo "Conda not found. Activate a conda environment before running this script." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ADS-moe

# ── Configurable defaults (override via environment variables) ───────────────
LORA_RANK="${LORA_RANK:-16}"
# OUTPUT_PREFIX and FLAN_TASK_DIR are relative to the repo root by default
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$REPO_ROOT/checkpoints_flan}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/deepseek-moe-16b-chat}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
FLAN_TASK_DIR="${FLAN_TASK_DIR:-$REPO_ROOT/flan_task}"

echo "SCRIPT_DIR    : $SCRIPT_DIR"
echo "REPO_ROOT     : $REPO_ROOT"
echo "FLAN_TASK_DIR : $FLAN_TASK_DIR"
echo "OUTPUT_PREFIX : $OUTPUT_PREFIX"

# ── Verify flan_task directory exists and is non-empty ───────────────────────
if [ ! -d "$FLAN_TASK_DIR" ]; then
    echo "ERROR: flan_task directory not found: $FLAN_TASK_DIR" >&2
    echo "       Set FLAN_TASK_DIR=/path/to/flan_task and re-run." >&2
    exit 1
fi

TASK_FILE_COUNT=$(find "$FLAN_TASK_DIR" -maxdepth 1 -type f | wc -l)
if [ "$TASK_FILE_COUNT" -lt 1 ]; then
    echo "ERROR: No files found inside $FLAN_TASK_DIR" >&2
    exit 1
fi
echo "Found $TASK_FILE_COUNT task file(s) in $FLAN_TASK_DIR"

# ── GPU discovery ────────────────────────────────────────────────────────────
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -lt 1 ]; then
    echo "No CUDA devices detected via nvidia-smi." >&2
    exit 1
fi
echo "Detected $GPU_COUNT GPU(s)."

GPU_IDS=()
for ((i=0; i<GPU_COUNT; i++)); do
    GPU_IDS+=("$i")
done

# ── Training launcher ─────────────────────────────────────────────────────────
run_training() {
    local file="$1"
    local prefix="$2"
    local gpu_id="$3"

    echo "[GPU $gpu_id] Starting task: $prefix  (file: $file)"

    CUDA_VISIBLE_DEVICES="$gpu_id" \
    conda run -n ADS-moe python "$SCRIPT_DIR/Train/batch_train.py" \
        --model_name_or_path    "$MODEL_NAME" \
        --dataset_name          "$file" \
        --output_prefix         "$OUTPUT_PREFIX" \
        --lora_r                "$LORA_RANK" \
        --max_seq_length        "$MAX_SEQ_LENGTH" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --num_train_epochs      "$NUM_EPOCHS" \
        --learning_rate         "$LEARNING_RATE" \
        --warmup_steps          0 \
        --logging_steps         10 \
        --save_total_limit      5 \
        --ref                   "flan"

    echo "[GPU $gpu_id] Finished task: $prefix"
}

# ── Main loop ────────────────────────────────────────────────────────────────
ACTIVE_JOBS=0
GPU_INDEX=0

for file in "$FLAN_TASK_DIR"/*; do
    [ -f "$file" ] || continue          # skip directories / non-files

    filename=$(basename "$file")

    # Derive task prefix (part before ":" if present, otherwise strip extension)
    if [[ "$filename" == *:* ]]; then
        prefix="${filename%%:*}"
    else
        prefix="${filename%%.*}"
    fi

    task_output_dir="$OUTPUT_PREFIX/$prefix"
    echo "Checking: $task_output_dir"

    if [ -d "$task_output_dir" ]; then
        echo "  → Already trained, skipping: $prefix"
        continue
    fi

    GPU_ID="${GPU_IDS[$GPU_INDEX]}"
    echo "  → Assigning '$prefix' to GPU $GPU_ID"

    run_training "$file" "$prefix" "$GPU_ID" &
    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))

    GPU_INDEX=$(( (GPU_INDEX + 1) % GPU_COUNT ))

    # Once every GPU has a job, wait for any one to finish before dispatching more
    if [ "$ACTIVE_JOBS" -ge "$GPU_COUNT" ]; then
        wait -n 2>/dev/null || wait   # 'wait -n' needs bash ≥ 4.3
        ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
    fi
done

# Wait for remaining background jobs
wait
echo "All flan_task training runs complete."