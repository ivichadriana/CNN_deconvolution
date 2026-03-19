#!/bin/sh

#SBATCH --job-name=tune_TRANS_MNIST
#SBATCH --account=amc-general
#SBATCH --output=output_TRANS_MNIST.log
#SBATCH --error=error_tune_TRANS_MNIST.err
#SBATCH --mail-type=ALL
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

echo "Activating environment..."
source ~/.bashrc
conda deactivate
conda activate env_cnn

sleep 5

# -----------------------------------------------------------------------------
# Change to the directory containing this Slurm script and 'tune.py'
cd "$(dirname "$0")"
echo "Running in directory: $(pwd)"

# Model/Dataset parameters
MODEL_TYPE="TRANS"
DATASET="MNIST"
NUM_ITERATIONS="1"

# HPC-specific large-scratch path
#   If you move HPCs, you only need to update this prefix
SCRATCH_PREFIX="/scratch/alpine/$USER"

# Subdirectories for Ray
TMP_DIR="${SCRATCH_PREFIX}/rayt"
WORKING_DIR="${SCRATCH_PREFIX}/rayw"

# Output JSON location is still relative to *this* directory:
OUTPUT_PATH="./data/best_configs_allshuffle_${MODEL_TYPE}_${DATASET}.json"

# PCam data path might be large, so let's also keep that on scratch
PCAM_DATA_PATH="${SCRATCH_PREFIX}/data/pcamv1"

echo "TMP_DIR: $TMP_DIR"
echo "WORKING_DIR: $WORKING_DIR"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "PCAM_DATA_PATH: $PCAM_DATA_PATH"

export RAY_TMPDIR="$TMP_DIR"

echo "Running script with model_type=$MODEL_TYPE, dataset=$DATASET..."
srun $(which python) tune.py \
    --model_type "$MODEL_TYPE" \
    --dataset "$DATASET" \
    --tmp_dir "$TMP_DIR" \
    --working_dir "$WORKING_DIR" \
    --output_path "$OUTPUT_PATH" \
    --pcam_data_path "$PCAM_DATA_PATH" \
    --num_iterations "$NUM_ITERATIONS"

sleep 5
conda deactivate
