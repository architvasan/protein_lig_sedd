#!/bin/bash

# ==============================================================================
#                      RUN SCRIPT FOR PROTLIG_DD SPECIAL TRAINING
# ==============================================================================
# This script configures and launches the training process for the 
# Protein-Ligand Shared Diffusion model using the 'special' training script.
#
# How to use:
# 1. Modify the variables in the "DEFINING VARIABLES" section below.
# 2. Make the script executable: chmod +x run_training_special.sh
# 3. Run the script: ./run_training_special.sh
# ==============================================================================


################################################################################
### 1. DEFINING VARIABLES (CONFIGURE YOUR EXPERIMENT HERE)
################################################################################

# --- Project and Environment Directories ---
# WORK_DIR should be the absolute path to the root of your project directory.
# ENV_DIR should be the path to your Conda environment.
WORK_DIR="/eagle/projects/FoundEpidem/xlian/protein_lig_sedd/protlig_dd/test"
ENV_DIR="$WORK_DIR/prot_lig_sedd" # Or your new environment name, e.g., protlig_dd_new

# --- Configuration File ---
# Path to the YAML configuration file for the model and training parameters.
CONFIG_FILE="$WORK_DIR/configs/config.yaml"

# --- Weights & Biases (WandB) Logging ---
# Set the project and a unique name for this specific run.
WANDB_PROJECT="protlig_sedd_special_runs"
WANDB_NAME="special_run_$(date +%Y%m%d_%H%M%S)" # Example: special_run_20250905_203000

# --- Data and Model IDs ---
# Path to the preprocessed training data file.
DATA_FILE="$WORK_DIR/input_data/merged_plinder.pt"

# Hugging Face model IDs for the pretrained encoders.
MOL_EMB_ID="ibm/MoLFormer-XL-both-10pct"
PROT_EMB_ID="facebook/esm2_t30_150M_UR50D"

# --- Hardware and Reproducibility ---
# The CUDA device ID to use for training.
# NOTE: This is the ID within the allocation. If you request one GPU, it will often be 'cuda:0'.
# The CUDA_VISIBLE_DEVICES environment variable below handles the physical device mapping.
DEVICE_ID="cuda:0"
# Seed for random number generators to ensure reproducibility.
SEED=42

# --- GPU Allocation (for HPC schedulers like Slurm/PBS) ---
# This line tells the script which physical GPU to make visible to PyTorch.
# If you are allocated GPU #3, you would set this to 3. 
# For a single GPU job, it's often 0 or 1. Check your job's allocation.
# Let's assume you've been allocated a single GPU, which the system labels as '0'.
export CUDA_VISIBLE_DEVICES=0


################################################################################
### 2. SETTING UP THE ENVIRONMENT
################################################################################

echo "================================================="
echo "Setting up the environment..."
echo "Work Directory: $WORK_DIR"
echo "Conda Environment: $ENV_DIR"
echo "================================================="

# Load the Conda module if required by your HPC system.
# This might need to be adjusted based on your system's module manager.
module use /soft/modulefiles
module load conda

# Navigate to the working directory.
cd "$WORK_DIR" || { echo "Error: Could not navigate to WORK_DIR. Exiting."; exit 1; }

# Activate the Conda environment.
source activate "$ENV_DIR" || { echo "Error: Could not activate Conda environment. Exiting."; exit 1; }


################################################################################
### 3. RUNNING THE TRAINING SCRIPT
################################################################################

# Create a directory for log files if it doesn't exist.
mkdir -p logs

echo "================================================="
echo "Starting the training run..."
echo "Project: $WANDB_PROJECT"
echo "Run Name: $WANDB_NAME"
echo "Log file: logs/run_${WANDB_NAME}.log"
echo "Error file: logs/run_${WANDB_NAME}.err"
echo "================================================="

# Execute the Python training script with all the configured arguments.
# The `python -m` flag helps ensure Python uses the correct paths.
# We redirect standard output (stdout) to a .log file and standard error (stderr) to a .err file.
python -m protlig_dd.training.run_train_protlig_special \
    -WD "$WORK_DIR" \
    -cf "$CONFIG_FILE" \
    -wp "$WANDB_PROJECT" \
    -wn "$WANDB_NAME" \
    -df "$DATA_FILE" \
    -me "$MOL_EMB_ID" \
    -pe "$PROT_EMB_ID" \
    -di "$DEVICE_ID" \
    --seed "$SEED" \
    > "logs/run_${WANDB_NAME}.log" \
    2> "logs/run_${WANDB_NAME}.err"

# Optional: Add a small sleep to ensure all logs are written before the script exits.
sleep 5

echo "================================================="
echo "Training script has finished."
echo "Check logs/run_${WANDB_NAME}.log for output and logs/run_${WANDB_NAME}.err for errors."
echo "================================================="

# Deactivate the environment (good practice).
conda deactivate