#!/bin/bash

###########################
### DEFINING VARIABLES ###
##########################

# Main directories
WORK_DIR=/eagle/projects/FoundEpidem/xlian/protein_lig_sedd
ENV_DIR=/lus/eagle/projects/FoundEpidem/xlian/conda/envs/protlig_dd

# Config and data
CONFIG_FILE="$WORK_DIR/configs/config_uniref50_stable.yaml"
DATAFILE=$WORK_DIR/input_data/processed_uniref50.pt

# Wandb settings
WANDBPROJ=protlig_sedd_crossatt_small_maxlen
WANDBNAME="xltest_$(date +'%Y%m%d_%H%M%S')"  # Added date/time safely
export WANDB_MODE=offline  # set to 'online' if you want live logging

# Model & device parameters
MOL_EMB_ID=ibm/MoLFormer-XL-both-10pct
PROT_EMB_ID=facebook/esm2_t30_150M_UR50D
DEV_ID=cuda:0
SEED=42

#########################
### SETTING UP ENVIRONMENT
#########################

cd "$WORK_DIR"
#source /lus/eagle/projects/FoundEpidem/xlian/conda/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

#########################
### RUN TRAINING
#########################

python -m protlig_dd.training.run_train_uniref50_optimized \
    --work_dir "$WORK_DIR" \
    --config "$CONFIG_FILE" \
    --datafile "$DATAFILE" \
    --wandb_project "$WANDBPROJ" \
    --wandb_name "$WANDBNAME" \
    --device "$DEV_ID" \
    --seed "$SEED" \
#    --mol_emb "$MOL_EMB_ID" \
#    --prot_emb "$PROT_EMB_ID"
