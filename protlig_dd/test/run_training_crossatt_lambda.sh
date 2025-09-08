#!/bin/bash

###########################
### DEFINING VARIABLES###
##########################
WORK_DIR=/nfs/ml_lab/projects/ml_lab/xlian/protein_lig_sedd
ENV_DIR=/homes/xlian/.conda/envs/protlig_dd
CONFIG_FILE=$WORK_DIR/configs/config.yaml
WANDBPROJ=protlig_sedd_crossatt
# how to add date/time to WANDBNAME?
WANDBNAME=plinder_crossatt_$(date +'%d-%m-%Y_%H-%M-%S')

#DATAFILE=$WORK_DIR/../input_data/filtered_missing_tokenized.pt
DATAFILE=$WORK_DIR/../input_data/smi_prot_test.pt
MOL_EMB_ID=ibm/MoLFormer-XL-both-10pct
PROT_EMB_ID=facebook/esm2_t30_150M_UR50D
DEV_ID=cuda:0
SEED=42

export WANDB_MODE=offline

#########################
### SETTING UP ENVIRONMENTS
########################

cd $WORK_DIR
mkdir -p logs
source "/nfs/ml_lab/projects/ml_lab/avasan/Diffusion/protein_lig_sedd/pl_dd_venv/bin/activate"
#source /software/lambda/Anaconda3/2022.10/etc/profile.d/conda.sh
#conda activate "$ENV_DIR"

#######################
#### RUNNING TRAINING WITH VARIABLES###
#####################
CUDA_VISIBLE_DEVICES=1 python -m protlig_dd.training.run_train_protlig_cross_att \
    -WD $WORK_DIR\
    -cf $CONFIG_FILE\
    -wp $WANDBPROJ\
    -wn $WANDBNAME\
    -df $DATAFILE\
    -me $MOL_EMB_ID\
    -pe $PROT_EMB_ID\
    -di $DEV_ID\
    --seed $SEED\
> logs/run_${WANDBPROJ}.log \
2> logs/run_${WANDBPROJ}.err