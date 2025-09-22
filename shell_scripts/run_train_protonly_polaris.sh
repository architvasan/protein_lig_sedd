#!/bin/bash

###########################
### DEFINING VARIABLES###
##########################
WORK_DIR=/eagle/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd
ENV_DIR=$WORK_DIR/prot_lig_sedd
CONFIG_FILE=$WORK_DIR/configs/config_prot.yaml
WANDBPROJ=protlig_sedd_crossatt_small_maxlen
# how to add date/time to WANDBNAME?
WANDBNAME=plinder_crossatt_protonly_uniref$(date +'%d-%m-%Y_%H-%M-%S')

DATAFILE=$WORK_DIR/input_data/processed_uniref50.pt
MOL_EMB_ID=ibm/MoLFormer-XL-both-10pct
PROT_EMB_ID=facebook/esm2_t30_150M_UR50D
DEV_ID=cuda:0
SEED=42

#########################
### SETTING UP ENVIRONMENTS
########################

cd $WORK_DIR
source ${ENV_DIR}/bin/activate

#######################
#### RUNNING TRAINING WITH VARIABLES###
#####################
CUDA_VISIBLE_DEVICES=4 python -m protlig_dd.training.run_train_protlig_cross_att_improved \
    -WD $WORK_DIR\
    -cf $CONFIG_FILE\
    -wp $WANDBPROJ\
    -wn $WANDBNAME\
    -df $DATAFILE\
    -me $MOL_EMB_ID\
    -pe $PROT_EMB_ID\
    -di $DEV_ID\
    --seed $SEED\
> logs/run_${WANDBPROJ}.prot_uniref.log \
2> logs/run_${WANDBPROJ}.prot_uniref.err
