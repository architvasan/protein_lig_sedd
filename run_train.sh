#!/bin/bash

###########################
### DEFINING VARIABLES###
##########################
WORK_DIR=/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd
ENV_DIR=$WORK_DIR/prot_lig_sedd
CONFIG_FILE=$WORK_DIR/configs/config.yaml
WANDBPROJ=protlig_sedd
WANDBNAME=run1
PLINDER_OUTPUT_DIR=$WORK_DIR/plinder_10k/processed_plinder_data
PLINDER_DATA_DIR=$WORK_DIR/plinder_10k/processed_plinder_data
MOL_EMB_ID=ibm/MoLFormer-XL-both-10pct
PROT_EMB_ID=facebook/esm2_t30_150M_UR50D
DEV_ID=cuda:0
SEED=42

#########################
### SETTING UP ENVIRONMENTS
########################

module use /soft/modulefiles
module load conda

cd $WORK_DIR
source ${ENV_DIR}/bin/activate

#######################
#### RUNNING TRAINING WITH VARIABLES###
#####################
python -m protlig_dd.training.run_train_protlig_condition \
    -WD $WORK_DIR\
    -cf $CONFIG_FILE\
    -wp $WANDBPROJ\
    -wn $WANDBNAME\
    -po $PLINDER_OUTPUT_DIR\
    -pd $PLINDER_DATA_DIR\
    -me $MOL_EMB_ID\
    -pe $PROT_EMB_ID\
    -di $DEV_ID\
    --seed $SEED\
> logs/run_${WANDBPROJ}_${WANDBNAME}.log \
2> logs/run_${WANDBPROJ}_${WANDBNAME}.err 



