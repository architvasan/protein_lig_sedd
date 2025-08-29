#!/bin/bash

###########################
### DEFINING VARIABLES###
##########################
WORK_DIR=/nfs/ml_lab/projects/ml_lab/avasan/Diffusion/protein_lig_sedd
ENV_DIR=$WORK_DIR/pl_dd_venv

CONFIG_FILE=$WORK_DIR/configs/config_ddp.yaml
WANDBPROJ=protlig_sedd_plinder
WANDBNAME=run1_081925

DATAFILE=$WORK_DIR/input_data/merged_plinder.pt
MOL_EMB_ID=ibm/MoLFormer-XL-both-10pct
PROT_EMB_ID=facebook/esm2_t30_150M_UR50D
DEV_ID=cuda:0
SEED=42

NUM_GPUS=2
#########################
### SETTING UP ENVIRONMENTS
########################

#module use /soft/modulefiles
#module load conda

cd $WORK_DIR
source ${ENV_DIR}/bin/activate

#######################
#### RUNNING TRAINING WITH VARIABLES###
#####################

       CUDA_VISIBLE_DEVICES=2 python protlig_dd/training/run_train_protlig_special.py \
       -WD $WORK_DIR\
       -cf $CONFIG_FILE\
       -wp $WANDBPROJ\
       -wn $WANDBNAME\
       -df $DATAFILE\
       -me $MOL_EMB_ID\
       -pe $PROT_EMB_ID\
       -di $DEV_ID\
       --seed $SEED\
> logs/run_${WANDBPROJ}_${WANDBNAME}.log \
2> logs/run_${WANDBPROJ}_${WANDBNAME}.err
