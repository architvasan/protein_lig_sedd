#!/bin/bash
#PBS -l select=1:ncpus=32:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -A FoundEpidem
#PBS -N smiles_diffusion_train

# Load modules
#module load conda
source ~/.bashrc
conda activate protlig_dd

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/nfs/ml_lab/projects/ml_lab/xlian/protein_lig_sedd:$PYTHONPATH"

# Training parameters
WORK_DIR="/nfs/ml_lab/projects/ml_lab/xlian/protein_lig_sedd/experiments/smiles_$(date +%Y%m%d_%H%M%S)"
#CONFIG_FILE="/nfs/ml_lab/projects/ml_lab/xlian/protein_lig_sedd/configs/config.yaml"
CONFIG_FILE="/nfs/ml_lab/projects/ml_lab/xlian/protein_lig_sedd/configs/config_pubchem_smiles.yaml"
DATA_FILE="/nfs/ml_lab/projects/ml_lab/avasan/Diffusion/protein_lig_sedd/input_data/processed_pubchem.pt"
#DATA_FILE="/nfs/ml_lab/projects/ml_lab/xlian/input_data/processed_pubchem_subset_1k.pt"

cd /nfs/ml_lab/projects/ml_lab/xlian/protein_lig_sedd

python protlig_dd/training/run_train_smiles.py \
    --work_dir $WORK_DIR \
    --config $CONFIG_FILE \
    --datafile $DATA_FILE \
    --device cuda:0 \
    --project "smiles-diffusion" \
    --name "pubchem-smiles-$(date +%Y%m%d_%H%M%S)" \
    --seed 42\
|| exit 1

echo "Training completed. Results saved to: $WORK_DIR"