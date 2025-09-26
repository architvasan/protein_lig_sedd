#!/bin/bash
#PBS -l select=1:ncpus=32:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -A FoundEpidem
#PBS -N smiles_diffusion_train

# Load modules
module load conda
conda activate protlig_dd

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd:$PYTHONPATH"

# Training parameters
WORK_DIR="/lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd/experiments/smiles_$(date +%Y%m%d_%H%M%S)"
#CONFIG_FILE="/lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd/configs/config.yaml"
CONFIG_FILE="/lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd/configs/config_pubchem_smiles.yaml"
DATA_FILE="/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/input_data/processed_pubchem.pt"
#DATA_FILE="/lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd/input_data/processed_pubchem_subset_1k.pt"

# Run training
cd /lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd

python protlig_dd/training/run_train_smiles.py \
    --work_dir $WORK_DIR \
    --config $CONFIG_FILE \
    --datafile $DATA_FILE \
    --device cuda:0 \
    --project "smiles-diffusion" \
    --name "pubchem-smiles-$(date +%Y%m%d_%H%M%S)" \
    --seed 42

echo "Training completed. Results saved to: $WORK_DIR"