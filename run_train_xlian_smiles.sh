#!/bin/bash
#PBS -l select=1:ncpus=32:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -A FoundEpidem
#PBS -N smiles_diffusion_train

# Load modules
module load conda
conda activate protlig_dd

#export PYTHONPATH="/lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd:$PYTHONPATH"

PROJECT_ROOT="/lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd"
WORK_DIR="$PROJECT_ROOT/experiments/smiles_$(date +%Y%m%d_%H%M%S)"
#CONFIG_FILE="/lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd/configs/config.yaml"
CONFIG_FILE="$PROJECT_ROOT/configs/config_pubchem_smiles.yaml"
DATA_FILE="/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/input_data/processed_pubchem.pt"
#DATA_FILE="/lus/eagle/projects/FoundEpidem/xlian/protein_lig_sedd/input_data/processed_pubchem_subset_1k.pt"

# Run training
cd $PROJECT_ROOT

python protlig_dd/training/run_train_smiles.py \
    --work_dir $WORK_DIR \
    --config $CONFIG_FILE \
    --datafile $DATA_FILE \
    --device cuda:0 \
    --project "smiles-diffusion-2" \
    --name "pubchem-smiles-$(date +%Y%m%d_%H%M%S)" \
    --seed 42

cp $CONFIG_FILE $WORK_DIR/config.yaml
echo "Training completed. Results saved to: $WORK_DIR"