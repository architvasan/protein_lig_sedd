conda deactivate
source /lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/prot_lig_sedd/bin/activate

python generate.py \
  --model_path "/lus/eagle/projects/FoundEpidem/avasan/IDEAL/DiffusionModels/protein_lig_sedd/checkpoints/checkpoint_9.pth" \
  --protein_sequence "MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFKHLKSEDEMKASEDLKKHGNTVLTALGGILKKKGHHEAEVKHLAESHANKHKAHVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR" \
  --num_samples 5 \
  --steps 1024