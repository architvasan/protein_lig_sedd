#!/bin/bash

# 🚀 POLARIS DDP FRESH TRAINING SCRIPT
#
# This script supports both interactive and queue-based distributed training on Polaris.
# All configuration parameters are hardcoded at the top for easy modification.
# NO INTERACTIVE PROMPTS - suitable for queue system submission.
#
# 🎯 EXECUTION MODES:
# - INTERACTIVE: Run training directly (for testing/debugging)
# - QUEUE: Submit PBS job to Polaris scheduler (for production)
#
# 📝 TO CUSTOMIZE YOUR TRAINING:
# 1. Edit the parameters in the CONFIGURATION section below
# 2. Set EXECUTION_MODE to "interactive" or "queue"
# 3. Run: ./shell_scripts/start_fresh_train_ddp_polaris.sh
#
# 🔧 Key parameters to modify:
# - EXECUTION_MODE: "interactive" or "queue"
# - NUM_GPUS: Number of CUDA devices (1-4 per node)
# - WALLTIME: Job duration (format: HH:MM:SS)
# - CONFIG_FILE: Training configuration file
# - WANDB_PROJECT: Experiment tracking project name
# - CHECKPOINT_ACTION: How to handle existing checkpoints (remove/backup/ignore/cancel)

#############################################
# 🔧 CONFIGURATION - MODIFY THESE AS NEEDED
#############################################

# Training Configuration
WORK_DIR="./experiments/polaris_ddp_10nodes$(date +%Y%m%d_%H%M%S)"
CONFIG_FILE="configs/config_uniref50_ddp_16nodes.yaml"
DATAFILE="./input_data/processed_uniref50.pt"
WANDB_PROJECT="uniref50_polaris_ddp"
DEVICE="cuda"  # Polaris uses NVIDIA CUDA
SEED=42

# DDP Configuration
NUM_GPUS=40                    # Number of CUDA devices to use
AFFINITY_FILE="shell_scripts/set_affinity_gpu_polaris.sh" #file to set gpu aff
MASTER_PORT=29500            # Port for DDP communication
BACKEND="nccl"                # Use NCCL backend for NVIDIA GPUs

# Job Configuration
JOB_NAME="polaris_sedd_ddp"
QUEUE="preemptable"                # Polaris queue name
WALLTIME="72:00:00"          # 2 hours
NODES=10                      # Number of nodes
PPN=4                        # Processes per node (should match NUM_GPUS)

# Environment
VENV_PATH="../../protein_lig_sedd/prot_lig_sedd"        # Virtual environment path
LOG_DIR="logs"               # Log directory

# Execution Mode
EXECUTION_MODE="queue"       # Options: "interactive", "queue"
                            # interactive: Run training directly (for testing/debugging)
                            # queue: Submit PBS job to Polaris queue system

# Checkpoint Handling (no interactive prompts for queue systems)
CHECKPOINT_ACTION="remove"   # Options: "remove", "backup", "ignore", "cancel"
                            # remove: Delete existing checkpoints
                            # backup: Move checkpoints to timestamped backup
                            # ignore: Use --fresh flag to ignore checkpoints
                            # cancel: Exit if checkpoints exist

#############################################
# 🚀 SCRIPT EXECUTION - DO NOT MODIFY BELOW
#############################################



echo "🚀 Polaris DDP FRESH TRAINING"
echo "============================"
echo ""

echo "📋 Configuration:"
echo "   Work Dir: $WORK_DIR"
echo "   Config: $CONFIG_FILE"
echo "   Data: $DATAFILE"
echo "   Wandb Project: $WANDB_PROJECT"
echo "   Device: $DEVICE"
echo "   Seed: $SEED"
echo "   GPUs: $NUM_GPUS"
echo "   Backend: $BACKEND"
echo "   Job Name: $JOB_NAME"
echo "   Walltime: $WALLTIME"
echo ""

echo "loading modules"
module use /soft/modulefiles
module load conda
source $VENV_PATH/bin/activate

# Create work directory
mkdir -p "$WORK_DIR"
mkdir -p "$LOG_DIR"

# Generate unique run name
RUN_NAME="polaris_ddp_$(date +%Y%m%d_%H%M%S)"

# Check if checkpoints exist in work directory
if [ -d "$WORK_DIR/checkpoints" ] && [ "$(ls -A $WORK_DIR/checkpoints)" ]; then
    echo "📂 Existing checkpoints found in $WORK_DIR:"
    ls -la "$WORK_DIR/checkpoints/"
    echo ""
    echo "🔧 Checkpoint action set to: $CHECKPOINT_ACTION"

    case $CHECKPOINT_ACTION in
        "remove")
            echo "🗑️  Removing existing checkpoints..."
            rm -rf "$WORK_DIR/checkpoints/"
            echo "✅ Checkpoints removed"
            ;;
        "backup")
            backup_dir="$WORK_DIR/checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
            echo "📦 Backing up checkpoints to $backup_dir..."
            mv "$WORK_DIR/checkpoints/" "$backup_dir"
            echo "✅ Checkpoints backed up to $backup_dir"
            ;;
        "ignore")
            echo "🚀 Using --fresh flag to ignore existing checkpoints"
            FRESH_FLAG="--fresh"
            ;;
        "cancel")
            echo "❌ Cancelled due to existing checkpoints"
            echo "💡 Change CHECKPOINT_ACTION to 'remove', 'backup', or 'ignore' to proceed"
            exit 0
            ;;
        *)
            echo "❌ Invalid CHECKPOINT_ACTION: $CHECKPOINT_ACTION"
            echo "💡 Valid options: remove, backup, ignore, cancel"
            exit 1
            ;;
    esac
else
    echo "✅ No existing checkpoints found - ready for fresh training"
fi

echo ""
echo "🚀 Starting Polaris DDP training..."
echo "=================================="

echo "📋 Final Training Configuration:"
echo "   Execution Mode: $EXECUTION_MODE"
echo "   Work Dir: $WORK_DIR"
echo "   Config: $CONFIG_FILE"
echo "   Data: $DATAFILE"
echo "   Wandb Project: $WANDB_PROJECT"
echo "   Run Name: $RUN_NAME"
echo "   Device: $DEVICE"
echo "   Seed: $SEED"
echo "   GPUs: $NUM_GPUS"
echo "   Checkpoint Action: $CHECKPOINT_ACTION"
echo "   Fresh Start: ${FRESH_FLAG:-No (checkpoints handled)}"
echo ""

if [ "$EXECUTION_MODE" = "interactive" ]; then
    echo "🚀 Running training interactively..."
else
    echo "🚀 Proceeding with Polaris DDP job submission..."
fi

echo ""

if [ "$EXECUTION_MODE" = "interactive" ]; then
    echo "🎬 RUNNING TRAINING INTERACTIVELY..."
    echo "=================================="

    # Set environment variables
    export PYTHONPATH="$VENV_PATH/bin/python"#"$VENV_PATH:$PYTHONPATH"
    #export MASTER_PORT=29500

    # Run DDP training directly
    # Set MASTER_ADDR for torchrun
    #export MASTER_ADDR=localhost
    #export MASTER_ADDR=$(getent hosts $(head -1 $PBS_NODEFILE) | awk '{print $1}')
    #export MASTER_PORT=$MASTER_PORT
    # In your job script, before running Python
    #export MASTER_ADDR=$(head -1 $PBS_NODEFILE)
    #export MASTER_PORT=29500
    #export WORLD_SIZE=$(wc -l < $PBS_NODEFILE)
    #export RANK=$PBS_O_WORKDIR

    #export NCCL_SOCKET_FAMILY=AF_INET
    #export NCCL_NET_GDR_LEVEL=PHB
    #export NCCL_CROSS_NIC=1
    #export NCCL_COLLNET_ENABLE=1
    #export NCCL_NET="AWS Libfabric"
    #export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
    #export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
    #export FI_CXI_DISABLE_HOST_REGISTER=1
    #export FI_MR_CACHE_MONITOR=userfaultfd
    #export FI_CXI_DEFAULT_CQ_SIZE=131072

    # Run DDP training with torchrun
    #$VENV_PATH/bin/python -m torch.distributed.run \
        #--nnodes=$NODES \
        #--nproc_per_node=$PPN \
        #--rdzv_id=100 \
        #--rdzv_backend=c10d \
        #--rdzv_endpoint=localhost:$MASTER_PORT \
        #unset NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_COLLNET_ENABLE NCCL_NET

        mpirun -np 8 -ppn 4 \
        --hostfile $PBS_NODEFILE \
        --cpu-bind depth -d 16 \
        ./shell_scripts/set_affinity_gpu_polaris.sh \
        $VENV_PATH/bin/python protlig_dd/training/run_train_uniref_ddp_polaris.py \
        --work_dir "$WORK_DIR" \
        --config "$CONFIG_FILE" \
        --datafile "$DATAFILE" \
        --tokenize_on_fly \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "$RUN_NAME" \
        --device "$DEVICE" \
        --seed $SEED \
        $FRESH_FLAG 2>&1 | tee "$LOG_DIR/polaris_ddp_interactive.log"

    # Check exit status
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 INTERACTIVE TRAINING COMPLETED SUCCESSFULLY!"
        echo "=============================================="
        echo "📊 Check your Wandb dashboard for results"
        echo "📁 Checkpoints saved in: $WORK_DIR/checkpoints/"
        echo "📝 Log file: $LOG_DIR/polaris_ddp_interactive.log"
    else
        echo ""
        echo "❌ INTERACTIVE TRAINING FAILED"
        echo "============================="
        echo "💡 Check the log file: $LOG_DIR/polaris_ddp_interactive.log"
    fi

else
    echo "📝 Creating Polaris job script..."

    # Create Polaris job script
    JOB_SCRIPT="$WORK_DIR/polaris_ddp_job.sh"
    cat > "$JOB_SCRIPT" << EOF

#!/bin/bash
#PBS -l select=${NODES}:system=polaris
#PBS -l place=scatter
#PBS -l walltime=${WALLTIME}
#PBS -q ${QUEUE}
#PBS -A FoundEpidem
#PBS -l filesystems=eagle
#PBS -N ${JOB_NAME}
#PBS -o ${LOG_DIR}/polaris_ddp_\${PBS_JOBID}.out
#PBS -e ${LOG_DIR}/polaris_ddp_\${PBS_JOBID}.err

# Load modules and activate environment
module use /soft/modulefiles
module load conda
source ${VENV_PATH}/bin/activate

# Set environment variables
export PYTHONPATH="\$PWD:\$PYTHONPATH"
#export MASTER_PORT=${MASTER_PORT}

# Change to work directory
cd \$PBS_O_WORKDIR

# Run DDP training with torchrun
#export MASTER_ADDR=localhost
#export MASTER_PORT=${MASTER_PORT}

mpirun -np $GPUs -ppn $PPN \
--hostfile $PBS_NODEFILE \
--cpu-bind depth -d 16 \
./shell_scripts/set_affinity_gpu_polaris.sh \
$VENV_PATH/bin/python protlig_dd/training/run_train_uniref_ddp_polaris.py \
--work_dir "$WORK_DIR" \
--config "$CONFIG_FILE" \
--datafile "$DATAFILE" \
--tokenize_on_fly \
--wandb_project "$WANDB_PROJECT" \
--wandb_name "$RUN_NAME" \
--device "$DEVICE" \
--seed $SEED \
$FRESH_FLAG 2>&1 | tee "$LOG_DIR/polaris_ddp_16nodes.log"

EOF

echo "✅ Job script created: $JOB_SCRIPT"
echo ""

echo "🎬 SUBMITTING POLARIS JOB..."
echo "=========================="

# Submit the job
JOB_ID=$(qsub "$JOB_SCRIPT")
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 JOB SUBMITTED SUCCESSFULLY!"
    echo "============================="
    echo "📋 Job ID: $JOB_ID"
    echo "📁 Work Dir: $WORK_DIR"
    echo "📊 Logs: $LOG_DIR/polaris_ddp_${JOB_ID}.{out,err}"
    echo ""
    echo "🔍 Monitor job status:"
    echo "   qstat $JOB_ID"
    echo ""
    echo "📊 Check Wandb dashboard for training progress:"
    echo "   Project: $WANDB_PROJECT"
    echo "   Run: $RUN_NAME"
else
    echo ""
    echo "❌ JOB SUBMISSION FAILED"
    echo "======================="
    echo "💡 Check the error messages above"
    echo "📝 Job script saved at: $JOB_SCRIPT"
fi

fi  # End of EXECUTION_MODE if-else block

echo ""
echo "=== Polaris DDP script finished ==="
