#!/bin/bash

# 🚀 AURORA DDP FRESH TRAINING SCRIPT
#
# This script supports both interactive and queue-based distributed training on Aurora.
# All configuration parameters are hardcoded at the top for easy modification.
# NO INTERACTIVE PROMPTS - suitable for queue system submission.
#
# 🎯 EXECUTION MODES:
# - INTERACTIVE: Run training directly (for testing/debugging)
# - QUEUE: Submit PBS job to Aurora scheduler (for production)
#
# 📝 TO CUSTOMIZE YOUR TRAINING:
# 1. Edit the parameters in the CONFIGURATION section below
# 2. Set EXECUTION_MODE to "interactive" or "queue"
# 3. Run: ./shell_scripts/start_fresh_train_ddp_aurora.sh
#
# 🔧 Key parameters to modify:
# - EXECUTION_MODE: "interactive" or "queue"
# - NUM_GPUS: Number of XPU devices (1-4 per node)
# - WALLTIME: Job duration (format: HH:MM:SS)
# - CONFIG_FILE: Training configuration file
# - WANDB_PROJECT: Experiment tracking project name
# - CHECKPOINT_ACTION: How to handle existing checkpoints (remove/backup/ignore/cancel)

#############################################
# 🔧 CONFIGURATION - MODIFY THESE AS NEEDED
#############################################

# Training Configuration
WORK_DIR="./experiments/polaris_ddp_$(date +%Y%m%d_%H%M%S)"
CONFIG_FILE="configs/config_uniref50_ddp.yaml"
DATAFILE="./input_data/subset_uniref50.pt"
WANDB_PROJECT="uniref50_polaris_ddp"
DEVICE="cuda"  # Aurora uses Intel XPU
SEED=42

# DDP Configuration
NUM_GPUS=4                    # Number of XPU devices to use
AFFINITY_FILE="shell_scripts/set_affinity_gpu_polaris.sh" #file to set gpu aff
MASTER_PORT=29500            # Port for DDP communication
BACKEND="nccl"                # Use CCL backend for Intel XPU

# Job Configuration
JOB_NAME="polaris_sedd_ddp"
QUEUE="debug"                # Aurora queue name
WALLTIME="01:00:00"          # 2 hours
NODES=1                      # Number of nodes
PPN=4                        # Processes per node (should match NUM_GPUS)

# Environment
VENV_PATH="../../protein_lig_sedd/prot_lig_sedd"        # Virtual environment path
LOG_DIR="logs"               # Log directory

# Execution Mode
EXECUTION_MODE="interactive"       # Options: "interactive", "queue"
                            # interactive: Run training directly (for testing/debugging)
                            # queue: Submit PBS job to Aurora queue system

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
RUN_NAME="aurora_ddp_$(date +%Y%m%d_%H%M%S)"

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
    echo "🚀 Proceeding with Aurora DDP job submission..."
fi

echo ""

if [ "$EXECUTION_MODE" = "interactive" ]; then
    echo "🎬 RUNNING TRAINING INTERACTIVELY..."
    echo "=================================="

    # Set environment variables
    export PYTHONPATH="$PWD:$PYTHONPATH"
    export MASTER_PORT=$MASTER_PORT

    # Run DDP training directly
    #mpiexec -n $NUM_GPUS -ppn $PPN \
    torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 \ 
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
        protlig_dd/training/run_train_uniref_ddp_polaris.py \
        --work_dir "$WORK_DIR" \
        --config "$CONFIG_FILE" \
        --datafile "$DATAFILE" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_name "$RUN_NAME" \
        --device "$DEVICE" \
        --seed $SEED \
        $FRESH_FLAG 2>&1 | tee "$LOG_DIR/aurora_ddp_interactive.log"

    # Check exit status
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 INTERACTIVE TRAINING COMPLETED SUCCESSFULLY!"
        echo "=============================================="
        echo "📊 Check your Wandb dashboard for results"
        echo "📁 Checkpoints saved in: $WORK_DIR/checkpoints/"
        echo "📝 Log file: $LOG_DIR/aurora_ddp_interactive.log"
    else
        echo ""
        echo "❌ INTERACTIVE TRAINING FAILED"
        echo "============================="
        echo "💡 Check the log file: $LOG_DIR/aurora_ddp_interactive.log"
    fi

else
    echo "📝 Creating Aurora job script..."

    # Create Aurora job script
    JOB_SCRIPT="$WORK_DIR/aurora_ddp_job.sh"
    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#PBS -l select=${NODES}:system=aurora_compute
#PBS -l place=scatter
#PBS -l walltime=${WALLTIME}
#PBS -q ${QUEUE}
#PBS -A Aurora_deployment
#PBS -N ${JOB_NAME}
#PBS -o ${LOG_DIR}/aurora_ddp_\${PBS_JOBID}.out
#PBS -e ${LOG_DIR}/aurora_ddp_\${PBS_JOBID}.err

# Load modules and activate environment
module load frameworks
source ${VENV_PATH}/bin/activate

# Set environment variables
export PYTHONPATH="\$PWD:\$PYTHONPATH"
export MASTER_PORT=${MASTER_PORT}

# Change to work directory
cd \$PBS_O_WORKDIR

# Run DDP training
mpiexec -n ${NUM_GPUS} -ppn ${PPN} python -m protlig_dd.training.run_train_uniref_ddp_aurora \\
    --work_dir "${WORK_DIR}" \\
    --config "${CONFIG_FILE}" \\
    --datafile "${DATAFILE}" \\
    --wandb_project "${WANDB_PROJECT}" \\
    --wandb_name "${RUN_NAME}" \\
    --device "${DEVICE}" \\
    --seed ${SEED} \\
    ${FRESH_FLAG} > ${LOG_DIR}/aurora_ddp_\${PBS_JOBID}.log 2> ${LOG_DIR}/aurora_ddp_\${PBS_JOBID}.err
EOF

echo "✅ Job script created: $JOB_SCRIPT"
echo ""

echo "🎬 SUBMITTING AURORA JOB..."
echo "=========================="

# Submit the job
JOB_ID=$(qsub "$JOB_SCRIPT")
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 JOB SUBMITTED SUCCESSFULLY!"
    echo "============================="
    echo "📋 Job ID: $JOB_ID"
    echo "📁 Work Dir: $WORK_DIR"
    echo "📊 Logs: $LOG_DIR/aurora_ddp_${JOB_ID}.{out,err}"
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
echo "=== Aurora DDP script finished ==="
