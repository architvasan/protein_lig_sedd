#!/bin/bash

# Script to run DDP checkpoint tests with proper environment setup

echo "🧪 Running DDP Checkpoint Tests"
echo "================================"

# Set up environment variables for DDP
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
export WORLD_SIZE="1"  # Single process for testing
export RANK="0"

# Set up Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# Disable NCCL for CPU testing
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo

echo "🔧 Environment setup:"
echo "   MASTER_ADDR: $MASTER_ADDR"
echo "   MASTER_PORT: $MASTER_PORT" 
echo "   WORLD_SIZE: $WORLD_SIZE"
echo "   RANK: $RANK"
echo "   PYTHONPATH: $PYTHONPATH"

echo ""
echo "📋 Running DDP checkpoint test..."

# Run the test
python test_checkpoint_ddp.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ DDP checkpoint test completed successfully!"
else
    echo ""
    echo "❌ DDP checkpoint test failed with exit code: $exit_code"
fi

exit $exit_code
