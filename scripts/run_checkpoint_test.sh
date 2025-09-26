#!/bin/bash

# Script to run checkpoint resume tests

echo "🧪 Running Checkpoint Resume Tests"
echo "=================================="

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the simple test
echo "📋 Running simple checkpoint test..."
python tests/test_checkpoint_simple.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Simple test passed!"
    
    # If pytest is available, run the full test suite
    if command -v pytest &> /dev/null; then
        echo ""
        echo "📋 Running full pytest suite..."
        pytest tests/test_checkpoint_resume.py -v
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "🎉 All checkpoint tests passed!"
        else
            echo ""
            echo "⚠️  Some pytest tests failed, but simple test passed"
        fi
    else
        echo ""
        echo "ℹ️  pytest not available, skipping full test suite"
        echo "   Install with: pip install pytest"
    fi
else
    echo ""
    echo "❌ Simple test failed!"
    exit 1
fi
