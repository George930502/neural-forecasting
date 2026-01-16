#!/bin/bash

# Wait for Beignet training to complete, then train Affi

echo "======================================================================"
echo "Automated Training: Beignet → Affi"
echo "======================================================================"

# Wait for beignet training to complete
echo ""
echo "Waiting for Beignet training to complete..."
echo "(This script will check every 30 seconds)"
echo ""

while pgrep -f "python develop/train.py --monkey beignet" > /dev/null; do
    python develop/monitor_training.py beignet | grep "Current epoch"
    sleep 30
done

echo ""
echo "✓ Beignet training completed!"
echo ""

# Show final results
echo "Final Beignet results:"
python develop/monitor_training.py beignet

# Start Affi training
echo ""
echo "======================================================================"
echo "Starting Affi training..."
echo "======================================================================"
echo ""

python develop/train.py --monkey affi

echo ""
echo "======================================================================"
echo "Both models trained!"
echo "======================================================================"
echo ""

# Show results
echo "Final Results:"
python develop/monitor_training.py beignet
echo ""
python develop/monitor_training.py affi

echo ""
echo "Next step: Create submission package"
echo "Run: python develop/create_submission.py"
