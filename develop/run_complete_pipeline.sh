#!/bin/bash

# Complete Neural Forecasting Pipeline
# This script runs the full workflow from training to submission

set -e  # Exit on error

echo "======================================================================"
echo "Neural Forecasting - Complete Pipeline"
echo "======================================================================"

# Configuration
TRAIN_BOTH=true  # Set to false to skip training
VISUALIZE=true
CREATE_SUBMISSION=true

# Step 1: Train models
if [ "$TRAIN_BOTH" = true ]; then
    echo ""
    echo "Step 1: Training models..."
    echo "----------------------------------------------------------------------"

    echo "Training Beignet model..."
    python develop/train.py --monkey beignet

    echo ""
    echo "Training Affi model..."
    python develop/train.py --monkey affi

else
    echo ""
    echo "Step 1: Skipping training (using existing checkpoints)"
    echo "----------------------------------------------------------------------"
fi

# Step 2: Monitor final results
echo ""
echo "Step 2: Monitoring training results..."
echo "----------------------------------------------------------------------"
python develop/monitor_training.py beignet
echo ""
python develop/monitor_training.py affi

# Step 3: Visualize results
if [ "$VISUALIZE" = true ]; then
    echo ""
    echo "Step 3: Generating visualizations..."
    echo "----------------------------------------------------------------------"
    python develop/visualize_results.py
else
    echo ""
    echo "Step 3: Skipping visualization"
    echo "----------------------------------------------------------------------"
fi

# Step 4: Local evaluation
echo ""
echo "Step 4: Running local evaluation..."
echo "----------------------------------------------------------------------"
python develop/evaluate.py

# Step 5: Test submission format
echo ""
echo "Step 5: Testing submission format..."
echo "----------------------------------------------------------------------"
python develop/test_submission_format.py

# Step 6: Create submission package
if [ "$CREATE_SUBMISSION" = true ]; then
    echo ""
    echo "Step 6: Creating Codabench submission package..."
    echo "----------------------------------------------------------------------"
    python develop/create_submission.py

    echo ""
    echo "Creating submission zip file..."
    cd develop/submission
    zip -r submission.zip .
    cd ../..
    echo "✓ Submission package created: develop/submission/submission.zip"

else
    echo ""
    echo "Step 6: Skipping submission creation"
    echo "----------------------------------------------------------------------"
fi

# Summary
echo ""
echo "======================================================================"
echo "Pipeline Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Review visualizations in develop/logs/"
echo "2. Check submission package in develop/submission/"
echo "3. Upload develop/submission/submission.zip to Codabench"
echo ""
echo "Good luck!"
