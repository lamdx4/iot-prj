#!/bin/bash

echo "════════════════════════════════════════════════════════════════════════════════"
echo "         🚀 Full Dataset Training Pipeline - Two-Stage Hierarchical 🚀"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Pipeline Steps:"
echo "  1. Merge files (74 files → 8 batches)"
echo "  2. Analyze batches → JSON statistics"
echo "  3. Train two-stage models"
echo "  4. Test models"
echo ""
echo "Total time: ~30-45 phút"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

cd "/home/lamdx4/Projects/IOT prj/src/dataset_full/scripts"

# Activate venv if exists
if [ -d "../../venv" ]; then
    source ../../venv/bin/activate
fi

# Step 1: Merge files
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "STEP 1: MERGING FILES (10 files → 1 batch)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

python 01_merge_files.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Step 1 failed! Stopping pipeline."
    exit 1
fi

# Step 2: Analyze batches
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "STEP 2: ANALYZING BATCHES → JSON"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

python 02_analyze_batches.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Step 2 failed! Stopping pipeline."
    exit 1
fi

# Step 3: Train models
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "STEP 3: TRAINING TWO-STAGE MODELS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

python 03_train_hierarchical.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Step 3 failed! Stopping pipeline."
    exit 1
fi

# Step 4: Test models
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "STEP 4: TESTING MODELS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

python 04_test_model.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Step 4 failed!"
    exit 1
fi

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ PIPELINE COMPLETED SUCCESSFULLY!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📂 Check results:"
echo "   • Batches: Data/Dataset/merged_batches/"
echo "   • Stats:   src/dataset_full/stats/"
echo "   • Models:  models/full_dataset/"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"


