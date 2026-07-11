#!/bin/bash
# test.sh – Execute the full prediction pipeline

set -e  # Exit immediately if any command fails

echo "============================================"
echo "  1/4  Fetching match data (main.py)"
echo "============================================"
python main.py

echo ""
echo "============================================"
echo "  2/4  Calculating Elo ratings (calculate_elo.py)"
echo "============================================"
python calculate_elo.py

echo ""
echo "============================================"
echo "  3/4  Engineering features (feature_engineering.py)"
echo "============================================"
python feature_engineering.py

echo ""
echo "============================================"
echo "  4/4  Training model (train_model.py)"
echo "============================================"
python train_model.py

echo ""
echo "============================================"
echo "  Pipeline completed successfully!"
echo "============================================"