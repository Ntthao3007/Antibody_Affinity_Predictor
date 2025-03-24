#!/bin/bash

ENV_NAME="protein"

echo "Creating project environment..."
conda env create -f environment.yml -n $ENV_NAME || echo "Environment already exists."

echo "Activating project environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Starting contrastive model training..."
python scripts/train_contrastive.py
echo "Contrastive training complete!"

echo "Starting predictor model training..."
python scripts/train_predictor.py
echo "Predictor training complete!"

echo "All training steps are completed completed!"