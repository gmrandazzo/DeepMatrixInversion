#!/usr/bin/env bash
# run.x: Automation script for training and evaluating DeepMatrixInversion
# 
# This script performs:
# 1. Training of an ensemble of ResNet models on 3x3 matrices.
# 2. Evaluation on several validation and test sets.
# 3. Prediction on singular matrices to demonstrate model behavior.

set -e # Exit immediately if a command exits with a non-zero status

# Configuration
MODEL_PREFIX="./Model_3x3"
MSIZE=3
EPOCHS=5000
N_REPEATS=3

echo "Starting training phase..."
# Training the ensemble
# Note: dmxtrain will create a folder like Model_3x3_YYYYMMDDHHMMSS
dmxtrain --msize $MSIZE --rmin -1 --rmax 1 --epochs $EPOCHS --batch_size 1024 --n_repeats $N_REPEATS --mout $MODEL_PREFIX

# Find the newly created model folder
MODEL_DIR=$(ls -td ${MODEL_PREFIX}_* | head -n 1)
echo "Using model directory: $MODEL_DIR"

echo "Evaluating on validation data..."
# Run inference on validation sets using the trained ensemble
# Corrected argument --invtarget for comparison and added plot output
dmxinvert --inputmx ../data/data1/val_matrix_3x3.mx --invtarget ../data/data1/val_target_inverse_3x3.mx --inverseout val_prediction.csv --model "$MODEL_DIR" --plotout val_validation_plot.png

echo "Inverting singular matrices (demo)..."
# Singular matrices demonstrate that the model will produce a result even without a mathematical inverse
dmxinvert --inputmx ../data/data1/sing_matrix_3x3.mx --inverseout singular_val_target_inverse_3x3.csv --model "$MODEL_DIR"

echo "Testing extrapolation/interpolation sets..."
dmxinvert --inputmx ../data/data2/val_matrix_3x3.mx --invtarget ../data/data2/val_target_inverse_3x3.mx --inverseout val_prediction_extrapolation.csv --model "$MODEL_DIR"
dmxinvert --inputmx ../data/data3/val_matrix_3x3.mx --invtarget ../data/data3/val_target_inverse_3x3.mx --inverseout val_prediction_interpolation.csv --model "$MODEL_DIR"

echo "All tasks completed successfully."
