#!/usr/bin/env bash

./train.py --inputmx ../Dataset/data1/train_matrix_3x3.mx --inversemx ../Dataset/data1/train_target_inverse_3x3.mx --epochs 6000 --batch_size 2000 --nunits 128  --n_splits 5 --n_repeats 1
./predict.py --inputmx ../Dataset/data1/val_matrix_3x3.mx --expetarget ../Dataset/data1/val_target_inverse_3x3.mx --inverseout val_prediction.csv --model Model_*
./predict.py --inputmx ../Dataset/data1/sing_matrix_3x3.mx --exptarget ../Dataset/data1/sing_moore-penrose_inverse_3x3.mx --inverseout sing_prediction.csv --model Model_*
./predict.py --inputmx ../Dataset/data2/val_matrix_3x3.mx --exptarget ../Dataset/data2/val_target_inverse_3x3.mx --inverseout val_prediction_extrapolation.csv --model Model_*
./predict.py --inputmx ../Dataset/data3/val_matrix_3x3.mx --exptarget ../Dataset/data3/val_target_inverse_3x3.mx --inverseout val_prediction_interpolation.csv --model Model_*


