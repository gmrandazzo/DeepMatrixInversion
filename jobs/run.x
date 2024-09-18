#!/usr/bin/env bash

dmxtrain --inputmx ../data/data1/train_matrix_3x3.mx --inversemx ../data/data1/train_target_inverse_3x3.mx --epochs 30000 --batch_size 2000 --nunits 128  --n_splits 5 --n_repeats 1
dmxinvert --inputmx ../data/data1/val_matrix_3x3.mx --exptarget ../data/data2/val_target_inverse_3x3.mx --inverseout val_prediction.csv --model Model_*
dmxinvert --inputmx ../data/data1/singular_val_matrix_3x3.mx --inverseout singular_val_target_inverse_3x3.csv --model Model_*
dmxinvert --inputmx ../data/data2/val_matrix_3x3.mx --exptarget ../data/data2/val_target_inverse_3x3.mx --inverseout val_prediction_extrapolation.csv --model Model_*
dmxinvert --inputmx ../data/data3/val_matrix_3x3.mx --exptarget ../data/data3/val_target_inverse_3x3.mx --inverseout val_prediction_interpolation.csv --model Model_*
