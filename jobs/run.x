#!/usr/bin/env bash

dmxtrain --msize 3 --rmin -1 --rmax 1 --epochs 5000 --batch_size 1024 --n_repeats 3 --mout ./Model_3x3
dmxinvert --inputmx ../data/data1/val_matrix_3x3.mx --exptarget ../data/data2/val_target_inverse_3x3.mx --inverseout val_prediction.csv --model Model_*
dmxinvert --inputmx ../data/data1/singular_val_matrix_3x3.mx --inverseout singular_val_target_inverse_3x3.csv --model Model_*
dmxinvert --inputmx ../data/data2/val_matrix_3x3.mx --exptarget ../data/data2/val_target_inverse_3x3.mx --inverseout val_prediction_extrapolation.csv --model Model_*
dmxinvert --inputmx ../data/data3/val_matrix_3x3.mx --exptarget ../data/data3/val_target_inverse_3x3.mx --inverseout val_prediction_interpolation.csv --model Model_*
