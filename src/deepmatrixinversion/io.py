#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""io.py

This file is part of DeepMatrixInversion.
Copyright (C) 2019 Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
PAutoDock is distributed under GPLv3 license.
To know more in detail how the license work,
please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/gpl-3.0.en.html"

"""
import csv
import numpy as np
from sklearn.model_selection import train_test_split

def read_dataset(fmx):
    """
    Read matrix txt format
    """
    mxs = []
    mx = []
    with open(fmx, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "END" in line:
                mxs.append(mx.copy())
                del mx[:]
            else:
                mx.append(str.split(line, ","))
    return np.array(mxs).astype(float)


def write_dataset(X: np.array, out: str):
    """
    Write a dataset
    """
    with open(out, 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for matrix in X:
            for row in matrix:
                writer.writerow(row)
            f.write('END\n')

def write_train_test_validation_sets(
        X: np.array,
        Y: np.array,
        matrix_size: str,
        suffix: str
    ):
    """
    Take a dataset of matrix and write in this format
    X: list of matrix to invert
    Y: list o matrix inverted
    
    """
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25, random_state=42)
    if suffix:
        name = f'{suffix}_train'
    else:
        name = 'train'

    write_dataset(X_train, f'{name}_matrix_{matrix_size}x{matrix_size}.mx')
    write_dataset(Y_train, f'{name}_target_inverse_{matrix_size}x{matrix_size}.mx')

    if suffix:
        name = f'{suffix}_test'
    else:
        name = 'test'

    write_dataset(X_test, f'{name}_matrix_{matrix_size}x{matrix_size}.mx')
    write_dataset(Y_test, f'{name}_target_inverse_{matrix_size}x{matrix_size}.mx')

    if suffix:
        name = f'{suffix}_val'
    else:
        name = 'val'
    
    write_dataset(X_val, f'{name}_matrix_{matrix_size}x{matrix_size}.mx')
    write_dataset(Y_val, f'{name}_target_inverse_{matrix_size}x{matrix_size}.mx')