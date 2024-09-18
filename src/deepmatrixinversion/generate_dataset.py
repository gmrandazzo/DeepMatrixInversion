#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_dataset.py

This file is part of DeepMatrixInversion.
Copyright (C) 2019 Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
PAutoDock is distributed under GPLv3 license.
To know more in detail how the license work,
please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/gpl-3.0.en.html"

"""
import sys

from deepmatrixinversion.dataset import (
    generate_matrix_inversion_dataset,
    generate_singular_matrix_inversion_dataset,
)
from deepmatrixinversion.io import write_train_test_validation_sets


def main():
    if len(sys.argv) != 5:
        print(
            f"Usage {sys.argv[0]} [matrix size] [number of samples] [range min] [range max]"
        )
        exit(0)
    matrix_size = int(sys.argv[1])
    num_samples = int(sys.argv[2])
    min_val = float(sys.argv[3])
    max_val = float(sys.argv[4])

    # Generate dataset
    num_samples = 10000
    matrix_size = 3
    X, Y = generate_matrix_inversion_dataset(
        num_samples,
        matrix_size,
        min_val,
        max_val,
    )

    # verify_matrix_inversion(X, Y)

    write_train_test_validation_sets(X, Y, matrix_size, "")

    X_Singular, Y_Singular = generate_singular_matrix_inversion_dataset(
        num_samples,
        matrix_size,
        matrix_size - 1,
        min_val,
        max_val,
    )
    # verify_pseudo_inversion(X_Singular, Y_Singular)
    write_train_test_validation_sets(X_Singular, Y_Singular, matrix_size, "singular")


if __name__ in "__main__":
    main()
