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
from deepmatrixinversion.io import write_dataset


def main():
    if len(sys.argv) != 6:
        print(
            f"Usage {sys.argv[0]} [matrix size] [number of samples] [range min] [range max] [outname_prefix]"
        )
        exit(0)
    matrix_size = int(sys.argv[1])
    num_samples = int(sys.argv[2])
    min_val = float(sys.argv[3])
    max_val = float(sys.argv[4])
    out_prefix = sys.argv[5]

    # Generate dataset
    X, Y = generate_matrix_inversion_dataset(
        num_samples,
        matrix_size,
        min_val,
        max_val,
    )
    # verify_matrix_inversion(X, Y)
    
    write_dataset(X, f'{out_prefix}_matrices_{matrix_size}x{matrix_size}.mx')
    write_dataset(Y, f'{out_prefix}_matrices_inverted_{matrix_size}x{matrix_size}.mx')

    X_Singular, Y_Singular = generate_singular_matrix_inversion_dataset(
        num_samples,
        matrix_size,
        matrix_size - 1,
        min_val,
        max_val,
    )
    # verify_pseudo_inversion(X_Singular, Y_Singular)
    write_dataset(X_Singular, f'{out_prefix}_singular_matrices_{matrix_size}x{matrix_size}.mx')
    write_dataset(Y_Singular, f'{out_prefix}_singular_matrices_pseudoinverted_{matrix_size}x{matrix_size}.mx')

if __name__ in "__main__":
    main()
