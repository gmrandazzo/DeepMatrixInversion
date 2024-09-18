"""dataset.py

This file is part of DeepMatrixInversion.
Copyright (C) 2019 Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
PAutoDock is distributed under GPLv3 license.
To know more in detail how the license work,
please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/gpl-3.0.en.html"

"""

import numpy as np


def generate_singular_matrix(size, rank, min_val, max_val):
    """Generate a singular matrix of given size and
    rank within the specified value range.
    """
    U = np.random.uniform(min_val, max_val, (size, rank))
    V = np.random.uniform(min_val, max_val, (rank, size))
    return np.dot(U, V)


def generate_singular_matrix_inversion_dataset(
    num_samples, matrix_size, max_rank, min_val=-1, max_val=1
):
    """Generate a dataset of singular matrices and their pseudoinverses."""
    X = []
    Y = []
    for _ in range(num_samples):
        rank = np.random.randint(1, max_rank + 1)
        matrix = generate_singular_matrix(matrix_size, rank, min_val, max_val)
        pseudoinverse = np.linalg.pinv(matrix)
        X.append(matrix)
        Y.append(pseudoinverse)

    return np.array(X), np.array(Y)


def is_invertible(matrix):
    return np.linalg.cond(matrix) < 1 / np.finfo(matrix.dtype).eps


def generate_matrix_inversion_dataset(num_samples, matrix_size, min_val=-1, max_val=1):
    X = []
    Y = []
    while len(X) < num_samples:
        matrix = np.random.uniform(min_val, max_val, (matrix_size, matrix_size))
        if is_invertible(matrix):
            try:
                inverse = np.linalg.inv(matrix)
                X.append(matrix)
                Y.append(inverse)
            except np.linalg.LinAlgError:
                continue
    return np.array(X), np.array(Y)


def verify_matrix_inversion(X: np.array, Y: np.array) -> bool:
    for i, x in enumerate(X):
        matrix_size = x.shape[0]
        product = np.dot(x, Y[i])
        deviation_from_identity = np.max(np.abs(product - np.eye(matrix_size)))
        if deviation_from_identity > 1e-3:
            print(f"Deviation from identity: {deviation_from_identity}")
            print(f"Problem with matrix\n {x}")
            print(f"inverted matrix\n {Y[i]}")
            return False
        return True


def verify_pseudo_inversion(X: np.array, Y: np.array) -> bool:
    for i, x in enumerate(X):
        if np.allclose(Y[i], np.dot(Y[i], np.dot(x, Y[i]))) is False:
            print(f"Problem with matrix\n{x}")
            return False
        return True
