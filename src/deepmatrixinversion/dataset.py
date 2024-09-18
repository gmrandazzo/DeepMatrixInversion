import sys
import numpy as np
import csv
from sklearn.model_selection import train_test_split


def generate_singular_matrix(size, rank, min_val, max_val):
    """Generate a singular matrix of given size and rank within the specified value range."""
    U = np.random.uniform(min_val, max_val, (size, rank))
    V = np.random.uniform(min_val, max_val, (rank, size))
    return np.dot(U, V)

def generate_singular_matrix_inversion_dataset(num_samples, matrix_size, max_rank, min_val=-1, max_val=1):
    """Generate a dataset of singular matrices and their pseudoinverses."""
    X = []  # Input matrices
    Y = []  # Pseudoinverse matrices (outputs)
    for _ in range(num_samples):
        # Generate a random rank for the matrix
        rank = np.random.randint(1, max_rank + 1)
        # Generate a singular matrix
        matrix = generate_singular_matrix(matrix_size, rank, min_val, max_val)
        # Calculate the pseudoinverse
        pseudoinverse = np.linalg.pinv(matrix)
        # Append to datasets
        X.append(matrix)
        Y.append(pseudoinverse)
    
    return np.array(X), np.array(Y)

def is_invertible(matrix):
    return np.linalg.cond(matrix) < 1 / np.finfo(matrix.dtype).eps

def generate_matrix_inversion_dataset(num_samples, matrix_size, min_val=-1, max_val=1):
    X = []  # Input matrices
    Y = []  # Inverse matrices (outputs)
    while len(X) < num_samples:
        # Generate a random square matrix
        matrix = np.random.uniform(min_val, max_val, (matrix_size, matrix_size))
        # Check if the matrix is invertible
        if is_invertible(matrix):
            try:
                # Calculate the inverse
                inverse = np.linalg.inv(matrix)
                # Append to datasets
                X.append(matrix)
                Y.append(inverse)
            except np.linalg.LinAlgError:
                # Skip this matrix if inversion fails
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
        # Verify the pseudoinverse properties
        # print("\nVerifying pseudoinverse properties:")
        # print("XX+X ≈ X:")
        # print(np.allclose(x, np.dot(x, np.dot(Y[i], x))))
        # print("X+XX+ ≈ X+:")
        if np.allclose(Y[i], np.dot(Y[i], np.dot(x, Y[i]))) == False:
            print(f'Problem with matrix\n{x}')
            return False
        # print(np.allclose(Y[i], np.dot(Y[i], np.dot(x, Y[i]))))
        # Calculate the condition number
        # cond_num = np.linalg.cond(x)
        # print(f"\nCondition number of X: {cond_num}")
        # For nearly singular matrices, we can use a higher tolerance
        # X_nearly_singular = x + np.random.normal(0, 1e-10, x.shape)
        # X_nearly_singular_pinv = np.linalg.pinv(X_nearly_singular, rcond=1e-8)
        # print("\nPseudoinverse of nearly singular matrix:")
        # print(X_nearly_singular_pinv)
        return True

