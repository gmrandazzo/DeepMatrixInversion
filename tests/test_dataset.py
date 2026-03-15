import numpy as np
import pytest
from deepmatrixinversion.dataset import (
    generate_matrix_inversion_dataset,
    is_invertible,
    verify_matrix_inversion,
    generate_singular_matrix,
    generate_singular_matrix_inversion_dataset,
    verify_pseudo_inversion
)

def test_is_invertible():
    # Invertible matrix
    m1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert is_invertible(m1)
    
    # Singular matrix
    m2 = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert not is_invertible(m2)

def test_generate_matrix_inversion_dataset():
    num_samples = 10
    matrix_size = 3
    X, Y = generate_matrix_inversion_dataset(num_samples, matrix_size)
    
    assert X.shape == (num_samples, matrix_size, matrix_size)
    assert Y.shape == (num_samples, matrix_size, matrix_size)
    
    # Check if they are actually inverses
    assert verify_matrix_inversion(X, Y) is True

def test_generate_singular_matrix():
    size = 4
    rank = 2
    m = generate_singular_matrix(size, rank, -1, 1)
    assert m.shape == (size, size)
    # Check rank
    actual_rank = np.linalg.matrix_rank(m)
    assert actual_rank <= rank

def test_generate_singular_matrix_inversion_dataset():
    num_samples = 5
    matrix_size = 4
    max_rank = 2
    X, Y = generate_singular_matrix_inversion_dataset(num_samples, matrix_size, max_rank)
    
    assert X.shape == (num_samples, matrix_size, matrix_size)
    assert Y.shape == (num_samples, matrix_size, matrix_size)
    assert verify_pseudo_inversion(X, Y) is True

def test_verify_matrix_inversion_failure():
    X = np.array([[[1, 0], [0, 1]]])
    Y = np.array([[[2, 0], [0, 2]]]) # Not the inverse
    assert verify_matrix_inversion(X, Y) is False
