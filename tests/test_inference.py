import numpy as np
import pytest
from deepmatrixinversion.inference import MatrixInversionInference

def test_split_matrix():
    inference = MatrixInversionInference(models_path=None, invert_mode="numpy")
    mx = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    A, B, C, D = inference.split_matrix(mx)
    
    np.testing.assert_array_equal(A, np.array([[1, 2], [5, 6]]))
    np.testing.assert_array_equal(B, np.array([[3, 4], [7, 8]]))
    np.testing.assert_array_equal(C, np.array([[9, 10], [13, 14]]))
    np.testing.assert_array_equal(D, np.array([[11, 12], [15, 16]]))

def test_block_matrix_inverse():
    # Use numpy for actual inversion, just testing the block formula logic
    inference = MatrixInversionInference(models_path=None, invert_mode="numpy")
    
    # 4x4 matrix
    mx = np.array([
        [4, 7, 2, 1],
        [1, 2, 1, 0],
        [8, 9, 2, 3],
        [1, 0, 4, 5]
    ])
    
    A, B, C, D = inference.split_matrix(mx)
    P_inv_block = inference.block_matrix_inverse(A, B, C, D)
    
    P_inv_numpy = np.linalg.inv(mx)
    
    np.testing.assert_allclose(P_inv_block, P_inv_numpy, atol=1e-7)

def test_invert_mismatch_size():
    # Mocking NN class for msize check
    inference = MatrixInversionInference(models_path=None, invert_mode="numpy")
    inference.nn.msize = 3 # Model trained for 3x3
    
    mx = np.array([[1, 2], [3, 4]]) # 2x2 matrix
    with pytest.raises(ValueError, match="The model is not suitted to predict the input matrix."):
        inference.invert([mx])
