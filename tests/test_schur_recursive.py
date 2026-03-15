import numpy as np
import pytest
from deepmatrixinversion.inference import MatrixInversionInference

def test_recursive_9x9_with_3x3_blocks():
    # Setup inference with msize=3
    inference = MatrixInversionInference(models_path=None, invert_mode="numpy")
    inference.nn.msize = 3
    
    # Create a random 9x9 invertible matrix
    np.random.seed(42)
    mx = np.random.uniform(-1, 1, (9, 9))
    while np.abs(np.linalg.det(mx)) < 0.1:
        mx = np.random.uniform(-1, 1, (9, 9))
        
    # Invert using recursive block formula
    # This should call _invert_single for 9x9
    #   -> calls block_matrix_inverse with A(3x3) and D(6x6)
    #      -> A_inv (3x3) base case
    #      -> S_inv (6x6) recursive call
    #         -> S_inv calls block_matrix_inverse with A'(3x3) and D'(3x3)
    #            -> both base cases
    
    inv_recursive = inference.invert([mx])[0]
    inv_numpy = np.linalg.inv(mx)
    
    np.testing.assert_allclose(inv_recursive, inv_numpy, atol=1e-7)

def test_recursive_invalid_size():
    inference = MatrixInversionInference(models_path=None, invert_mode="numpy")
    inference.nn.msize = 3
    
    mx = np.random.uniform(-1, 1, (5, 5)) # 5 is not a multiple of 3
    with pytest.raises(ValueError, match="must be a multiple of the model size 3"):
        inference.invert([mx])
