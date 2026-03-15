import os
import numpy as np
import pytest
from deepmatrixinversion.io import read_dataset, write_dataset

def test_read_write_dataset(tmp_path):
    X = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    out_file = tmp_path / "test_dataset.mx"
    
    # Write dataset
    write_dataset(X, str(out_file))
    
    # Check if file exists
    assert out_file.exists()
    
    # Read dataset
    X_read = read_dataset(str(out_file))
    
    # Assert they are equal
    np.testing.assert_array_equal(X, X_read)

def test_read_dataset_no_end_marker(tmp_path):
    # Test our fix for reading the last matrix if END is missing
    out_file = tmp_path / "test_no_end.mx"
    with open(out_file, "w") as f:
        f.write("1.0,2.0\n")
        f.write("3.0,4.0\n")
        # No END marker
        
    X_read = read_dataset(str(out_file))
    expected = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    np.testing.assert_array_equal(expected, X_read)

def test_read_dataset_mixed_end_marker(tmp_path):
    # One with END, one without
    out_file = tmp_path / "test_mixed_end.mx"
    with open(out_file, "w") as f:
        f.write("1.0,2.0\n")
        f.write("3.0,4.0\n")
        f.write("END\n")
        f.write("5.0,6.0\n")
        f.write("7.0,8.0\n")
        # No END marker at last matrix
        
    X_read = read_dataset(str(out_file))
    expected = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ])
    np.testing.assert_array_equal(expected, X_read)
