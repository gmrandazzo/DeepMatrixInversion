"""inference.py

This file is part of DeepMatrixInversion.
Copyright (C) 2019 Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
PAutoDock is distributed under GPLv3 license.
To know more in detail how the license work,
please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/gpl-3.0.en.html"

"""

import numpy as np
import os
from pathlib import Path
from deepmatrixinversion.nnmodel import NN
from deepmatrixinversion.hub import download_model_from_hub

class MatrixInversionInference:
    def __init__(self, models_path: str, invert_mode="nn"):
        actual_path = models_path
        if models_path and not os.path.isdir(models_path):
            # Check if it looks like a Hugging Face repo (contains '/')
            if "/" in models_path:
                actual_path = download_model_from_hub(models_path)
        
        self.nn = NN(models_path=actual_path)
        self.invert_mode = invert_mode

    def _get_base_inv(self, mx: np.array):
        if self.invert_mode == "nn":
            return self.nn.predict(mx)
        else:
            return np.linalg.inv(mx)

    def split_matrix(self, mx: np.array):
        """
        Split matrix into blocks. 
        A is msize x msize, B is msize x (N-msize),
        C is (N-msize) x msize, D is (N-msize) x (N-msize)
        """
        m = self.nn.msize
        A = mx[:m, :m]
        B = mx[:m, m:]
        C = mx[m:, :m]
        D = mx[m:, m:]
        return A, B, C, D

    def block_matrix_inverse(
        self,
        A: np.array,
        B: np.array,
        C: np.array,
        D: np.array,
    ) -> np.array:
        """
        Sherman-Morrison-Woodbury matrix block inversion formula
        (Block Matrix Inversion using Schur Complement)
        """
        # Invert A recursively
        A_inv = self._invert_single(A)

        # Compute the Schur complement S = D - C * A^(-1) * B
        S = D - C @ A_inv @ B

        # Invert the Schur complement recursively
        S_inv = self._invert_single(S)

        # Construct the inverse of the block matrix
        # P^(-1) = [ A^(-1) + A^(-1) * B * S^(-1) * C * A^(-1),  -A^(-1) * B * S^(-1) ]
        #          [ -S^(-1) * C * A^(-1),                       S^(-1)            ]
        
        top_left = A_inv + A_inv @ B @ S_inv @ C @ A_inv
        top_right = -A_inv @ B @ S_inv
        bottom_left = -S_inv @ C @ A_inv
        bottom_right = S_inv

        P_inv = np.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])
        return P_inv

    def _invert_single(self, mx: np.array) -> np.array:
        """
        Recursively invert a single matrix using block inversion.
        Includes automatic scaling to fit the model's training range.
        """
        size = mx.shape[0]
        
        # Base case: Matrix matches model size
        if size == self.nn.msize:
            # Scale input to model's expected range (usually -1 to 1)
            # We use max absolute value to keep it centered if possible
            expected_max = max(abs(self.nn.range_min), abs(self.nn.range_max))
            current_max = np.max(np.abs(mx))
            
            if current_max > expected_max:
                k = current_max / expected_max
                # If A' = A/k, then (A')^-1 = k * A^-1
                # nn.predict returns A'^-1, so we divide by k to get A^-1
                scaled_mx = mx / k
                return self._get_base_inv(scaled_mx) / k
            else:
                return self._get_base_inv(mx)
        
        if size < self.nn.msize or size % self.nn.msize != 0:
            raise ValueError(
                f"Matrix size {size} must be a multiple of the model size {self.nn.msize}."
            )
        
        A, B, C, D = self.split_matrix(mx)
        return self.block_matrix_inverse(A, B, C, D)

    def invert(self, mxlst: list) -> np.array:
        """
        Invert a list of matrices
        """
        inv = []
        for mx in mxlst:
            inv.append(self._invert_single(mx))
        return np.array(inv)
