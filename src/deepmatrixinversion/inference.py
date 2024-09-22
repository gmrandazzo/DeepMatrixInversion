"""inference.py

This file is part of DeepMatrixInversion.
Copyright (C) 2019 Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
PAutoDock is distributed under GPLv3 license.
To know more in detail how the license work,
please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/gpl-3.0.en.html"

"""

from deepmatrixinversion.nnmodel import NN
import numpy as np

class MatrixInversionInference:
    def __init__(self, models_path:str, invert_mode='nn'):
        self.nn = NN(models_path=models_path)
        if invert_mode == 'nn':
            self.inv = self.nn.predict
        else:
            self.inv = np.linalg.inv

    def split_matrix(self, mx: np.array):
        if mx.shape[0] % 2 != 0 or mx.shape[1] % 2 != 0:
            raise ValueError("Matrix dimensions must be even")
        rows, cols = mx.shape
        row_mid = rows // 2
        col_mid = cols // 2
        A = mx[:row_mid, :col_mid]
        B = mx[:row_mid, col_mid:]
        C = mx[row_mid:, :col_mid]
        D = mx[row_mid:, col_mid:]
        return A, B, C, D

    def block_matrix_inverse(self,
            A: np.array,
            B: np.array,
            C: np.array,
            D: np.array,) -> np.array:
        """
        Sherman-Morrison-Woodbury matrix block inversion formula
        
        P = [ A  B ]
            [ C  D ]

        The inverse of this block matrix can be computed using:

        P^(-1) = [ A^(-1) + A^(-1) * B * (D - C * A^(-1) * B)^(-1) * C * A^(-1),  -A^(-1) * B * (D - C * A^(-1) * B)^(-1) ]
                [ -(D - C * A^(-1) * B)^(-1) * C * A^(-1),               (D - C * A^(-1) * B)^(-1) ]

        Where:
        - A and D are square matrices.
        - The Schur complement is defined as S = D - C * A^(-1) * B.
        """
        # Invert A
        A_inv = self.inv(A)

        # Compute the Schur complement
        S = D - C @ A_inv @ B

        # Invert the Schur complement
        S_inv = self.inv(S)

        # Construct the inverse of the block matrix
        P_inv = np.block([
            [A_inv + A_inv @ B @ S_inv @ C @ A_inv, -A_inv @ B @ S_inv],
            [-S_inv @ C @ A_inv, S_inv]
        ])
        return P_inv

    def invert(self, mxlst: list) -> np.array:
        """
        Invert a lsit of matrices
        """
        inv = []
        for mx in mxlst:
            if mx.shape[0] == self.nn.msize:
                inv.append(self.nn.predict(mx))
            elif mx.shape[0] % 2 == 0 and  mx.shape[0]/2. == self.nn.msize:
                A, B, C, D = self.split_matrix(mx)
                inv.append(self.block_matrix_inverse(A, B, C, D))
            else:
                raise ValueError("The model is not suitted to predict the input matrix.")
        return np.array(inv)
