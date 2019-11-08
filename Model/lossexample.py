import numpy as np
from math import sqrt

def loss_example(y_true, y_pred):
    norm_sum = 0.
    for i in range(len(y_true)):
        # ytr and ypr are serialized square matrix
        ytr = y_true[i]
        ypr = y_pred[i]
        # we get the size of the square matrix
        msize = int(sqrt(ytr.shape[0]))
        eye = np.eye(msize)
        # with this we reconstruct the original matrix
        yt = np.ndarray(buffer=ytr, shape=(msize, msize), dtype=float)
        yp = np.ndarray(buffer=ypr, shape=(msize, msize), dtype=float)
        #Loss = norm(abs(Identity - A*A^-1))
        # A is the original matrix 
        # A^-1 is the inverted matrix.
        # yt is the ground truth inverse matrix. By inverting it again
        # we get the A.
        # yp is the result of the prediction of the inverse matrix
        # which is the A^-1
        yt_inv = np.linalg.inv(yt)
        r = np.dot(yt_inv, yt)
        norm_sum += np.linalg.norm((eye-r))
    return norm_sum/float(len(y_true))



if __name__ in "__main__":
    y_true = [[-0.2155,-0.2795,-0.7222, 0.4812,0.3795,-0.6584, 0.8146,-0.5542,0.1987],
              [0.4314,-1.0001,0.4554, -0.3503,-0.6756,-0.2024, -0.8144,0.4826,0.5320]]
    y_pred = [[-0.2087,-0.2759,-0.7942, 0.4648,0.3695,-0.7046, 0.9497,-0.5386,0.1822],
              [0.3086,-0.8765,0.4761, -0.4229,-0.6214,-0.1289, -0.7945,0.3255,0.6026]]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(loss_example(y_true, y_pred))



