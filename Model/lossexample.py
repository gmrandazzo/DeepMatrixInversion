#!/usr/bin/env python
import tensorflow as tf
import numpy as np


def loss_example(y_true, y_pred):
    """
        Numpy version of the loss function
        Loss = norm(abs(Identity - A*A^-1))

    norm_sum = 0.
    for i in range(len(y_true)):
        msize = y_true[1].shape[1]
        eye = np.eye(msize)
        # A is the original matrix
        # A^-1 is the inverted matrix.
        # y_ttrue is the ground truth inverse matrix. By inverting it again
        # we get the A.
        # y_pred is the result of the prediction of the inverse matrix
        # which is the A^-1
        yt_inv = np.linalg.inv(y_true[i])
        r = np.dot(yt_inv, y_pred[i])
        norm_sum += np.linalg.norm((eye-r))
    return norm_sum/float(len(y_true))
    """
    eye = []
    for i in range(len(y_true)):
        eye.append(np.eye(3))
    yt_inv = np.linalg.inv(y_true)
    r = np.matmul(yt_inv, y_pred)
    r = eye-r
    r = np.abs(r)
    r = np.square(r)
    r = np.sum(np.sum(r, axis=1), axis=1)
    norms = np.sqrt(r)
    return np.array(norms).mean()


def loss_example_tf(y_true, y_pred):
    """
        Tensorflow Loss function version
    """
    sess = tf.InteractiveSession()
    yt = tf.constant(y_true, dtype=tf.float32)
    yp = tf.constant(y_pred, dtype=tf.float32)
    """
    Version iterate trough samples...
    def single_floss(elems):
        eye = tf.eye(3)
        return tf.norm(tf.subtract(eye, tf.linalg.matmul(tf.linalg.inv(elems[0]), elems[1])), ord='euclidean')
    elems = (yt, yp)
    res = tf.reduce_mean(tf.map_fn(single_floss, elems, dtype=tf.float32))
    """
    eye = tf.eye(3, batch_shape=[tf.shape(yt)[0]])
    res = tf.subtract(eye, tf.linalg.matmul(tf.linalg.inv(yt), yp))
    res = tf.abs(res)
    res = tf.square(res)
    res = tf.reduce_sum(tf.reduce_sum(res, axis=1), axis=1)
    res = tf.sqrt(res)
    # print(res.eval())
    # res = tf.norm(elems)
    res = tf.reduce_mean(res)
    retval = res.eval()
    sess.close()
    return retval


if __name__ in "__main__":
    y_true = [[[-0.2155,-0.2795,-0.7222],
               [0.4812,0.3795,-0.6584],
               [0.8146,-0.5542,0.1987]],
              [[0.4314,-1.0001,0.4554],
               [-0.3503,-0.6756,-0.2024],
               [-0.8144,0.4826,0.5320]]]

    y_pred = [[[-0.2087,-0.2759,-0.7942],
               [0.4648,0.3695,-0.7046],
               [0.9497,-0.5386,0.1822]],
              [[0.3086,-0.8765,0.4761],
               [-0.4229,-0.6214,-0.1289],
               [-0.7945,0.3255,0.6026]]]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print("The loss result must be: 0.20402847")
    print("Numpy implementation %.8f" % (loss_example(y_true, y_pred)))
    print("TF implementation %.8f" % (loss_example_tf(y_true, y_pred)))
