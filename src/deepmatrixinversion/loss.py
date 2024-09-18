#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""loss.py

This file is part of DeepMatrixInversion.
Copyright (C) 2019 Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
PAutoDock is distributed under GPLv3 license.
To know more in detail how the license work,
please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/gpl-3.0.en.html"

"""

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable()
def floss(y_true, y_pred):
    """
    Loss function

    loss = || I - AA^{-1}||
    where
    A is the original not inverted matrix
    A^{-1} is the inverted matrix
    I is the identiy matrix
    
    y_true is the true inverse
    y_pred is the predicted inverse
    
    """

    """
    Fast Forbenius L-2 Norm
    """
    shape = tf.shape(y_true)
    batch_size = shape[0]
    msize = shape[1]
    eye = tf.eye(
        msize,
        batch_shape=[batch_size]
    )
    res = tf.linalg.matmul(tf.linalg.inv(y_true), y_pred)
    res = tf.cast(res, tf.float32)
    res = tf.subtract(eye, res)
    res = tf.abs(res)
    res = tf.square(res)
    res = tf.reduce_sum(tf.reduce_sum(res, axis=1), axis=1)
    res = tf.sqrt(res)
    res = tf.reduce_mean(res)
    return res