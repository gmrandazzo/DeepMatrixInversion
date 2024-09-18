#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lossexample.py

This file is part of DeepMatrixInversion.
Copyright (C) 2019 Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
PAutoDock is distributed under GPLv3 license.
To know more in detail how the license work,
please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/gpl-3.0.en.html"

"""

from deepmatrixinversion.loss import floss
import tensorflow as tf
import numpy as np


def numpy_loss_example(y_true, y_pred):
    eye = []
    for _ in range(len(y_true)):
        eye.append(np.eye(3))
    yt_inv = np.linalg.inv(y_true)
    r = np.matmul(yt_inv, y_pred)
    r = eye-r
    r = np.abs(r)
    r = np.square(r)
    r = np.sum(np.sum(r, axis=1), axis=1)
    r = np.sqrt(r)
    return np.array(r).mean()


def tf_loss_example(y_true, y_pred):
    yt = tf.constant(y_true, dtype=tf.float32)
    yp = tf.constant(y_pred, dtype=tf.float32)
    return floss(yt, yp)

def test_floss():
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
    assert np.fabs(numpy_loss_example(y_true, y_pred)- 0.20402847) < 1e-4
    assert np.fabs(tf_loss_example(y_true, y_pred)- 0.20402847) < 1e-4
