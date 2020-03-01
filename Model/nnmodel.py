#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (c) 2019 gmrandazzo@gmail.com
# This file is part of DeepMatrixInversion.
# You can use,modify, and distribute it under
# the terms of the GNU General Public Licenze, version 3.
# See the file LICENSE for details

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
if int(tf.__version__[0]) == 1:
    import keras as keras
else:
    from tensorflow import keras as keras

from keras import backend as K
# Some memory clean-up

from datetime import datetime
import numpy as np
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from keras.layers import Flatten, Dense, Dropout, Reshape, BatchNormalization
from keras.models import Sequential, Model
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sys import argv
import time

K.clear_session()


def ReadMatrix(fmx):
    """
    mx = []
    row = []
    f = open(fmx, "r")
    for line in f:
        line = line.strip()
        if "END" in line:
            mx.append(row.copy())
            del row[:]
        else:
            row.extend(str.split(line, ","))
    f.close()
    """
    mxs = []
    mx = []
    f = open(fmx, "r")
    for line in f:
        line = line.strip()
        if "END" in line:
            mxs.append(mx.copy())
            del mx[:]
        else:
            mx.append(str.split(line, ","))
    f.close()
    return np.array(mxs).astype(float)


class NN(object):
    def __init__(self, data_mx_input, target_mx_output):
        if data_mx_input is not None and target_mx_output is not None:
            self.X = ReadMatrix(data_mx_input)
            self.y = ReadMatrix(target_mx_output)
            # print(self.X.shape)
        elif data_mx_input is not None and target_mx_output is None:
            self.X = ReadMatrix(data_mx_input)
            self.y = None
        else:
            print("Error in NN object")
            return
        self.msize = self.X.shape[1]
        self.verbose = 0
        if int(tf.__version__[0]) == 1:
            self.sess = K.get_session()
        else:
            self.sess = tf.compat.v1.Session()

    def floss(self, y_true, y_pred):
        # loss = || I - AA^{-1}||
        # y_true is the true inverse
        # y_pred is the predicted inverse
        # A is the original not inverted matrix
        # A^{-1} is the inverted matrix
        # I is the identiy matrix
        """
        Iterate trough tensor elements in y_true and y_pred
        WARNING: SLOW!!
        def single_floss(elems):
            eye = tf.eye(self.msize)
            return tf.norm(tf.subtract(eye, tf.linalg.matmul(tf.linalg.inv(elems[0]), elems[1])), ord='euclidean')
        elems_ = (y_true, y_pred)
        return tf.reduce_mean(tf.map_fn(single_floss, elems_, dtype=tf.float32))
        """

        """
        Iterate trough matrix and calculate the norm of each matrix in a tensor
        WARNING: SLOW!!
        def apply_norm(elem):
            return tf.norm(elem)
        eye = tf.eye(self.msize,
                     batch_shape=[tf.shape(y_true)[0]])
        # return tf.norm(tf.subtract(eye, tf.linalg.matmul(tf.linalg.inv(y_true), y_pred)), ord='euclidean')
        elems = tf.subtract(eye, tf.linalg.matmul(tf.linalg.inv(y_true), y_pred))
        norms = tf.map_fn(apply_norm, elems, dtype=tf.float32)
        """

        """
        Fast Forbenius L-2 Norm
        """
        eye = tf.eye(self.msize,
                     batch_shape=[tf.shape(y_true)[0]])
        res = tf.subtract(eye, tf.linalg.matmul(tf.linalg.inv(y_true), y_pred))
        res = tf.abs(res)
        res = tf.square(res)
        res = tf.reduce_sum(tf.reduce_sum(res, axis=1), axis=1)
        res = tf.sqrt(res)
        res = tf.reduce_mean(res)
        return res

    def build_model(self, msize, nunits):
        model = Sequential()
        # model.add(BatchNormalization(input_shape=(msize, msize,)))
        # model.add(Flatten())
        model.add(Flatten(input_shape=(msize, msize, )))
        model.add(Dense(nunits, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(nunits, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(nunits, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(nunits, activation='relu'))
        model.add(Dense(msize * msize))
        model.add(Reshape((msize, msize)))
        """
        # remove comments if you want to test floss
        model.compile(loss=self.floss,
                      optimizer=optimizers.Adam(),
                      metrics=['mse', 'mae', self.floss])
        """
        model.compile(loss=self.floss,
                      optimizer=optimizers.Adam(),
                      metrics=['mse', 'mae', self.floss])

        return model

    def train(self,
              batch_size_,
              num_epochs,
              nunits,
              n_splits,
              n_repeats,
              mout_path_):
        """
        Train models with a RepeatedKFold model type
        """
        predictions = []
        for i in range(len(self.X)):
            predictions.append([])

        strftime = time.strftime("%Y%m%d%H%M%S")
        mout_path = Path("%s_%s" % (mout_path_, strftime))
        mout_path.mkdir()
        rkf = RepeatedKFold(n_splits=n_splits,
                            n_repeats=n_repeats,
                            random_state=datetime.now().microsecond)
        mid = 0
        for train_index, test_index in rkf.split(self.X):
            K.clear_session()
            model = self.build_model(self.msize, nunits)
            print(model.summary())
            X_subset, X_test = self.X[train_index], self.X[test_index]
            y_subset, y_test = self.y[train_index], self.y[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_subset,
                                                              y_subset,
                                                              test_size=0.33,
                                                              random_state=datetime.now().microsecond)
            print("Train: %d Validation: %d Test: %d" % (len(X_train),
                                                         len(X_val),
                                                         len(X_test)))

            logfile = ("./logs/#b%d_#e%d_#u%d_#mid_%d_" % (batch_size_,
                                                           num_epochs,
                                                           nunits,
                                                           mid))
            logfile += strftime

            model_output = "%s/%d.h5" % (str(mout_path.absolute()), mid)
            callbacks_ = [TensorBoard(log_dir=logfile,
                                      histogram_freq=0,
                                      write_graph=False,
                                      write_images=False),
                          ModelCheckpoint(model_output,
                                          monitor='val_loss',
                                          verbose=0,
                                          save_best_only=True)]

            model.fit(X_train, y_train,
                      epochs=num_epochs,
                      batch_size=batch_size_,
                      verbose=self.verbose,
                      validation_data=(X_val, y_val),
                      callbacks=callbacks_)

            model_ = load_model(model_output,
                                custom_objects={"floss": self.floss})
            ytp = model_.predict(X_test)
            for i in range(len(test_index)):
                predictions[test_index[i]].append(ytp[i])
            mid += 1

        ypred = []
        ytrue = []
        for i in range(len(predictions)):
            # predictions[i] contains multiple prediction of the same things.
            # So average by column
            prow = np.array(predictions[i]).mean(axis=0)
            for row in prow:
                for number in row:
                    ypred.append(number)
            for row in self.y[i]:
                for number in row:
                    ytrue.append(number)
        ytrue = np.array(ytrue)
        ypred = np.array(ypred)
        print("R2: %.4f MSE: %.4f MAE: %.4f" % (r2_score(ytrue, ypred),
                                                mse(ytrue, ypred),
                                                mae(ytrue, ypred)))
        plt.scatter(ytrue, ypred, s=3)
        plt.xlabel('Experimental inverted matrix values')
        plt.ylabel('Predicted inverted matrix values')
        plt.show()

    def predict(self, model_path, pred_inverse_out):
        # Load input matrix to predicts
        # X = ReadMatrix(inputmx, model_path)
        # Load models
        p = Path(model_path).glob('**/*.h5')
        files = [x for x in p if x.is_file()]
        models = []
        for file_ in files:
            # print("Load %s" % (file_))
            models.append(load_model(str(file_),
                                     custom_objects={"floss": self.floss}))
        inverse = []
        for row in self.X:
            inv = []
            for model in models:
                inv.append(model.predict(np.array([row])))
            inv = np.array(inv)
            inverse.append(inv.mean(axis=0).tolist()[0])
        fo = open(pred_inverse_out, "w")
        for inv in inverse:
            for i in range(self.msize):
                for j in range(self.msize-1):
                    fo.write("%.4f," % (inv[i][j]))
                fo.write("%.4f\n" % (inv[i][-1]))
            fo.write("END\n")
        fo.close()

        if self.y is not None:
            # Calculate the prediction score!
            ypred = []
            ytrue = []
            for i in range(len(inverse)):
                for row in inverse[i]:
                    for number in row:
                        ypred.append(number)
                for row in self.y[i]:
                    for number in row:
                        ytrue.append(number)
            ytrue = np.array(ytrue)
            ypred = np.array(ypred)
            print("R2: %.4f MSE: %.4f MAE: %.4f" % (r2_score(ytrue, ypred),
                                                    mse(ytrue, ypred),
                                                    mae(ytrue, ypred)))
            plt.scatter(ytrue, ypred, s=3)
            plt.xlabel('Experimental inverted matrix values')
            plt.ylabel('External predicted inverted matrix values')
            plt.show()
