"""nnmodel.py

This file is part of DeepMatrixInversion.
Copyright (C) 2019 Giuseppe Marco Randazzo <gmrandazzo@gmail.com>
PAutoDock is distributed under GPLv3 license.
To know more in detail how the license work,
please read the file "LICENSE" or
go to "http://www.gnu.org/licenses/gpl-3.0.en.html"

"""

import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.models import Sequential, load_model

from deepmatrixinversion.io import read_dataset
from deepmatrixinversion.loss import floss

# Some memory clean-up


K.clear_session()


class NN:
    def __init__(self, data_mx_input, target_mx_output):
        if data_mx_input is not None and target_mx_output is not None:
            self.X = read_dataset(data_mx_input)
            self.y = read_dataset(target_mx_output)
            # print(self.X.shape)
        elif data_mx_input is not None and target_mx_output is None:
            self.X = read_dataset(data_mx_input)
            self.y = None
        else:
            print("Error in NN object")
            return
        self.msize = self.X.shape[1]
        self.verbose = 0

    def build_model_(self, msize, nunits):
        starting_units = msize**4
        model = Sequential(
            [
                Dense(
                    starting_units,
                    activation="relu",
                    input_shape=(
                        msize,
                        msize,
                    ),
                ),
                Dense(nunits, activation="relu"),
                Dense(nunits, activation="relu"),
                # Dense(nunits, activation='relu'),
                # Dense(int(nunits/2.), activation='relu'),
                Dense(msize, activation="linear"),
                Reshape((msize, msize)),
            ]
        )

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=floss,
            metrics=["mse", "mae", floss],
        )
        return model

    def build_model(self, msize, nunits):
        model = Sequential()
        # model.add(Dense(nunits, activation='relu', input_shape=(msize, msize,)))
        #
        # model.add(Flatten())
        # model.add(InputLayer(
        #        input_shape=(
        #            msize,
        #            msize,
        #        )
        #    )
        # )
        model.add(
            Flatten(
                input_shape=(
                    msize,
                    msize,
                )
            )
        )
        model.add(BatchNormalization())
        model.add(Dense(msize**4, activation="relu"))
        # model.add(Dense(nunits, activation='relu'))
        model.add(Dropout(0.1))
        # model.add(Dense(nunits, activation='relu'))
        # model.add(Dropout(0.1))
        # model.add(Dense(nunits, activation='relu'))
        # model.add(Dropout(0.1))
        # model.add(Dense(nunits, activation='relu'))
        model.add(Dense(msize * msize, activation="linear"))
        model.add(Reshape((msize, msize)))
        model.compile(
            loss=floss,
            optimizer=optimizers.Adam(learning_rate=1e-4),
            metrics=["mse", "mae", floss],
        )
        return model

    def train(self, batch_size_, num_epochs, nunits, n_splits, n_repeats, mout_path_):
        """
        Train models with a RepeatedKFold model type
        """
        predictions = []
        for i in range(len(self.X)):
            predictions.append([])

        strftime = time.strftime("%Y%m%d%H%M%S")
        mout_path = Path("%s_%s" % (mout_path_, strftime))
        mout_path.mkdir()
        rkf = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=datetime.now().microsecond,
        )
        mid = 0
        for train_index, test_index in rkf.split(self.X):
            K.clear_session()
            model = self.build_model(self.msize, nunits)
            print(model.summary())
            X_subset, X_test = self.X[train_index], self.X[test_index]
            y_subset, _ = self.y[train_index], self.y[test_index]
            X_train, X_val, y_train, y_val = train_test_split(
                X_subset,
                y_subset,
                test_size=0.33,
                random_state=datetime.now().microsecond,
            )
            print(
                "Train: %d Validation: %d Test: %d"
                % (len(X_train), len(X_val), len(X_test))
            )

            logfile = "./logs/#b%d_#e%d_#u%d_#mid_%d_" % (
                batch_size_,
                num_epochs,
                nunits,
                mid,
            )
            logfile += strftime

            model_output = "%s/%d.keras" % (str(mout_path.absolute()), mid)
            callbacks_ = [
                TensorBoard(
                    log_dir=logfile,
                    histogram_freq=0,
                    write_graph=False,
                    write_images=False,
                ),
                ModelCheckpoint(
                    model_output, monitor="val_loss", verbose=0, save_best_only=True
                ),
            ]

            model.fit(
                X_train,
                y_train,
                epochs=num_epochs,
                batch_size=batch_size_,
                verbose=self.verbose,
                validation_data=(X_val, y_val),
                callbacks=callbacks_,
            )

            model_ = load_model(model_output, custom_objects={"floss": floss})
            ytp = model_.predict(X_test)
            for i in range(len(test_index)):
                predictions[test_index[i]].append(ytp[i])
            mid += 1

        # Final output
        ytrue = []
        ypred = []
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
        print(
            "R2: %.4f MSE: %.4f MAE: %.4f"
            % (r2_score(ytrue, ypred), mse(ytrue, ypred), mae(ytrue, ypred))
        )
        plt.scatter(ytrue, ypred, s=3)
        plt.xlabel("Experimental inverted matrix values")
        plt.ylabel("Predicted inverted matrix values")
        plt.show()

    def predict(self, model_path, pred_inverse_out):
        # Load input matrix to predicts
        # Load models
        p = Path(model_path).glob("**/*.keras")
        files = [x for x in p if x.is_file()]
        models = []
        for file_ in files:
            print("Load %s" % (file_))
            models.append(load_model(str(file_), custom_objects={"floss": floss}))
        inverse = []
        for row in self.X:
            inv = []
            for model in models:
                inv.append(model.predict(np.array([row]), verbose=False))
            inv = np.array(inv)
            inverse.append(inv.mean(axis=0).tolist()[0])
        fo = open(pred_inverse_out, "w")
        for inv in inverse:
            for i in range(self.msize):
                for j in range(self.msize - 1):
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
            print(
                "R2: %.4f MSE: %.4f MAE: %.4f"
                % (r2_score(ytrue, ypred), mse(ytrue, ypred), mae(ytrue, ypred))
            )
            plt.scatter(ytrue, ypred, s=3)
            plt.xlabel("Experimental inverted matrix values")
            plt.ylabel("External predicted inverted matrix values")
            plt.show()
