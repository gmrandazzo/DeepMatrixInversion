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
import tensorflow as tf
import toml
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    InputLayer,
    Reshape,
)
from tensorflow.keras.models import Sequential, load_model

from deepmatrixinversion.dataset import generate_matrix_inversion_dataset
from deepmatrixinversion.loss import floss

# Some memory clean-up

K.clear_session()


class NN:
    def __init__(self, msize: int, range_min: float, range_max: float):
        self.msize = int(msize)
        self.range_min = range_min
        self.range_max = range_max
        self.scaling_factor = range_max - range_min
        self.verbose = 1

    def get_scaling_factor(
        self,
    ):
        global_min = global_max = self.X[0].flatten()[0]
        for x in self.X:
            flat_matrix = x.flatten()
            global_min = min(global_min, np.min(flat_matrix))
            global_max = max(global_max, np.max(flat_matrix))
        return np.ceil(global_max - global_min)

    def build_model(self):
        inp = tf.keras.layers.Input(shape=[self.msize, self.msize])
        x = tf.keras.layers.Flatten()(inp)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        for _ in range(7):
            skip = x
            for _ in range(4):
                y = tf.keras.layers.Dense(256, activation="relu")(x)
                x = tf.keras.layers.concatenate([x, y])
            # x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(
                128,
                kernel_initializer=tf.keras.initializers.Zeros(),
                bias_initializer=tf.keras.initializers.Zeros(),
            )(x)
            x = skip + x
            # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(self.msize * self.msize)(x)
        x = tf.keras.layers.Reshape([self.msize, self.msize])(x)
        model = tf.keras.models.Model(inp, x)
        model.compile(
            loss=floss,
            optimizer=optimizers.Adam(learning_rate=1e-4),
            metrics=["mse", "mae", floss],
        )
        return model
        # model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(learning_rate=.00001))

    def build_model___(self, msize, nunits):
        model = Sequential()
        model.add(InputLayer(input_shape=(msize, msize)))
        model.add(Flatten())
        model.add(Dense(nunits, activation="relu"))
        for _ in range(7):
            model.add(ResidualBlock(nunits, 4))
        model.add(Dense(msize * msize, activation="linear"))
        model.add(Reshape((msize, msize)))
        model.compile(
            loss=floss,
            optimizer=optimizers.Adam(learning_rate=1e-4),
            metrics=["mse", "mae", floss],
        )
        return model

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

    def build_model__(self, msize, nunits):
        model = Sequential()
        # model.add(Dense(nunits, activation='relu', input_shape=(msize, msize,)))
        #
        # model.add(Flatten())
        model.add(
            InputLayer(
                input_shape=(
                    msize,
                    msize,
                )
            )
        )
        model.add(Flatten())
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

    def load_models(self, model_path: str) -> list:
        p = Path(model_path).glob("**/*.keras")
        files = [x for x in p if x.is_file()]
        models = []
        for file_ in files:
            print("Load %s" % (file_))
            models.append(load_model(str(file_), custom_objects={"floss": floss}))
        with open(f"{Path(model_path)}/config.toml", "r") as file:
            configs = toml.load(file)
            if "scaling_factor" in configs:
                self.scaling_factor = float(configs["scaling_factor"])
        return models

    def train_single_model(
        self,
        model_id: int,
        num_epochs: int = 5000,
        batch_size: int = 1024,
        mout_path: str = "./",
        outname_suffix: str = "",
    ):
        model = self.build_model()
        for _ in range(num_epochs):
            X, Y = generate_matrix_inversion_dataset(
                1000000, 3, self.range_min, self.range_max
            )
            X_subset, X_val, Y_subset, Y_val = train_test_split(
                X,
                Y,
                test_size=0.20,
                random_state=datetime.now().microsecond,
            )
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_subset,
                Y_subset,
                test_size=0.20,
                random_state=datetime.now().microsecond,
            )

            logfile = "./logs/#b%d_#e%d_#mid_%d_" % (
                batch_size,
                num_epochs,
                model_id,
            )
            logfile += outname_suffix

            model_output = "%s/%d.keras" % (str(mout_path.absolute()), model_id)
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
            # Y_train/self.scaling_factor # normalize target values, large target values hamper training
            model.fit(
                X_train,
                Y_train / self.scaling_factor,
                epochs=1,
                batch_size=batch_size,
                verbose=self.verbose,
                validation_data=(X_val, Y_val / self.scaling_factor),
                callbacks=callbacks_,
            )
            model_ = load_model(model_output, custom_objects={"floss": floss})
            Y_test_pred = model_.predict(X_test) * self.scaling_factor
            mse = tf.reduce_mean(tf.square(Y_test - Y_test_pred))
            print(f"Current MSE in testing: {mse}")
            del model_
        return model

    def train(
        self,
        batch_size: int = 1024,
        num_epochs: int = 5000,
        n_repeats: int = 5,
        mout_path_: str = "./",
    ):
        """
        Train models
        """
        strftime = time.strftime("%Y%m%d%H%M%S")
        mout_path = Path("%s_%s" % (mout_path_, strftime))
        mout_path.mkdir()

        with open(f"{mout_path}/config.toml", "w") as file:
            data = {"scaling_factor": self.scaling_factor}
            toml.dump(data, file)

        for model_id in range(n_repeats):
            self.train_single_model(
                model_id=model_id + 1,
                num_epochs=num_epochs,
                batch_size=batch_size,
                mout_path=mout_path,
                outname_suffix=strftime,
            )

        X, Y = generate_matrix_inversion_dataset(
            1000000, 3, self.range_min, self.range_max
        )
        models = self.load_models(mout_path)
        predictions = [[] for _ in range(len(Y))]
        for model in models:
            ytp = model.predict(X) * self.scaling_factor
            for i, yp in enumerate(ytp):
                predictions[i].append(yp)

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
            for row in Y[i]:
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
        plt.savefig("training_validation.png", dpi=300, bbox_inches="tight")

    def predict(self, model_path, pred_inverse_out):
        # Load input matrix to predicts
        # Load models
        models = self.load_models(model_path)
        inverse = []
        for row in self.X:
            inv = []
            for model in models:
                inv.append(model.predict(np.array([row]), verbose=False))
            inv = np.array(inv)
            inverse.append(inv.mean(axis=0).tolist()[0] * self.scaling_factor)
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
