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

import numpy as np
import tensorflow as tf
import toml
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from deepmatrixinversion.analytics import plot_exp_vs_pred
from deepmatrixinversion.dataset import generate_matrix_inversion_dataset
from deepmatrixinversion.loss import floss

# Some memory clean-up

K.clear_session()


class BatchMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.writer = tf.summary.create_file_writer(log_dir)
        self.batch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        with self.writer.as_default():
            tf.summary.scalar("batch_loss", logs["loss"], step=self.batch_count)
            tf.summary.scalar("batch_mse", logs["mse"], step=self.batch_count)
            tf.summary.scalar("batch_mae", logs["mae"], step=self.batch_count)
            self.batch_count += 1

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            tf.summary.scalar("epoch_loss", logs["loss"], step=epoch)
            tf.summary.scalar("epoch_mse", logs["mse"], step=epoch)
            tf.summary.scalar("epoch_mae", logs["mae"], step=epoch)


class NN:
    def __init__(
        self,
        msize: int = 3,
        range_min: float = -1,
        range_max: float = 1,
        nunits: int = 128,
        nlayers: int = 7,
        models_path: str = None,
    ):
        self.msize = int(msize)
        self.range_min = range_min
        self.range_max = range_max
        self.nunits = nunits
        self.nlayers = nlayers
        self.scaling_factor = range_max - range_min
        self.verbose = 1
        self.config_mappings = {
            "scaling_factor": ["scaling_factor", float],
            "msize": ["msize", int],
            "nunits": ["nunits", int],
            "nlayers": ["nlayers", int],
            "range_min": ["range_min", float],
            "range_max": ["range_max", float],
        }
        self.models = []
        if models_path:
            self.models = self.load_models(models_path)

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
        x = tf.keras.layers.Dense(self.nunits, activation="relu")(x)
        for _ in range(self.nlayers):
            skip = x
            for _ in range(4):
                y = tf.keras.layers.Dense(self.nunits * 2, activation="relu")(x)
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

    def build_simple_model(self):
        inp = tf.keras.layers.Input(shape=[self.msize, self.msize])
        x = tf.keras.layers.Flatten()(inp)
        x = tf.keras.layers.Dense(self.nunits, activation="relu")(x)
        for _ in range(self.nlayers):
            x = tf.keras.layers.Dense(self.nunits, activation="relu")(x)
        x = tf.keras.layers.Dense(self.msize * self.msize)(x)
        x = tf.keras.layers.Reshape([self.msize, self.msize])(x)
        model = tf.keras.models.Model(inp, x)
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
            models.append(load_model(str(file_), custom_objects={"floss": floss}))
        with open(f"{Path(model_path)}/config.toml", "r") as file:
            configs = toml.load(file)
            for config_key, (attr_name, attr_type) in self.config_mappings.items():
                if config_key in configs:
                    setattr(self, attr_name, attr_type(configs[config_key]))
        return models

    def train_single_model(
        self,
        model_id: int,
        nmx_samples: int = 1000000,
        num_epochs: int = 5000,
        batch_size: int = 1024,
        mout_path: str = "./",
        outname_suffix: str = "",
    ):
        logfile = "./logs/#b%d_#e%d_#mid_%d_" % (
            batch_size,
            num_epochs,
            model_id,
        )
        logfile += outname_suffix

        model_output = "%s/%d.keras" % (str(mout_path.absolute()), model_id)
        """
        TensorBoard(
                log_dir=logfile,
                histogram_freq=0,
                write_graph=False,
                write_images=False,
            ),
        """
        tb_callback = BatchMetricsCallback(logfile)
        callbacks_ = [
            tb_callback,
            ModelCheckpoint(
                model_output, monitor="val_loss", verbose=0, save_best_only=True
            ),
        ]

        model = self.build_model()
        # model = self.build_simple_model()
        for epoch in range(num_epochs):
            X, Y = generate_matrix_inversion_dataset(
                nmx_samples, self.msize, self.range_min, self.range_max
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

            # Y_train/self.scaling_factor # normalize target values, large target values hamper training
            history = model.fit(
                X_train,
                Y_train / self.scaling_factor,
                epochs=1,
                batch_size=batch_size,
                verbose=self.verbose,
                validation_data=(X_val, Y_val / self.scaling_factor),
                callbacks=callbacks_,
            )

            with tb_callback.writer.as_default():
                tf.summary.scalar("epoch_loss", history.history["loss"][0], step=epoch)
                tf.summary.scalar("epoch_mse", history.history["mse"][0], step=epoch)
                tf.summary.scalar("epoch_mae", history.history["mae"][0], step=epoch)
                if (epoch + 1) % 100 == 0:
                    model_ = load_model(model_output, custom_objects={"floss": floss})
                    Y_test_pred = model_.predict(X_test) * self.scaling_factor
                    mse = tf.reduce_mean(tf.square(Y_test - Y_test_pred))
                    del model_
                    print(
                        f"Epoch {epoch + 1}/num_epochs, Loss: {history.history['loss'][0]:.4f}, "
                        f"MSE: {history.history['mse'][0]:.4f}, MAE: {history.history['mae'][0]:.4f} "
                        f"Current MSE in testing: {mse}"
                    )
        return model

    def train(
        self,
        nmx_samples: int = 1000000,
        batch_size: int = 1024,
        num_epochs: int = 1000,
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
            data = {}
            for config_key, (attr_name, _) in self.config_mappings.items():
                data[config_key] = getattr(self, attr_name, None)
            toml.dump(data, file)

        from_id = len(self.models) + 1
        for model_id in range(n_repeats):
            self.train_single_model(
                model_id=(model_id + from_id),
                nmx_samples=nmx_samples,
                num_epochs=num_epochs,
                batch_size=batch_size,
                mout_path=mout_path,
                outname_suffix=strftime,
            )

    def model_validator(
        self, mout_path: str, nmx_sample: int = 1000000, plotout: str = None
    ):
        X, Y = generate_matrix_inversion_dataset(
            nmx_sample, self.msize, self.range_min, self.range_max
        )
        models = self.load_models(mout_path)
        predictions = [[] for _ in range(len(Y))]
        for model in models:
            ytp = model.predict(X) * self.scaling_factor
            for i, yp in enumerate(ytp):
                predictions[i].append(yp)
        plot_exp_vs_pred(Y, np.arrat(predictions), plotout)

    def predict(self, mx: np.array) -> np.array:
        """
        Predict a single matrix
        """
        inv = np.array(
            [model.predict(np.array([mx]), verbose=False)[0] for model in self.models]
        )
        return inv.mean(axis=0) * self.scaling_factor
