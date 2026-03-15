import numpy as np
import pytest
import tensorflow as tf
from deepmatrixinversion.nnmodel import NN

def test_nn_initialization():
    nn = NN(msize=3, nunits=64, nlayers=2)
    assert nn.msize == 3
    assert nn.nunits == 64
    assert nn.nlayers == 2
    assert nn.scaling_factor == 2 # 1 - (-1)

def test_get_scaling_factor():
    nn = NN(msize=2)
    nn.X = np.array([
        [[1, 2], [3, 4]],
        [[-1, -2], [-3, -4]]
    ])
    # global_min = -4, global_max = 4
    # scaling_factor = ceil(4 - (-4)) = 8
    assert nn.get_scaling_factor() == 8.0

def test_build_model():
    nn = NN(msize=3, nunits=128, nlayers=2)
    model = nn.build_model()
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 3, 3)
    assert model.output_shape == (None, 3, 3)

def test_build_simple_model():
    nn = NN(msize=2, nunits=32, nlayers=1)
    model = nn.build_simple_model()
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 2, 2)
    assert model.output_shape == (None, 2, 2)
