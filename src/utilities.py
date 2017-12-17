import tensorflow as tf
import keras
import numpy as np

from keras import backend as K
from keras.datasets import mnist
import itertools as itr
from functools import reduce
import operator

# %%


def batch_L_norm_distances(X: np.array, Y: np.array, ord=2) -> np.array:
    """
    Takes 2 arrays of N x d examples and calculates the p-norm between
    them. Result is dimension N. If the inputs are N x h x w etc, they
    are first flattened to be N x d
    """
    assert X.shape == Y.shape, "X and Y must have the same dimensions"
    N = X.shape[0]
    rest = X.shape[1:]
    d = reduce(operator.mul, rest, 1)  # product of leftover dimensions

    x = X.reshape(N, d)
    y = Y.reshape(N, d)

    if ord == 2:
        return np.sum((x - y) ** 2, axis=1)
    elif ord == 1:
        return np.sum(np.abs(x - y), axis=1)
    elif ord == 0:
        return np.isclose(x, y).astype(np.float).sum(axis=1)
        # return the number of entries in x that differ from y.
        # Use a tolerance to allow numerical precision errors.
    elif ord == np.inf:
        return np.max(np.abs(x - y), axis=1)
    else:
        raise NotImplementedError(
            "Norms other than 0, 1, 2, inf not implemented")


def tile_images(imlist: [np.array], horizontal=True) -> np.array:
    """
    Takes a list of images and tiles them into a single image for plotting 
    purposes.
    """
    ax = 1 if horizontal else 0
    tile = np.concatenate([x.squeeze() for x in imlist], axis=ax)
    return tile


def get_mnist():
    """
    Return the mnist data, scaled to [0,1].
    """
    num_classes = 10
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def mc_dropout_preds(model, x: tf.Tensor, n_mc: int) -> tf.Tensor:
    """
    Take a model, and a tensor of size batch_size x n_classes and return the
    result of doing n_mc stochastic forward passes as a n_mc x batch_size x
    n_classes tensor. This assumes the model has some VI layers like dropout or
    whatever, and that the model has been loaded with
    keras.backend.set_learning_phase(True). Also note that this takes and
    returns keras tensors, not arrays.
    """
    # tile x n_mc times and predict in a batch
    xs = K.stack(list(itr.repeat(x, n_mc)))
    mc_preds = K.map_fn(model, xs)  # [n_mc x batch_size x n_classes]
    return mc_preds


def m_entropy(mc_preds: tf.Tensor) -> tf.Tensor:
    """
    Take a tensor mc_preds [n_mc x batch_size x n_classes] and return the
    mean entropy of the predictive distribution across the MC samples.
    """

    entropy = K.sum(
        -mc_preds * tf.log(tf.clip_by_value(mc_preds, 1e-10, 1.0)),
        # avoid log 0
        axis=-1)  # n_mc x batch_size
    return K.mean(entropy, axis=0)  # batch_size

def entropy_m(mc_preds: tf.Tensor) -> tf.Tensor:
    """
    Take a tensor mc_preds [n_mc x batch_size x n_classes] and return the
    entropy of the mean predictive distribution across the MC samples.
    """
    expected_p = K.mean(mc_preds, axis=0)  # batch_size x n_classes
    H_expectation = K.sum(
        - expected_p * K.log(K.clip(expected_p, 1e-10, 1.0)),
        axis=-1
    )  # batch_size
    return H_expectation

def BALD(mc_preds: tf.Tensor) -> tf.Tensor:
    """
    Calculate the BALD (Bayesian Active Learning by Disagreement) of a model;
    the difference between the mean of the entropy and the entropy of the mean
    of the predicted distribution on the n_mc x batch_size x n_classes tensor
    mc_preds
    """
    # H := entropy. saves a few keystrokes
    expectation_H = m_entropy(mc_preds)  # batch_size
    H_expectation = entropy_m(mc_preds) 
    BALD = H_expectation - expectation_H
    return BALD

def batches_generator(x: np.array, y: np.array, batch_size=100):

    # todo; maybe add the ability to shuffle?
    N = x.shape[0]
    n_batches = N // batch_size + (N % batch_size != 0)
    for i in range(n_batches):
        lo = i * batch_size
        hi = (i + 1) * batch_size
        yield x[lo:hi], y[lo:hi]


def batch_eval(k_function, batch_iterable):
    """
    eval a keras function across a list, hiding the fact that keras requires
    you to pass a list to everything for some reason.
    """
    return [k_function([bx]) for bx in batch_iterable]
