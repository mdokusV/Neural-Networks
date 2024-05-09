import random

import numpy as np
from scipy.special import expit

RANDOM = True
c = 2
epsilon = 1e-5
beta = 2
max_iter = 1e6
BATCH = 1


def sigmoid(z):
    return expit(beta * z)


def sigmoid_prime(z):
    sigmoid_out = sigmoid(z)
    return beta * sigmoid_out * (1 - sigmoid_out)


def layer(x: np.ndarray, w: np.ndarray):
    return np.dot(w, x.tolist() + [1])


def gradient_outer(
    error_prime: np.float32,
    weights: np.ndarray,
    values: np.ndarray,
    sigmoid_prime_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    err_sig = error_prime * sigmoid_prime_value
    return (
        np.multiply(err_sig, values.tolist() + [1]),
        np.multiply(err_sig.reshape(-1, 1), weights[:, :-1]),
    )


def gradient_inner(
    prev_partial_values: np.ndarray,
    sigmoid_prime_values: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return (prev_partial_values * sigmoid_prime_values).reshape(-1, 1) * (
        [values.tolist() + [1]] * prev_partial_values.shape[0]
    )


def error(real: np.ndarray, predicted: np.ndarray):
    return np.sum(np.square(real - predicted))


def error_prime(predicted, expected):
    return predicted - expected


def loop_condition_value(
    weights_inner: np.ndarray,
    weights_outer: np.ndarray,
    weights_inner_new: np.ndarray,
    weights_outer_new: np.ndarray,
):
    inner_max = np.max(np.abs(weights_inner - weights_inner_new))
    outer_max = np.max(np.abs(weights_outer - weights_outer_new))
    return np.max([inner_max, outer_max])


def propagate(
    input, weights_inner, weights_outer
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    layer_one = layer(input, weights_inner)
    layer_one_sigmoid = sigmoid(layer_one)
    layer_two = layer(layer_one_sigmoid, weights_outer)
    layer_two_sigmoid = sigmoid(layer_two)

    return (layer_two_sigmoid, layer_two, layer_one_sigmoid, layer_one)
