import random

import numpy as np

RANDOM = False
c = 0.1
epsilon = 1e-5
beta = 1.0
max_iter = 1e4


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-beta * z))


def sigmoid_prime(z):
    return beta * sigmoid(z) * (1 - sigmoid(z))


def layer(x: np.ndarray, w: np.ndarray):
    x = np.append(x, 1)
    return np.dot(w, x)


def gradient_outer(
    error_prime: np.float32,
    weights: np.ndarray,
    values: np.ndarray,
    sigmoid_prime_value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    err_sig = error_prime * sigmoid_prime_value
    return (
        np.dot(err_sig, np.append(values, 1)) + epsilon / 10,
        np.dot(err_sig, weights[:-1]),
    )


def gradient_inner(
    prev_partial_values: np.ndarray,
    sigmoid_prime_values: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    diag_val = np.diag(prev_partial_values)
    diag_sig = np.diag(sigmoid_prime_values)
    scalar_matrix = np.multiply(diag_val, diag_sig)
    values = np.append(values, 1)
    values = np.tile(values, (2, 1))
    return np.dot(scalar_matrix, values) + epsilon / 10


def error(real: np.ndarray, predicted: np.ndarray):
    return np.sum(np.square(real - predicted))


def error_prime(predicted, expected):
    return predicted - expected


def loop_condition(previous_weight: np.ndarray, current_weight: np.ndarray):
    return np.max(np.abs(previous_weight - current_weight)) > epsilon


def propagate(
    input, weights_inner, weights_outer
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layer_one = layer(input, weights_inner)
    layer_one_sigmoid = sigmoid(layer_one)
    layer_two = layer(layer_one_sigmoid, weights_outer)
    layer_two_sigmoid = sigmoid(layer_two)

    return (layer_two_sigmoid, layer_two, layer_one)


input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected = np.array([0, 1, 1, 0])
weights_inner = np.array([[0, 1, 2], [0, 1, 2]])
weights_outer = np.array([0, 1, 2])
if RANDOM:
    weights_inner = np.array(
        [[random.random() for _ in range(3)], [random.random() for _ in range(3)]]
    )
    weights_outer = np.array([random.random() for _ in range(3)])


# first iteration
layer_two_sigmoid, layer_two, layer_one = propagate(
    input[0], weights_inner, weights_outer
)
err_prime_diff = error_prime(layer_two_sigmoid, expected[0])


gradient_outer_value, cashed_partial_value = gradient_outer(
    err_prime_diff,
    weights_outer,
    layer_one,
    sigmoid_prime(layer_two),
)
weights_outer_new = weights_outer - c * gradient_outer_value
gradient_inner_value = gradient_inner(
    cashed_partial_value,
    sigmoid_prime(layer_one),
    input[0],
)
weights_inner_new = weights_inner - c * gradient_inner_value


iter = 0
while (
    loop_condition(
        np.append(weights_inner, weights_outer),
        np.append(weights_inner_new, weights_outer_new),
    )
    and iter < max_iter
):
    iter += 1
    weights_inner = weights_inner_new
    weights_outer = weights_outer_new

    layer_two_sigmoid, layer_two, layer_one = propagate(
        input[iter % 4], weights_inner, weights_outer
    )
    err_prime_diff = error_prime(layer_two_sigmoid, expected[iter % 4])

    gradient_outer_value, cashed_partial_value = gradient_outer(
        err_prime_diff,
        weights_outer,
        layer_one,
        sigmoid_prime(layer_two),
    )
    weights_outer_new = weights_outer - c * gradient_outer_value

    gradient_inner_value = gradient_inner(
        cashed_partial_value,
        sigmoid_prime(layer_one),
        input[iter % 4],
    )
    weights_inner_new = weights_inner - c * gradient_inner_value


print("predictions:")
for i in input:
    output, _, _ = propagate(i, weights_inner, weights_outer)
    print(i, output)

print("error", error(expected, output))
