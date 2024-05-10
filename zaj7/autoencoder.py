from enum import Enum
from os import sep
import random

import numpy as np
from scipy.special import expit


class VisualEnum(Enum):
    FALSE = 0
    TRUE = 1
    BOTH = 2


RANDOM = True
VISUAL = VisualEnum.BOTH
c = 1.3
epsilon = 1e-5
beta = 0.5
max_iter = 1e6
BATCH = 2

ZAD = 2

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
        np.multiply(err_sig.reshape(-1, 1), [values.tolist() + [1]] * weights.shape[0]),
        np.sum(np.multiply(err_sig.reshape(-1, 1), weights[:, :-1]), axis=0),
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


def show_array(list, len_vertical, len_horizontal):
    for i in range(len_vertical):
        to_show = ""
        for j in range(len_horizontal):
            match VISUAL:
                case VisualEnum.FALSE:
                    print("x" if list[i * len_horizontal + j] > 0.5 else "o", end="\t")
                case VisualEnum.TRUE:
                    print(np.round(list[i * len_horizontal + j], 4), end="\t")
                case VisualEnum.BOTH:
                    print(np.round(list[i * len_horizontal + j], 4), end="\t")
                    to_show += (
                        f"{'x' if list[i * len_horizontal + j] > 0.5 else 'o'}" + "\t"
                    )
        print(to_show)
        print()
    print()


def show_tests():
    for test in input:
        layer_two_sigmoid, _, layer_one_sigmoid, _ = propagate(
            test, weights_inner, weights_outer
        )
        print("Input:")
        show_array(test, 3, 3)

        print("Inner layer:")
        show_array(layer_one_sigmoid, 1, 2)

        print("Output layer:")
        show_array(layer_two_sigmoid, 3, 3)


input_one = np.array(
    [
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
)
input_two = np.array(
    [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]
)

input = np.array([input_one.flatten(), input_two.flatten()])
inner_node_number = 2
expected = input.copy()

if ZAD == 1:
    weights_inner_array = [
        [0 for _ in range(len(input[0]) + 1)] for _ in range(inner_node_number)
    ]
    weights_inner_array[0][0] = 20
    weights_inner_array[0][9] = -10
    weights_inner_array[1][2] = 20
    weights_inner_array[1][9] = -10

    weights_outer_array = [
        [20 * input[i][j] if i in [0, 1] else -10 for i in range(inner_node_number + 1)]
        for j in range(len(expected[0]))
    ]

    weights_inner = np.array(weights_inner_array)
    weights_outer = np.array(weights_outer_array)

    show_tests()


if ZAD == 2:
    weights_inner_array = [
        [0 for _ in range(len(input[0]) + 1)] for _ in range(inner_node_number)
    ]
    weights_inner_array[0][0] = 20
    weights_inner_array[0][9] = -10
    weights_inner_array[1][2] = 20
    weights_inner_array[1][9] = -10

    weights_outer_array = [
        [20 * input[i][j] if i in [0, 1] else -10 for i in range(inner_node_number + 1)]
        for j in range(len(expected[0]))
    ]

    if RANDOM:
        weights_inner_array = [
            [random.uniform(-10, 10) for _ in range(len(input[0]) + 1)]
            for _ in range(inner_node_number)
        ]

        weights_outer_array = [
            [random.uniform(-10, 10) for _ in range(inner_node_number + 1)]
            for _ in range(len(expected[0]))
        ]

    weights_inner = np.array(weights_inner_array)
    weights_outer = np.array(weights_outer_array)

    # first iteration
    layer_two_sigmoid, layer_two, layer_one_sigmoid, layer_one = propagate(
        input[0], weights_inner, weights_outer
    )
    err_prime_diff = error_prime(layer_two_sigmoid, expected[0])

    gradient_outer_value, cashed_partial_value = gradient_outer(
        err_prime_diff,
        weights_outer,
        layer_one_sigmoid,
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
    loop_con_val = loop_condition_value(
        weights_inner,
        weights_outer,
        weights_inner_new,
        weights_outer_new,
    )

    while loop_con_val > epsilon and iter < max_iter:
        gradient_outer_value_batch = np.zeros_like(gradient_outer_value)
        gradient_inner_value_batch = np.zeros_like(gradient_inner_value)
        for i in range(BATCH):
            iter += 1

            weights_inner = weights_inner_new
            weights_outer = weights_outer_new

            layer_two_sigmoid, layer_two, layer_one_sigmoid, layer_one = propagate(
                input[iter % len(input)], weights_inner, weights_outer
            )
            err_prime_diff = error_prime(layer_two_sigmoid, expected[iter % len(input)])

            gradient_outer_value, cashed_partial_value = gradient_outer(
                err_prime_diff,
                weights_outer,
                layer_one_sigmoid,
                sigmoid_prime(layer_two),
            )
            gradient_inner_value = gradient_inner(
                cashed_partial_value,
                sigmoid_prime(layer_one),
                input[iter % len(input)],
            )

            gradient_outer_value_batch += gradient_outer_value
            gradient_inner_value_batch += gradient_inner_value

        weights_outer_new = weights_outer - c * gradient_outer_value_batch

        weights_inner_new = weights_inner - c * gradient_inner_value_batch
        loop_con_val = loop_condition_value(
            weights_inner,
            weights_outer,
            weights_inner_new,
            weights_outer_new,
        )

        if iter % 10000 == 0:
            print(f"iter: {iter}, loop_con_val: {round(loop_con_val, 8)}")

    show_tests()
