from tracemalloc import stop
import numpy as np


c = 0.1
epsilon = 1
beta = 1.0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-beta * z))


def sigmoid_prime(z):
    return beta * sigmoid(z) * (1 - sigmoid(z))


def layer(x: np.ndarray, w: np.ndarray):
    np.append(x, 1)
    return sigmoid(np.dot(x, w))


def gradient_outer(
    error_prime: np.float32,
    weights: np.ndarray,
    values: np.ndarray,
    sigmoid_prime_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    err_sig = error_prime * sigmoid_prime_values
    return (np.dot(err_sig, values) + epsilon / 10, np.dot(err_sig, weights))


def gradient_inner(
    prev_partial_values: np.ndarray,
    sigmoid_prime_value: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    diag_val = np.diag(prev_partial_values)
    diag_sig = np.diag(sigmoid_prime_value)
    scalar_matrix = np.multiply(diag_val, diag_sig)
    return np.dot(scalar_matrix, values) + epsilon / 10


def error(real: np.ndarray, predicted: np.ndarray):
    return np.sum(np.square(real - predicted))


def error_prime(expected: np.ndarray, predicted: np.ndarray):
    return 2 * (predicted - expected)


def loop_condition(previous_weight: np.ndarray, current_weight: np.ndarray):
    return np.all(np.abs(previous_weight - current_weight) > epsilon)


input = np.array([[0, 0], [0, 1], [1, 0], [0, 1]])
expected = np.array([0, 1, 1, 0])


weights_inner = np.array([[0, 1, 2], [0, 1, 2]])
weights_outer = np.array([0, 1, 2])

weights_inner_new = weights_inner + c * gradient_inner(
    
)

# np.append(weights_inner, weights_outer)
# print(
#     stop_condition(
#         np.append(weights_inner, weights_outer),
#         np.append(weights_inner_new, weights_outer_new),
#     )
# )

for
