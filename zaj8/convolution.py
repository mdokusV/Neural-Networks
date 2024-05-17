from pprint import pprint
from numpy import array, ndarray
import numpy as np

THRESHOLD = 2.5


def activation_function(x: float) -> bool:
    return x >= THRESHOLD


input: list[ndarray] = []
weights: list[ndarray] = []
# region input
# input 0
input.append(
    array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
)

# input 1
input.append(
    array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ]
    )
)

# input 2
input.append(
    array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    )
)

# input 3
input.append(
    array(
        [
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
)

# input 4
input.append(
    array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ]
    )
)
# endregion

# region weights

# weights 0
weights.append(
    array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )
)
# weights 1
weights.append(
    array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]
    )
)
# weights 2
weights.append(
    array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]
    )
)
# weights 3
weights.append(
    array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]
    )
)
# endregion


def convolution(input: ndarray, weights: ndarray) -> ndarray:
    result: ndarray = array(
        [
            [
                (
                    sum(
                        (
                            activation_function(
                                weights[i][j]
                                * input[i_matrix + i - 1][j_matrix + j - 1]
                            )
                            if 0 <= i_matrix + i - 1 < len(input[0])
                            and 0 <= j_matrix + j - 1 < len(input)
                            else 0
                        )
                        for i in range(len(weights[0]))
                        for j in range(len(weights))
                    )
                )
                for i_matrix in range(len(input[0]))
            ]
            for j_matrix in range(len(input))
        ]
    )

    return result


result = convolution(input[0], weights[0])
print(result)
