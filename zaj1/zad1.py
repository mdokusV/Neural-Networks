def step_function(x):
    if x >= 0:
        return 1
    else:
        return 0


def McCullough_pitts_and(x1, x2):
    weights = [2, 2, -3]
    result = x1 * weights[0] + x2 * weights[1] + weights[2]
    return step_function(result)


def McCullough_pitts_or(x1, x2):
    weights = [2, 2, -1]
    result = x1 * weights[0] + x2 * weights[1] + weights[2]
    return step_function(result)


def McCullough_pitts_not(x):
    weights = [-2, -1]
    result = x * weights[0] - weights[1]
    return step_function(result)


def McCullough_pitts_nand(x1, x2):
    weights = [-2, -2, 3]
    result = x1 * weights[0] + x2 * weights[1] + weights[2]
    return step_function(result)


def McCullough_pitts_nor(x1, x2):
    weights = [-2, -2, 1]
    result = x1 * weights[0] + x2 * weights[1] + weights[2]
    return step_function(result)


test_data_two_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
test_data_one_input = [[0], [1]]

print("\nAND:")
for i in test_data_two_input:
    print(i[0], i[1], McCullough_pitts_and(i[0], i[1]))

print("\nOR:")
for i in test_data_two_input:
    print(i[0], i[1], McCullough_pitts_or(i[0], i[1]))

print("\nNOT:")
for i in test_data_one_input:
    print(i[0], McCullough_pitts_not(i[0]))


print("\nNAND:")
for i in test_data_two_input:
    print(i[0], i[1], McCullough_pitts_nand(i[0], i[1]))

print("\nNOR:")
for i in test_data_two_input:
    print(i[0], i[1], McCullough_pitts_nor(i[0], i[1]))
