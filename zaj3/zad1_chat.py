import sympy as sp
import random


def gradient_descent(F, variables, initial_point, c, epsilon, max_iter=1000):
    n = len(variables)
    x_old = initial_point

    # Define gradients of the function with respect to variables
    gradients = [sp.diff(F, var) for var in variables]

    for _ in range(max_iter):
        # Calculate new points
        x_new = [
            x_old[i] - c * gradients[i].subs({variables[j]: x_old[j] for j in range(n)})
            for i in range(n)
        ]

        # Check convergence
        diff = [abs(x_new[i] - x_old[i]) for i in range(n)]

        if all(diff[i] < epsilon for i in range(n)):
            break

        # Update old points
        x_old = x_new

    return x_new, F.subs({variables[i]: x_new[i] for i in range(n)})


# Function F1
x1, x2, x3 = sp.symbols("x1 x2 x3")
F1 = 2.4 * x1**2 + 2 * x2**2 + x3**2 - 2 * x1 * x2 - 2 * x2 * x3 - 2 * x1 + 3
F1 = x1 ** (1 / 3)
variables_F1 = [x1, x2, x3]
initial_point_F1_a = [random.uniform(-1, 1) for _ in range(3)]
initial_point_F1_b = [random.uniform(-1, 1) for _ in range(3)]

# Function F2
x1, x2 = sp.symbols("x1 x2")
F2 = 3 * x1**4 + 4 * x1**3 - 12 * x1**2 + 12 * x2**2 - 24 * x2
variables_F2 = [x1, x2]
initial_point_F2_a = [random.uniform(-1, 1) for _ in range(2)]
initial_point_F2_b = [random.uniform(-1, 1) for _ in range(2)]

# Parameters
c = 0.01
epsilon = 1e-5

# Applying gradient descent
print("Function F1:")
print("Starting point (i-a):", initial_point_F1_a)
result_a_F1 = gradient_descent(F1, variables_F1, initial_point_F1_a, c, epsilon)
print("Minimum point (i-a):", result_a_F1[0])
print("Minimum value (i-a):", result_a_F1[1])
print()
print("Starting point (i-b):", initial_point_F1_b)
result_b_F1 = gradient_descent(F1, variables_F1, initial_point_F1_b, c, epsilon)
print("Minimum point (i-b):", result_b_F1[0])
print("Minimum value (i-b):", result_b_F1[1])
print()

print("Function F2:")
print("Starting point (i-a):", initial_point_F2_a)
result_a_F2 = gradient_descent(F2, variables_F2, initial_point_F2_a, c, epsilon)
print("Minimum point (i-a):", result_a_F2[0])
print("Minimum value (i-a):", result_a_F2[1])
print()
print("Starting point (i-b):", initial_point_F2_b)
result_b_F2 = gradient_descent(F2, variables_F2, initial_point_F2_b, c, epsilon)
print("Minimum point (i-b):", result_b_F2[0])
print("Minimum value (i-b):", result_b_F2[1])
