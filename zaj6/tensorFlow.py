import random
import tensorflow as tf
import numpy as np

np.random.seed(50)


def init():
    X = tf.Variable(np.random.uniform(-10, 10), trainable=True)
    Y = tf.Variable(np.random.uniform(-10, 10), trainable=True)
    return X, Y


def function(X, Y):
    return 3 * X**4 + 4 * X**3 - 12 * X**2 + 12 * Y**2 - 24 * Y


X, Y = init()
optimizer = tf.optimizers.SGD(learning_rate=0.01, momentum=0)
for epoch in range(1000):
    optimizer.minimize(lambda: function(X, Y), var_list=[X, Y])

print(X.numpy(), Y.numpy(), function(X, Y).numpy())
