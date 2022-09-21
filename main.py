from matplotlib import pyplot as plt
from sklearn import datasets as ds
import numpy as np

x, y = ds.make_blobs(n_samples=300, n_features=2,
                     centers=2, cluster_std=1.40,
                     random_state=2)


def step_func(z):
    return 1.0 if (z > 0) else 0.0


def perceptron(x, y, lr, epochs):
    m, n = x.shape
    t = np.zeros((n + 1, 1))
    n_miss_list = []

    # Training
    for epoch in range(epochs):
        n_miss = 0
        for idx, x_i in enumerate(x):
            x_i = np.insert(x_i, 0, 1).reshape(-1, 1)
            y_hat = step_func(np.dot(x_i.T, t))
            if (np.squeeze(y_hat) - y[idx]) != 0:
                t += lr * ((y[idx] - y_hat) * x_i)
                n_miss += 1
        n_miss_list.append(n_miss)
    return t, n_miss_list


def plot_graph(x, t):
    x1 = [min(x[:, 0]), max(x[:, 0])]
    m = -t[1] / t[2]
    c = -t[0] / t[2]
    x2 = m * x1 + c

    plt.figure(figsize=(14, 10))
    plt.plot(x[:, 0][y == 0], x[:, 1][y == 0], "bo")
    plt.plot(x[:, 0][y == 1], x[:, 1][y == 1], "ro")
    plt.plot(x1, x2, 'k-')
    plt.show()


t, miss_l = perceptron(x, y, 0.5, 150)
plot_graph(x, t)
