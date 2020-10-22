import numpy as np


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


n, k, m, epsilon = 100, 4, 2, 1

x = np.random.randint(1, 100, n)
y = np.random.randint(1, 100, n)

x_cc = np.mean(x)
y_cc = np.mean(y)

r = []

for i in range(0, n):
    r.append(dist(x[i], y[i], x_cc, y_cc))

R = max(r)

x_c, y_c = [], []

for i in range(k):
    x_c.append(R * np.cos(2 * np.pi * i / k) + x_cc)
    y_c.append(R * np.sin(2 * np.pi * i / k) + y_cc)

u_matrix = []

for i in range(n):
    u_matrix.append([])

for i in range(n):
    for j in range(k):
# pending...
