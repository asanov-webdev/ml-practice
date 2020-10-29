import matplotlib.pyplot as plt
import numpy as np
from enum import Enum


class Flag(Enum):
    GREEN = 0
    YELLOW = 1
    RED = 2
    UNKNOWN = -1


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


n = 100
x = np.random.randint(1, 100, n)
y = np.random.randint(1, 100, n)
eps, minPts = 10, 3
flags = []
for i in range(0, n):
    neighb = -1
    for j in range(0, n):
        if dist(x[i], y[i], x[j], y[j]) < eps:
            neighb += 1
    if neighb >= minPts:
        flags.append(Flag.GREEN.value)
    else:
        flags.append(Flag.UNKNOWN.value)
for i in range(0, n):
    if flags[i] == Flag.UNKNOWN.value:
        for j in range(0, n):
            if flags[j] == Flag.GREEN.value and dist(x[i], y[i], x[j], y[j]) < eps:
                flags[i] = Flag.YELLOW.value
                break
    if flags[i] == Flag.UNKNOWN.value:
        flags[i] = Flag.RED.value
clusters = np.zeros(n)
cl = 1
for i in range(0, n):
    if flags[i] == Flag.GREEN.value:
        if clusters[i] == 0:
            clusters[i] = cl
            cl += 1
        for j in range(0, n):
            if dist(x[i], y[i], x[j], y[j]) < eps:
                clusters[j] = clusters[i]
for i in range(0, n):
    clr = (clusters[i] + 1) / cl
    plt.scatter(x[i], y[i], color=(clr, 0.2, clr ** 2))
plt.show()
