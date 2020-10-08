import matplotlib.pyplot as plt
import numpy as np


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def cluster(n, k, x, y, x_c, y_c):
    clust = []

    for i in range(n):
        min = dist(x[i], y[i], x_c[0], y_c[0])
        min_numb = 0

        for j in range(k):
            current_dist = dist(x[i], y[i], x_c[j], y_c[j])
            if (min > current_dist):
                min = current_dist
                min_numb = j

        clust.append(min_numb)

    return clust


def recalc_c(k, n, x, y, clust):
    new_c = []

    for i in range(k):
        new_c.append([])

    for i in range(k):
        sum_k_x, sum_k_y, count = 0, 0, 0

        for j in range(n):
            if clust[j] == i:
                sum_k_x += x[j]
                sum_k_y += y[j]
                count += 1

        new_c[i].append(sum_k_x / count)
        new_c[i].append(sum_k_y / count)

    return new_c


n, k = 100, 4

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

clust = cluster(n, k, x, y, x_c, y_c)
new_c = recalc_c(k, n, x, y, clust)

new_x_c, new_y_c = [], []

for i in range(k):
    new_x_c.append(new_c[i][0])
    new_y_c.append(new_c[i][1])

result_x_c = []
result_y_c = []

while (True):
    clust = cluster(n, k, x, y, new_x_c, new_y_c)

    if (x_c != new_x_c) | (y_c != new_y_c):
        x_c = new_x_c
        y_c = new_y_c

        c = recalc_c(k, n, x, y, clust)
        new_x_c, new_y_c = [], []

        for i in range(k):
            new_x_c.append(c[i][0])
            new_y_c.append(c[i][1])
    else:
        break

plt.scatter(x, y)
plt.scatter(new_x_c, new_y_c, color='r')
plt.show()


