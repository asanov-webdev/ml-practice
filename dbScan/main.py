import numpy as np
import matplotlib.pyplot as plt
import colorsys
from scipy.spatial import distance
from sklearn.datasets.samples_generator import make_blobs

points, classes = make_blobs(n_samples=50,
                             centers=[[-10, -10], [-10, 10], [10, -10], [10, 10]],
                             cluster_std=0.65,
                             shuffle=True)
plt.scatter(points[:, 0], points[:, 1])
plt.title('Data')
plt.show()


def range_query(points, p_idx, eps):
    neighbors = list()

    for x in range(points.shape[0]):
        if ((distance.euclidean(points[p_idx], points[x]) <= eps) and (x != p_idx)):
            neighbors.append(x)

    return np.array(neighbors)


def mark_points(points, min_pts=2, eps=0.5):
    labels = np.full(points.shape[0], 0, dtype=int)

    for p_idx in range(points.shape[0]):
        cur_neighbors = range_query(points, p_idx, eps)

        if cur_neighbors.shape[0] >= min_pts:
            labels[p_idx] = 1
        elif min_pts > cur_neighbors.shape[0] > 0:
            labels[p_idx] = 2
        else:
            continue

    return labels


def plot_clusters(points, classes, class_num):
    HSV_tuples = [(x * 1.0 / class_num, 0.5, 0.5) for x in range(class_num)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    for p_idx in range(points.shape[0]):
        plt.scatter(points[p_idx, 0], points[p_idx, 1], color=RGB_tuples[classes[p_idx]])
    plt.show()


labels = mark_points(points, eps=1)

plot_clusters(points, labels, 3)


def mark_neighbors(clusters, points, root_idx, neighbors, eps):
    neighbors = np.setdiff1d(neighbors, root_idx)

    for neighbor_idx in neighbors:
        child_neighbors = range_query(points, neighbor_idx, eps)

        if child_neighbors.shape[0] > 0:
            clusters[child_neighbors] = clusters[neighbor_idx]
            root_idx = np.append(root_idx, neighbor_idx)

            mark_neighbors(clusters, points, root_idx, child_neighbors, eps)


def clusterize(points, labels, eps=0.5):
    cl_num = -1
    clusters = np.full(points.shape[0], -1)

    for p_idx in range(points.shape[0]):
        if clusters[p_idx] == -1:
            cl_num = cl_num + 1
            clusters[p_idx] = cl_num
        else:
            continue

        neighbors = range_query(points, p_idx, eps)

        if neighbors.shape[0] > 0:
            clusters[neighbors] = clusters[p_idx]

            root_idx = np.array([p_idx])
            mark_neighbors(clusters, points, p_idx, neighbors, eps)

    return clusters, cl_num + 1


clusters, cl_num = clusterize(points, labels, eps=1)
plot_clusters(points, clusters, cl_num)
