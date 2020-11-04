import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    class C_Means():
        def __init__(self, dataset, n_clusters=3, fuzzy_c=2, cut_param=.9):
            self.dataset = dataset
            self.n_clusters = n_clusters
            self.fuzzy_c = fuzzy_c
            self.cut_param = cut_param
            self.max_iter_num = 100
            self.tolerance = .01
            self.dist = np.zeros((self.dataset.shape[0], self.n_clusters))
            self.centroids = np.array(
                [[np.random.uniform(0, 640), np.random.uniform(0, 480)] for i in range(self.n_clusters)])
            self.u = np.array(
                [[np.random.uniform(0, 1) for i in range(self.n_clusters)] for j in range(self.dataset.shape[0])])
            self.labels = np.array([])

        def get_dist2(self, list1, list2):
            return sum((i - j) ** 2 for i, j in zip(list1, list2))

        def distribute_data(self):
            self.dist = np.array([[self.get_dist2(i, j) for i in self.centroids] for j in self.dataset])
            self.u = (1 / self.dist) ** (1 / (self.fuzzy_c - 1))
            self.normalize_arr()
            self.u = (self.u / self.u.sum(axis=1)[:, None])

        def normalize_arr(self):
            arr_with_inf = np.where(np.isinf(self.u))
            lines = arr_with_inf[0]
            rows = arr_with_inf[1]
            for i in range(0, len(lines)):
                self.u[lines[i]] = 0
                self.u[lines[i]][rows[i]] = 1

        def recalculate_centroids(self):
            self.centroids = (self.u.T).dot(self.dataset) / self.u.sum(axis=0)[:, None]

        def fit(self):
            iter = 1
            while iter < self.max_iter_num:
                prev_centroids = np.copy(self.centroids)
                # рассчитываем центры кластеров
                self.recalculate_centroids()
                self.distribute_data()
                if max([self.get_dist2(i, k) for i, k in zip(self.centroids, prev_centroids)]) < self.tolerance:
                    break
                iter += 1

        def getLabels(self):
            labels = np.array([])
            for i in range(len(self.u)):
                i_max = self.u[i][0]
                i_max_indx = 0
                for j in range(len(self.u[i])):
                    if (self.u[i][j] > i_max):
                        i_max = self.u[i][j]
                        i_max_indx = j
                if (i_max > self.cut_param):
                    labels = np.append(list(labels), i_max_indx + 1).astype(int)
                else:
                    labels = np.append(list(labels), 0).astype(int)
            return labels


    dataset = np.array([[np.random.uniform(0, 20), np.random.uniform(0, 20)] for k in range(10)])

    test = C_Means(dataset, 3, 2, .5)
    test.fit()
    colors = np.array(['black', 'green', 'blue', 'red'])
    pred = test.getLabels();

    plt.figure(figsize=(4, 4))
    plt.scatter(dataset[:, 0], dataset[:, 1], color=colors[pred])
    plt.show()
