import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random
from PIL import Image


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


class Kmeans:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.clusters = None
        self.assignments = None
        self.m = None
        self.d = None

    def fit(self, X):
        """Train the K-Means algorithm on the data X"""
        self.m, self.d = X.shape
        self.centroids = self._init_centroids(X)
        self.assignments = [-1 for _ in range(self.m)]

        for i in range(self.max_iter):
            self.clusters = self._create_clusters(X)
            old_centroids = self.centroids
            self.centroids = self._calculate_centroids(X)
            if self._is_converged(old_centroids):
                break
        return self.assignments

    def _init_centroids(self, X):
        """Initialize the centroids as k random samples from X"""
        return [X[random.randint(0, self.m - 1)] for _ in range(self.k)]

    def _create_clusters(self, X):
        """Assign the samples to the closest centroids to create clusters"""
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(X):
            centroid_idx = self._closest_centroid(sample)
            self.assignments[idx] = centroid_idx
            clusters[centroid_idx].append(sample)
        return clusters

    def _closest_centroid(self, sample):
        """Return the index of the closest centroid to the sample"""
        distances = [euclidean_distance(sample, point) for point in self.centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _calculate_centroids(self, X):
        """Calculate the new centroids as the means of the samples in each cluster, if the cluster is empty,
        choose a random sample from X"""
        return [np.mean(cluster, axis=0) if len(cluster) != 0 else X[np.random.randint(self.m - 1)] for cluster in
                self.clusters]

    def _is_converged(self, old_centroids):
        """Check if the centroids have changed"""
        return np.array_equal(old_centroids, self.centroids)


class PDCDPmeans(Kmeans):
    def __init__(self, l, max_iter=100):
        super().__init__(k=None, max_iter=max_iter)
        self.l = l
        self.j_max = None
        self.d_max = None

    def fit(self, X):
        """Train the K-Means algorithm on the data X"""
        self.m, self.d = X.shape

        self.k = 1
        self.centroids = self._init_centroid(X)
        self.assignments = [0 for _ in range(self.m)]

        for i in range(self.max_iter):
            self.j_max, self.d_max = self.farthest_point(X)
            if not self.split_if_needed(X):
                break
            self.clusters = self._create_clusters(X)
            old_centroids = self.centroids
            self.centroids = self._calculate_centroids(X)
            if self._is_converged(old_centroids):
                break
        return self.assignments

    def _init_centroid(self, X):
        """Initialize the centroid as mean of samples from X"""
        return [np.mean(X, axis=0)]

    def farthest_point(self, X):
        """Return the index, distance of the farthest point from its corresponding centroid"""
        distances = [euclidean_distance(x, self.centroids[self.assignments[idx]]) for idx, x in enumerate(X)]
        farthest_idx = np.argmax(distances)
        return farthest_idx, distances[farthest_idx]

    def split_if_needed(self, X):
        """Split the cluster if the farthest point is farther than l"""
        if self.d_max > self.l:
            self.k += 1
            self.centroids.append(X[self.j_max])
            self.assignments[self.j_max] = self.k
            return True
        return False


def generate_data(n=1000, centers=None, random_state=None):
    X, y = make_blobs(n_samples=n, n_features=2, centers=centers, random_state=random_state)
    plt.scatter(X[:, 0], X[:, 1])
    plt.title(f"1000 sampled points in 2D, random state {random_state}")
    plt.savefig(f"images/data_{random_state}.png")
    plt.show()
    return X


def run_Kmeans(k: int, X: np.ndarray, task_initials: str, random_state=None):
    kmeans = Kmeans(k)
    labels = kmeans.fit(X)
    title = f"Kmeans with k = {k}"
    if random_state is not None:
        title += f", random state {random_state}"
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.colorbar()
    plt.title(title)
    plt.savefig(f"images/{task_initials}_k{k}.png")
    plt.show()


def run_pdc_dp_means(l: int, X: np.ndarray, task_initials: str, random_state=None):
    pdc_dp_means = PDCDPmeans(l)
    labels = pdc_dp_means.fit(X)
    title = f"PDC DP means with lambda = {l}"
    if random_state is not None:
        title += f", random state {random_state}"
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.colorbar()
    plt.title(title)
    plt.savefig(f"images/{task_initials}_l{l}.png")
    plt.show()


def diff_random_state():
    task_initials = "task2_drs_"
    for i in range(5):
        X = generate_data(random_state=i)
        run_Kmeans(3, X, task_initials + f"rs{i}_", i)
        run_pdc_dp_means(4, X, task_initials + f"rs{i}_", i)


def diff_k_values():
    task_initials = "task2_dkv_"
    X = generate_data(random_state=0)
    k_values = [1, 3, 200, 500, 1000]
    for k in k_values:
        run_Kmeans(k, X, task_initials)


def diff_l_values():
    task_initials = "task2_dlv_"
    X = generate_data(random_state=0)
    l_values = [10, 5, 4, 1, 0.5, 0.01]
    for l in l_values:
        run_pdc_dp_means(l, X, task_initials)


def monkey_clustering_kmeans(k, X, shape):
    # Apply K-means clustering to the RGB values
    kmeans = Kmeans(k)
    labels = kmeans.fit(X)

    # Replace each pixel's color with the value of the centroid of the cluster it was assigned to
    x_clustered = np.array([kmeans.centroids[label] for label in labels])
    x_clustered = x_clustered.reshape(shape)
    img_clustered = Image.fromarray(x_clustered.astype('uint8'), 'RGB')
    img_clustered.save(f"images/monkey_clustered_kmeans_k_{k}.png")
    img_clustered.show()


def monkey_clustering_pdc_dp_means(l, X, shape):
    # Apply PDC DP-means clustering to the RGB values
    pdc_dp_means = PDCDPmeans(l)
    labels = pdc_dp_means.fit(X)

    # Replace each pixel's color with the value of the centroid of the cluster it was assigned to
    x_clustered = np.array([pdc_dp_means.centroids[label] for label in labels])
    x_clustered = x_clustered.reshape(shape)
    img_clustered = Image.fromarray(x_clustered.astype('uint8'), 'RGB')
    img_clustered.save(f"images/monkey_clustered_pdc_dp_means_l_{l}.png")
    img_clustered.show()


def task2():
    diff_random_state()
    diff_k_values()
    diff_l_values()


def task3():
    img = Image.open("images/monkey.jpg")

    # Scale down the image by a factor input
    factor = 4
    width, height = img.size
    new_size = (width // factor, height // factor)
    img_compressed = img.resize(new_size, resample=Image.BILINEAR)
    img_compressed.save(f"images/monkey_compressed.png")

    # Convert the image to a numpy array
    X = np.array(img_compressed)

    # flatten the array
    flat_X = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))

    # CLuster with different values of k
    for k in [10, 50, 100]:
        monkey_clustering_kmeans(k, flat_X, X.shape)

    # Cluster with different values of l
    for l in [1]:
        monkey_clustering_pdc_dp_means(l, flat_X, X.shape)


def main():
    # task2()
    task3()


if __name__ == "__main__":
    main()
