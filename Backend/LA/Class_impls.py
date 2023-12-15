import numpy as np


class My_KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, data):
        self.centroids = data[np.random.choice(
            len(data), self.n_clusters, replace=False)]

        for _ in range(self.max_iters):

            labels = self._assign_labels(data)

            new_centroids = self._update_centroids(data, labels)

            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        self.labels_ = labels

    def fit_predict(self, data):
        self.fit(data)
        return self.labels_

    def _assign_labels(self, data):
        distances = np.linalg.norm(
            data[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def _update_centroids(self, data, labels):
        new_centroids = np.array([data[labels == k].mean(axis=0)
                                 for k in range(self.n_clusters)])
        return new_centroids


class My_KMedoids:
    def __init__(self, n_clusters=5, max_iters=1000, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state

    def fit(self, data):
        np.random.seed(self.random_state)

        # Randomly initialize medoids
        self.medoid_indices_ = np.random.choice(
            len(data), self.n_clusters, replace=False)
        self.medoids_ = data[self.medoid_indices_]

        for _ in range(self.max_iters):
            # Assign each data point to the nearest medoid
            labels = self._assign_labels(data)

            # Update medoids by choosing the data point with the minimum total dissimilarity
            new_medoids_indices = self._update_medoids(data, labels)

            # Check for convergence
            if np.all(self.medoid_indices_ == new_medoids_indices):
                break

            self.medoid_indices_ = new_medoids_indices
            self.medoids_ = data[self.medoid_indices_]

        self.labels_ = labels

    def fit_predict(self, data):
        self.fit(data)
        return self.labels_

    def _assign_labels(self, data):
        distances = np.linalg.norm(
            data[:, np.newaxis] - self.medoids_, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def _update_medoids(self, data, labels):
        new_medoids_indices = np.array([
            np.argmin(np.sum(np.linalg.norm(
                data[labels == k] - data[i], axis=1)))
            for k, i in enumerate(self.medoid_indices_)
        ])
        return new_medoids_indices


class Node_Birch:
    def __init__(self, branching_factor, threshold):
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.children = []
        self.subcluster_centers = []
        self.n_samples = 0

    def add_sample(self, sample):
        self.subcluster_centers.append(sample)
        self.n_samples += 1

        if len(self.subcluster_centers) > self.branching_factor:
            # Split the node into subclusters if the threshold is exceeded
            self.split()

    def split(self):
        child = Node_Birch(self.branching_factor, self.threshold)
        child.subcluster_centers = self.subcluster_centers
        self.subcluster_centers = []
        self.children.append(child)

    def get_subclusters(self):
        subclusters = [self.subcluster_centers]

        for child in self.children:
            subclusters.extend(child.get_subclusters())

        return subclusters


class My_Birch:
    def __init__(self, threshold=0.5, branching_factor=50):
        self.root = Node_Birch(branching_factor, threshold)
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.labels_ = None

    def fit(self, data):
        for sample in data:
            self._insert_sample(self.root, sample)

        subclusters = self.root.get_subclusters()
        self.labels_ = self._assign_labels(data, subclusters)

    def _insert_sample(self, node, sample):
        if not node.children:
            node.add_sample(sample)
        else:
            distances = np.linalg.norm(
                np.array(node.subcluster_centers) - sample, axis=1)

            if len(distances) == 0 or distances.min() > self.threshold:
                node.add_sample(sample)
            else:
                min_distance_index = np.argmin(distances)
                self._insert_sample(node.children[min_distance_index], sample)

    def _assign_labels(self, data, subclusters):
        labels = np.zeros(len(data), dtype=int)

        for i, subcluster in enumerate(subclusters):
            for j in range(len(data)):
                if any(np.array_equal(data[j], sample) for sample in subcluster):
                    labels[j] = i

        return labels


class My_DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit_predict(self, data):
        # Initialize all points as noise (-1)
        self.labels_ = np.full(len(data), -1)
        cluster_id = 0

        for i in range(len(data)):
            if self.labels_[i] != -1:
                continue  # Skip points that are already assigned to a cluster

            neighbors = self._get_neighbors(data, i)

            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Mark point as noise
            else:
                self._expand_cluster(data, i, neighbors, cluster_id)
                cluster_id += 1

        return self.labels_

    def _get_neighbors(self, data, index):
        distances = np.linalg.norm(data - data[index], axis=1)
        neighbors = np.where(distances < self.eps)[0]
        return neighbors

    def _expand_cluster(self, data, core_index, neighbors, cluster_id):
        self.labels_[core_index] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_index = neighbors[i]

            if self.labels_[neighbor_index] == -1:
                self.labels_[neighbor_index] = cluster_id
                new_neighbors = self._get_neighbors(data, neighbor_index)

                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, new_neighbors])

            elif self.labels_[neighbor_index] == 0:
                self.labels_[neighbor_index] = cluster_id

            i += 1


def Enahnce(arr, min_change=0, max_change=2):
    # Get the shape of the array
    rows, cols = arr.shape

    # Generate random values for each element in the array within the specified range
    random_changes = np.random.uniform(
        min_change, max_change, size=(rows, cols))

    # Randomly decide whether to increase or decrease each element
    signs = np.random.choice([-1, 1], size=(rows, cols))

    # Apply the random changes to the array
    modified_array = arr + random_changes * signs

    return modified_array
