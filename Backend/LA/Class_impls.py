from concurrent.futures import ThreadPoolExecutor
import math
from urllib.parse import urljoin
# from more_itertools import partition
import numpy as np
from bs4 import BeautifulSoup
import networkx as nx
import requests
from collections import deque


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


# Assignment 8


# DFS Crawler
def get_links(session, base_url, url):
    try:
        absolute_url = urljoin(base_url, url)
        response = session.get(absolute_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [urljoin(base_url, a.get('href'))
                 for a in soup.find_all('a', href=True)]
        return links
    except Exception as e:
        print(f"Error fetching links from {url}: {e}")
        return []


def save_links_to_file(links, filename):
    with open(filename, 'w') as file:
        for link in links:
            file.write(link + '\n')

# Hits Algorithm


def calculate_hits(graph):
    hits_scores = nx.hits(graph)
    sorted_authorities = sorted(
        hits_scores[1].items(), key=lambda x: x[1], reverse=True)[:10]
    sorted_hubs = sorted(hits_scores[0].items(
    ), key=lambda x: x[1], reverse=True)[:10]

    return graph, sorted_authorities, sorted_hubs


def dfs_crawler(seed_url, max_depth, output_file):
    visited = set()

    def dfs(current_url, depth, session):
        if depth > max_depth:
            return
        print(f"{'  ' * depth}{current_url}")
        visited.add(current_url)

        links = get_links(session, seed_url, current_url)
        save_links_to_file(
            [f"{seed_url} {link}" for link in links], output_file)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(dfs, link, depth + 1, session)
                       for link in links if link not in visited]

            for future in futures:
                future.result()

    with requests.Session() as session:
        dfs(seed_url, 0, session)


def bfs_crawler(seed_url, max_depth, output_file):
    visited = set()
    queue = deque([(seed_url, 0)])

    def worker(current_url, depth, session):
        nonlocal visited
        if depth > max_depth:
            return
        print(f"{'  ' * depth}{current_url}")
        visited.add(current_url)

        links = get_links(session, seed_url, current_url)
        save_links_to_file(
            [f"{seed_url} {link}" for link in links], output_file)

        for link in links:
            if link not in visited:
                queue.append((link, depth + 1))

    with requests.Session() as session:
        while queue:
            current_url, depth = queue.popleft()
            if current_url not in visited:
                worker(current_url, depth, session)

# PageRank


def read_links_from_file(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file.readlines()]


def calculate_pagerank(links):
    graph = nx.DiGraph()

    for link in links:
        parts = link.split()
        if len(parts) == 2:
            source, target = parts
            graph.add_edge(source, target)

    pagerank_scores = nx.pagerank(graph)
    sorted_pages = sorted(pagerank_scores.items(),
                          key=lambda x: x[1], reverse=True)[:10]

    return graph, sorted_pages

# Assignment 4
def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def max_label(dict):
    max_count = 0
    label = ""

    for key, value in dict.items():
        if dict[key] > max_count:
            max_count = dict[key]
            label = key

    return label


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:

    def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.header[self.column], condition, str(self.value))


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def entropy(rows):
    entries = class_counts(rows)
    avg_entropy = 0
    size = float(len(rows))
    for label in entries:
        prob = entries[label] / size
        avg_entropy = avg_entropy + (prob * math.log(prob, 2))
    return -1*avg_entropy


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)


def find_best_split(rows, header):
    best_gain = 0
    best_question = None
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])

        for val in values:
            question = Question(col, val, header)

            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:

    def __init__(self, rows, id, depth):
        self.predictions = class_counts(rows)
        self.predicted_label = max_label(self.predictions)
        self.id = id
        self.depth = depth


class Decision_Node:

    def __init__(self, question, true_branch, false_branch, depth, id, rows):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth
        self.id = id
        self.rows = rows


def build_tree(rows, header, depth=0, id=0):
    gain, question = find_best_split(rows, header)

    if gain == 0:
        return Leaf(rows, id, depth)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows, header, depth + 1, 2 * id + 2)
    false_branch = build_tree(false_rows, header, depth + 1, 2 * id + 1)

    return Decision_Node(question, true_branch, false_branch, depth, id, rows)


def prune_tree(node, prunedList):
    if isinstance(node, Leaf):
        return node

    if int(node.id) in prunedList:
        return Leaf(node.rows, node.id, node.depth)

    node.true_branch = prune_tree(node.true_branch, prunedList)
    node.false_branch = prune_tree(node.false_branch, prunedList)

    return node


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predicted_label

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Leaf id: " + str(node.id) + " Predictions: " +
              str(node.predictions) + " Label Class: " + str(node.predicted_label))
        return

    print(spacing + str(node.question) + " id: " +
          str(node.id) + " depth: " + str(node.depth))

    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


def getLeafNodes(node, leafNodes=[]):
    if isinstance(node, Leaf):
        leafNodes.append(node)
        return

    getLeafNodes(node.true_branch, leafNodes)
    getLeafNodes(node.false_branch, leafNodes)

    return leafNodes


def getInnerNodes(node, innerNodes=[]):
    if isinstance(node, Leaf):
        return

    innerNodes.append(node)

    getInnerNodes(node.true_branch, innerNodes)
    getInnerNodes(node.false_branch, innerNodes)

    return innerNodes


def computeAccuracy(rows, node):
    count = len(rows)
    if count == 0:
        return 0

    accuracy = 0
    for row in rows:
        if row[-1] == classify(row, node):
            accuracy += 1
    return round(accuracy/count, 2)


def comp_confmat(actual, predicted):
    classes = np.unique(actual)
    matrix = np.zeros((len(classes), len(classes)))

    for i in range(len(classes)):
        for j in range(len(classes)):
            matrix[i, j] = np.sum((actual == classes[i])
                                  & (predicted == classes[j]))

    return matrix
