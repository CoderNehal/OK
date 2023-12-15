import base64
from io import BytesIO
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from matplotlib import pyplot as plt
import pandas as pd
import requests
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, Birch, KMeans
from sklearn.metrics import silhouette_score
from .models import Data
import numpy as np
from .Class_impls import My_KMeans, My_KMedoids, My_DBSCAN, My_Birch, Enahnce
from .utils import data1
import matplotlib.pyplot as plt
import numpy as np
from sklearn_extra.cluster import KMedoids
from mlxtend.frequent_patterns import apriori
from .temp import str
import networkx as nx
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from urllib.parse import urljoin
from bs4 import BeautifulSoup


X = Enahnce(np.array(data1))
data  = X
length = len(data)
num_clusters = 5
threshold=0.5
eps=0.5
min_samples=5

@csrf_exempt
def hello_world(request):
    return HttpResponse("Hello World!")


@csrf_exempt
def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']

        # Save the file to the database
        new_data = Data(file=uploaded_file)
        new_data.save()

        return JsonResponse({'message': 'File uploaded successfully'})
    else:
        return JsonResponse({'error': 'No file provided'}, status=400)


def get_data():
    try:
        first_data = Data.objects.first()

        if first_data:
            # Modify this response based on your data model
            # d = first_data
            X = Enahnce(np.array(data))
            # print(X)
            return {
                'X': X,
                'length': len(X) + 1
            }
        else:
            print('No data available')
            # return JsonResponse({'error': 'No data available'}, status=404)
    except Exception as e:
        print("error:", str(e))
        # return JsonResponse({'error': str(e)}, status=500)


def all6(request):
    # file1 = Files.objects.all()
    # d = file1[1]

    X = Enahnce(np.array(data))
    length = len(data)
    labels = range(1, length+1)
    plt.figure(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(X[:, 0], X[:, 1], label='True Position')
    print("Mahiya ")
    for label, x, y in zip(labels, X[:, 0], X[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-3, 3),
            textcoords='offset points', ha='right', va='bottom')

    print("There")
    # dendrogram_img = plot_dendrogram(X, length)
    # KMeans_img = plot_KMeans(X, 5)
    # KMedoids = plot_KMedoids(X, 3)
    # Birch = plot_BIRCH(X, 0.5)
    # DBSCAN = plot_DBSCAN(X, 0.5, 5)
    # TabularAccuracy = tabulate_cluster_accuracy(X)

    print("Here")

    return {
        'X': X,
        'length': length
    }

def plot_dendrogram(request):


    linked = linkage(X, 'single')
    labelList = range(1, length+1)

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               labels=labelList,
               distance_sort='descending',
               show_leaf_counts=True)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return JsonResponse({
        'result': base64.b64encode(img.getvalue()).decode()
    })
    



def plot_KMeans(request):

    num_clusters = 3
    kmeans = My_KMeans(n_clusters=num_clusters)
    kmeans.fit(data)

    # Getting the centroids and labels based on KMeans clustering
    centroids = kmeans.centroids
    labels = kmeans.labels_

    plt.figure(figsize=(10, 7))

    # Plotting the data points with different colors for each cluster
    for i in range(num_clusters):
        plt.scatter(data[labels == i, 0],
                    data[labels == i, 1], label=f'Cluster {i+1}')

    # Plotting the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='X', s=200, color='red', label='Centroids')

    plt.title('KMeans Clustering')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Saving the plot to an image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return JsonResponse({
        'result': base64.b64encode(img.getvalue()).decode()
    })


# k-Medoids (PAM)


def plot_KMedoids(request):
    kmedoids = KMedoids(num_clusters, random_state=42)
    kmedoids.fit(data)

    # Getting the medoids and labels based on k-Medoids clustering
    medoids = kmedoids.medoid_indices_
    labels = kmedoids.labels_

    plt.figure(figsize=(12, 9))

    # Plotting the data points with different colors for each cluster
    for i in range(num_clusters):
        plt.scatter(data[labels == i, 0],
                    data[labels == i, 1], label=f'Cluster {i+1}')

    # Plotting the medoids
    plt.scatter(data[medoids, 0], data[medoids, 1],
                marker='X', s=200, color='red', label='Medoids')

    # Adding labels for each data point
    for j, (x, y) in enumerate(data):
        plt.text(x, y, f'{j+1}', fontsize=8,
                 ha='center', va='center', color='black')

    # Adding lines connecting data points to their medoids
    for i, medoid in enumerate(medoids):
        plt.plot([data[medoid, 0]] * len(data[labels == i, 0]),
                 data[labels == i, 1], linestyle='--', color='gray')

    plt.title('k-Medoids Clustering')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Saving the plot to an image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return JsonResponse({
        'result': base64.b64encode(img.getvalue()).decode()
    })


def plot_BIRCH(request):
    birch = Birch(threshold=threshold, n_clusters=None)
    birch.fit(data)

    # Getting the labels based on BIRCH clustering
    labels = birch.labels_

    # Creating a colormap for better visualization
    cmap = plt.cm.get_cmap('tab10', len(np.unique(labels)))

    plt.figure(figsize=(10, 7))

    # Plotting the data points with different colors for each cluster
    for label in np.unique(labels):
        plt.scatter(data[labels == label, 0], data[labels ==
                    label, 1], label=f'Cluster {label}', cmap=cmap)

    plt.title('BIRCH Clustering')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Adding colorbar for better understanding of cluster labels
    norm = plt.Normalize(labels.min(), labels.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # dummy empty array to feed the color data to the ScalarMappable
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=np.unique(labels), label='Cluster Labels')

    # Saving the plot to an image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return JsonResponse({
        'result': base64.b64encode(img.getvalue()).decode()
    })


def plot_DBSCAN(request):
    dbscan = My_DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)

    # Plotting the data points with different colors and markers for each cluster
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - 1 if - \
        1 in unique_labels else len(unique_labels)

    plt.figure(figsize=(10, 7))

    for label in unique_labels:
        if label == -1:
            plt.scatter(data[labels == label, 0], data[labels == label, 1],
                        color='gray', marker='x', s=50, label='Noise')
        else:
            plt.scatter(data[labels == label, 0], data[labels ==
                        label, 1], label=f'Cluster {label}', alpha=0.7)

    plt.title('DBSCAN Clustering')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Adding a legend with more details
    legend = plt.legend()
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]

    # Adding a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.5)

    # Saving the plot to an image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return JsonResponse({
        'result': base64.b64encode(img.getvalue()).decode()
    })


    
def preprocess_data():
    
    transactions = str
    
    return transactions

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
@csrf_exempt
def association_rule(request):
    
    # Preprocess data
    transactions = preprocess_data()
    # print("11111")
    # Use Apriori algorithm
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
    # print(frequent_itemsets,'items')
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    print(rules,'rules it is')
    # Display the rules
    result = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    # print()
    json_data = result.to_json(orient='records')
    # print(json_data)
    # Return JSON response
    return JsonResponse(json_data, safe=False)


@csrf_exempt
def crawlers(request):
    # Your input parameters (replace with actual values or receive them through request)
    seed_url = "https://gdscwce.vercel.app/about"
    max_depth = 1
    output_file = "output.txt"


    print("\nDFS Crawler......")
    dfs_crawler(seed_url, max_depth, output_file)

    print("\nBFS Crawler......")
    bfs_crawler(seed_url, max_depth, output_file)

    # PageRank
    links = read_links_from_file(output_file)
    graph, ranked_pages = calculate_pagerank(links)

    # Hits Algorithm
    graph, authorities, hubs = calculate_hits(graph)

    # Adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(graph).todense()

    # Convert the adjacency matrix to an image
    img_adjacency = BytesIO()
    plt.figure(figsize=(8, 6))
    plt.imshow(adjacency_matrix, cmap='Blues', interpolation='nearest')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(img_adjacency, format='png')
    img_adjacency.seek(0)
    img_base64_adjacency = base64.b64encode(img_adjacency.getvalue()).decode()

    # Clear the plot for the next use
    plt.clf()

    # Ranked pages table
    headers = ["Page", "Pagerank Score"]
    table_data = [(page, score) for page, score in ranked_pages]

    # Convert the ranked pages table to an image
    img_table = BytesIO()
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    plt.savefig(img_table, format='png')
    img_table.seek(0)
    img_base64_table = base64.b64encode(img_table.getvalue()).decode()

    # Clear the plot for the next use
    plt.clf()

    # Authorities table
    headers = ["Page", "Authority Score"]
    table_data_authorities = [(page, score) for page, score in authorities]

    # Convert the authorities table to an image
    img_authorities = BytesIO()
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.table(cellText=table_data_authorities, colLabels=headers, cellLoc='center', loc='center')
    plt.savefig(img_authorities, format='png')
    img_authorities.seek(0)
    img_base64_authorities = base64.b64encode(img_authorities.getvalue()).decode()

    # Clear the plot for the next use
    plt.clf()

    # Hubs table
    headers = ["Page", "Hub Score"]
    table_data_hubs = [(page, score) for page, score in hubs]

    # Convert the hubs table to an image
    img_hubs = BytesIO()
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.table(cellText=table_data_hubs, colLabels=headers, cellLoc='center', loc='center')
    plt.savefig(img_hubs, format='png')
    img_hubs.seek(0)
    img_base64_hubs = base64.b64encode(img_hubs.getvalue()).decode()

    # Clear the plot for the next use
    plt.clf()

    # Return the output as a JSON response
    return JsonResponse({
        'adjacency_matrix_data': adjacency_matrix.tolist(),
        'ranked_pages_table_data': [{'Page': page, 'Pagerank Score': score} for page, score in ranked_pages],
        'authorities_table_data': [{'Page': page, 'Authority Score': score} for page, score in authorities],
        'hubs_table_data': [{'Page': page, 'Hub Score': score} for page, score in hubs]
    })



# DFS Crawler
def get_links(session, base_url, url):
    try:
        absolute_url = urljoin(base_url, url)
        response = session.get(absolute_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [urljoin(base_url, a.get('href')) for a in soup.find_all('a', href=True)]
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
    sorted_authorities = sorted(hits_scores[1].items(), key=lambda x: x[1], reverse=True)[:10]
    sorted_hubs = sorted(hits_scores[0].items(), key=lambda x: x[1], reverse=True)[:10]
    
    return graph, sorted_authorities, sorted_hubs

def dfs_crawler(seed_url, max_depth, output_file):
    visited = set()

    def dfs(current_url, depth, session):
        if depth > max_depth:
            return
        print(f"{'  ' * depth}{current_url}")
        visited.add(current_url)

        links = get_links(session, seed_url, current_url)
        save_links_to_file([f"{seed_url} {link}" for link in links], output_file)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(dfs, link, depth + 1, session) for link in links if link not in visited]

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
        save_links_to_file([f"{seed_url} {link}" for link in links], output_file)

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
    sorted_pages = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return graph, sorted_pages

