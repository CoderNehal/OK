import base64
from io import BytesIO
import io
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
from sklearn.model_selection import train_test_split
from .models import Data
import numpy as np
from .Class_impls import My_KMeans,  My_DBSCAN, Enahnce, read_links_from_file, calculate_pagerank, calculate_hits, dfs_crawler, bfs_crawler, getLeafNodes, getInnerNodes, computeAccuracy, build_tree, prune_tree, print_tree, comp_confmat
from .utils import data1
import matplotlib.pyplot as plt
import numpy as np
from sklearn_extra.cluster import KMedoids
from mlxtend.frequent_patterns import apriori
from .temp import str
import networkx as nx
from sklearn import tree
# from tabulate import tabulate
# from concurrent.futures import ThreadPoolExecutor
from collections import deque
from urllib.parse import urljoin
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

# from LA import models


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
        Data.objects.create(file=uploaded_file)
        return JsonResponse({'message': 'File uploaded successfully'})
    else:
        return JsonResponse({'error': 'No file provided'}, status=400)

@csrf_exempt
def classifier(request):
    node = Data.objects.all()

    if len(node) == 0:
        return HttpResponse("No csv file in database !!")

    df = pd.read_csv(node[0].file)
    header = list(df.columns)
    lst = df.values.tolist()
    trainDF, testDF = train_test_split(lst, test_size=0.2)
# print("Before")
# for row in trainDF:
#     print(row[-1])
# building the tree
    t = build_tree(trainDF, header)

    # get leaf and inner nodes
    # print("\nLeaf nodes ****************")
    leaves = getLeafNodes(t)
    # for leaf in leaves:
    #     print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))

    # print("\nNon-leaf nodes ****************")
    innerNodes = getInnerNodes(t)

    # print tree
    maxAccuracy = computeAccuracy(testDF, t)
    # print("\nTree before pruning with accuracy: " + str(maxAccuracy*100) + "\n")
    # print_tree(t)

    # TODO: You have to decide on a pruning strategy
    # Pruning strategy
    nodeIdToPrune = -1
    for node in innerNodes:
        if node.id != 0:
            prune_tree(t, [node.id])
            currentAccuracy = computeAccuracy(testDF, t)
            # print("Pruned node_id: " + str(node.id) + " to achieve accuracy: " + str(currentAccuracy*100) + "%")
            # print("Pruned Tree")
            # print_tree(t)
            if currentAccuracy > maxAccuracy:
                maxAccuracy = currentAccuracy
                nodeIdToPrune = node.id
            t = build_tree(trainDF, header)
            if maxAccuracy == 1:
                break

    if nodeIdToPrune != -1:
        t = build_tree(trainDF, header)
        prune_tree(t, [nodeIdToPrune])
        # print("\nFinal node Id to prune (for max accuracy): " + str(nodeIdToPrune))
    else:
        t = build_tree(trainDF, header)
        # print("\nPruning strategy did'nt increased accuracy")

    # print("\n********************************************************************")
    # print("*********** Final Tree with accuracy: " + str(maxAccuracy*100) + "%  ************")
    # print("********************************************************************\n")
    print_tree(t)
    train = []
    test = []
    for data, node in zip(testDF, leaves):
        train.append(data[-1])
        test.append(node.predicted_label)
    print('comparision ', comp_confmat(train, test))
    # print("Confusion Matrix:")
    # print(metrics.multilabel_confusion_matrix(train, test, labels=list(set(train))))
    # print(metrics.classification_report(train, test))
    node = Data.objects.all()
    data = pd.read_csv(node[0].file)

    df = pd.DataFrame(data)
    file_name = 'fil1.csv'

    Y = df['variety']
    X = df.drop('variety', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.4, random_state=100)
    clf = DecisionTreeClassifier(max_depth=1)
    # print(clf._build_tree(),'data')
    # print(data_return)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    misclassification_rate = 1 - accuracy
    sensitivity = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    fig, ax = plt.subplots(figsize=(8, 8))
    tree.plot_tree(clf, ax=ax)
    fig.tight_layout()

    # Save the decision tree classifier image to a buffer.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    # print(buf.getvalue())
  # Create the HttpResponse object.
    response_data = {
        "name": file_name,
        'confusion_matrix': conf_matrix.tolist(),
        'accuracy': accuracy,
        'misclassification_rate': misclassification_rate,
        'sensitivity': sensitivity,
        'precision': precision,
    }
    print(response_data)
    if request.method == 'POST':
        return HttpResponse(buf.getvalue(), content_type="image/png")
    else:
        return JsonResponse(response_data)



@csrf_exempt
def decision_tree(request):
    file1 = Data.objects.all()
    d = file1[0]
    if request.method == 'POST':
        test_size = 60
        data = pd.read_csv(d.file)
        df = pd.DataFrame(data)
        X = df.drop('variety', axis=1)
        Y = df['variety']
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_size, random_state=100)
        clf = DecisionTreeClassifier(
            criterion='entropy', splitter='best', max_depth=2)
        clf.fit(X_train, y_train)

        rules = export_text(clf, feature_names=list(X.columns.tolist()))
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        coverage = len(y_pred) / len(y_test) * 100

        rule_count = len(rules.split('\n'))

        my_data = {
            'rules': rules,
            'accuracy': accuracy,
            'coverage': coverage,
            'toughness': rule_count,
        }
        return JsonResponse(my_data)
    else:
        return JsonResponse({"error": "Something went wrong"})


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

