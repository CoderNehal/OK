import base64
from io import BytesIO
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from matplotlib import pyplot as plt
import pandas as pd
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