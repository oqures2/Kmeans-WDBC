#importing necessary libraries
import pandas as pd #to help load the data
import numpy as np #to use numpy arrays as that is what I'm most comfortbale with
import matplotlib.pyplot as plt # to plot the curve L(k) vs K. value

# load the data from given link in the footnote of the assignment
URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
data = pd.read_table(URL, sep=',', header=None)
# remove examples with missing values
data = data = data[data[6] != '?']
# drop the id and class label column
data = data.drop(data.columns[[0, 10]], axis=1)
# put data to numpy array
data = data.to_numpy(dtype=int)

# data is a global variable
# here are the rest of the global variables below
num_training_examples = data.shape[0]
num_features = data.shape[1]
K_values = [2,3,4,5,6,7,8]
np.random.seed(35) # setting a seed to be able to reproduce results

# randomly choose K data points from the data set as the centroids of K clusters
# without replacement
def initialize_k_centroids(k):
    return data[np.random.choice(range(num_training_examples), k, replace=False)]

# form k clusters by assigning each point to its closest centroid
def form_k_clusters(k, centroids):
    cluster_assignments = np.zeros(num_training_examples, dtype=int)
    for index, example in enumerate(data):
        euclidean_distances = np.zeros(k)
        for i in range(k):
            euclidean_distances[i] = np.linalg.norm(example-centroids[i], ord=2)
        cluster_assignments[index] = np.argmin(euclidean_distances)
    return cluster_assignments

# recompute the k centroids
def recompute_k_centroids(k, cluster_assignments):
    new_centroids = np.zeros((k,num_features))
    counts = np.zeros(k)
    for i in range(num_training_examples):
        counts[cluster_assignments[i]] += 1

    # handle the case of finding an empty cluster in a certain iteration by
    # dropping the empty cluster and then randomly splitting the largest cluster
    # into two clusters to maintain the total number of clusters at K
    if 0.0 in counts:
        # which cluster is the empty cluster
        empty_cluster_idx = np.argmin(counts)
        # which is the largest cluster
        max_cluster_idx = np.argmax(counts)
        # get all examples in the max cluster
        max_cluster = np.where(cluster_assignments == max_cluster_idx)[0]
        cluster_to_split = cluster_assignments[max_cluster]
        # randomly split the largest cluster into 2 clusters
        np.random.shuffle(cluster_to_split)
        splitted_clusters = np.array_split(cluster_to_split, 2)
        cluster_assignments[splitted_clusters[0]] = max_cluster_idx
        counts[max_cluster_idx] = len(splitted_clusters[0])
        cluster_assignments[splitted_clusters[1]] = empty_cluster_idx
        counts[empty_cluster_idx] = len(splitted_clusters[1])

    for i in range(num_training_examples):
        new_centroids[cluster_assignments[i]] += data[i]

    for i in range(k):
        new_centroids[i] /= counts[i]

    return new_centroids

# compute the potential function
def potential(centroids, cluster_assignments):
    sum = 0
    for i in range(num_training_examples):
        squaredDist = (np.linalg.norm(centroids[cluster_assignments[i]]-data[i], ord=2))**2
        sum += squaredDist
    return sum

# plots the curve of L(K) vs. K value
def plotCurve(LK_values):
    plt.plot(K_values, LK_values, marker="o")
    plt.title("L(K) vs. K value")
    plt.xlabel("K value")
    plt.ylabel("L(K)")
    plt.show()

# run kmeans algorithm for each k value plots the curve of L(K) vs. K value
def kmeans():
    LK_values = []
    for k in K_values:
        converged = False
        centroids = initialize_k_centroids(k)
        cluster_assignments = np.zeros(num_training_examples)
        while not converged:
            cluster_assignments = form_k_clusters(k, centroids)
            new_centroids = recompute_k_centroids(k, cluster_assignments)
            if np.array_equal(new_centroids, centroids):
                converged = True
            else:
                centroids = new_centroids
        LK_values.append(potential(centroids, cluster_assignments))
    plotCurve(LK_values)

#run the program
kmeans()
