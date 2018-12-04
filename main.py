# ======================================================================= 
# This file is part of the CS519_Project_5 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import clusterers
from sklearn.neighbors import NearestNeighbors

for dataset in ["iris", "plates"]:

    # Iris dataset
    if dataset == "iris":
        print("\n\n************")
        print("Iris dataset")
        print("************")

        # Preprocessing
        # Reading the file
        df = pd.read_csv("iris.data", header=None)

        # Separating x and y
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        names = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        y = [names[y[i]] for i in range(len(y))]

    elif dataset == "plates":
        print("\n\n***************************")
        print("Faulty Steel Plates dataset")
        print("***************************")

        # Preprocessing
        # Reading the file
        df = pd.read_csv("faults.csv", header=0)

        # Separating x and y
        x = df.iloc[:, :-7].values
        y = df.iloc[:, -7:].values
        y = ["".join(item) for item in y.astype(str)]
        names = {"1000000": 0, "0100000": 1, "0010000": 2, "0001000": 3, "0000100": 4, "0000010": 5, "0000001": 6}
        y = [names[y[i]] for i in range(len(y))]


    # Elbow method
    sse_list = []

    for k in range(1, 15):
        km = KMeans(n_clusters=k, init="k-means++", max_iter=300, random_state=0)
        km.fit(x)
        sse_list.append(km.inertia_)

    plt.plot(range(1, 15), sse_list)
    plt.title("Finding the number of clusters (elbow method)")
    plt.ylabel("SSE")
    plt.xlabel("k")
    plt.show()

    if dataset == "iris":
        k = 3
    elif dataset == "plates":
        k = 7

    # Find MinPts and eps
    num_nn = 10
    nn = NearestNeighbors(n_neighbors=num_nn + 1)
    nbrs = nn.fit(x)
    distances, indices = nbrs.kneighbors(x)
    distanceK = np.empty([num_nn, x.shape[0]])
    for i in range(num_nn):
        di = distances[:, (i+1)]
        di.sort()
        di = di[::-1]
        distanceK[i] = di
    for i in range(num_nn):
        plt.plot(distanceK[i], label="K=%d" %(i+1))
        plt.ylabel("Distance")
        plt.xlabel("Points")
        plt.legend()
        plt.show()

    clusterer = clusterers.Clusterers(n_clusters=2, init="k-means++", max_iter=300, tol=1e-04, affinity="euclidean",
                                      linkage="single", method="single", metric="euclidean", t=2.5, criterion="distance",
                                      eps=0.2, min_samples=5, seed=0, x=x)

    for clusterer_name in ["kmeans", "scipy_agnes", "sklearn_agnes", "dbscan"]:
        print("\n\n==========================")
        print(clusterer_name.upper())

        y_pred = clusterer.call("run_" + clusterer_name)

        print("Accuracy")

        correct = 0
        for instance in range(x.shape[0]):
            if y[instance] == y_pred[instance]:
                correct += 1

        accuracy = correct / x.shape[0] * 100
        print("Accuracy: " + str(accuracy) + " %")

















