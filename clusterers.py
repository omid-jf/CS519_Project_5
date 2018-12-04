# ======================================================================= 
# This file is part of the CS519_Project_5 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

from time import time
import inspect
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN


# This class will encapsulate all clusterers
class Clusterers(object):
    # Constructor
    def __init__(self, n_clusters, init, max_iter, tol, affinity, linkage, method, metric, t, criterion, eps, min_samples, seed=1, x=[]):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.affinity = affinity
        self.linkage = linkage
        self.method = method
        self.metric = metric
        self.t = t
        self.criterion = criterion
        self.eps = eps
        self.min_samples = min_samples
        self.seed = seed
        self.x = x
        self.__obj = None

    def call(self, method):
        return getattr(self, method)()

    def __fit_predict(self):
        start = int(round(time() * 1000))
        y = self.__obj.fit_predict(self.x)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + " fit_prediction time: " + str(end) + " ms")

        return y

    def run_kmeans(self):
        self.__obj = KMeans(n_clusters=self.n_clusters, init=self.init, max_iter=self.max_iter, tol=self.tol, random_state=self.seed)
        return self.__fit_predict()

    def run_scipy_agnes(self):
        self.__obj = AgglomerativeClustering(n_clusters=self.n_clusters, affinity=self.affinity, linkage=self.linkage)
        return self.__fit_predict()

    def run_sklearn_agnes(self):
        start = int(round(time() * 1000))
        row_cluster = linkage(y=self.x, method=self.method, metric=self.metric)
        end = int(round(time() * 1000)) - start
        print("SKlearn AGNES time: " + str(end) + " ms")

        return fcluster(row_cluster, t=self.t, criterion=self.criterion)

    def run_dbscan(self):
        self.__obj = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        return self.__fit_predict()