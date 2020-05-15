from __future__ import print_function, division
import numpy as np
from utils.operations import euclidean_distance

class KMeans():

	def __init__(self, k=3, max_iterations=100):
		self.k = k
		self.max_iterations = max_iterations


	def initialize_centroids(self, X):
		n_samples, n_features = np.shape(X)
		centroids = np.zeros((self.k, n_features))
		for i in range(n_samples):
			centroids[i] = X[np.random.choice(range(n_samples))] # randomly select one sample as our initial centroid
		return (centroids)


	def form_clusters(self, centroids, X):
		n_samples = np.shape(X)[0]
		clusters = [[] for _ in range(self.k)]
		for idx_sample, sample in enumerate(X):
			idx_centroid = self.closest_centroid(sample, centroids)
			clusters[idx_centroid].append(idx_sample)
		return (clusters)


	def closest_centroid(self, sample, centroids):
		idx_closest_centroid = 0
		centroid_dist = float('inf')
		for idx, centroid in enumerate(centroids):
			dist = euclidean_distance(sample, centroid)
			if dist < centroid_dist:
				centroid_dist = dist
				idx_closest_centroid = idx 
		return (idx_closest_centroid)


	def compute_centroids(self, clusters, X):
		n_features = np.shape(X)[1]
		centroids = np.zeros((self.k, n_features))
		for idx_cluster, cluster in enumerate(clusters): # (!) 'cluster' is itself a list
			centroids[idx_cluster] = np.mean(X[cluster], axis=0) # 'X[cluster]' is a way of selecting only observations belonging to the same cluster
		return (centroids)


	def get_cluster_labels(self, clusters, X):
		y_pred = np.zeros(np.shape(X)[0])
		for idx_cluster, cluster in enumerate(clusters):
			y_pred[[cluster]] = idx_cluster
		return (y_pred) 


	def predict(self, X):
		centroids = self.initialize_centroids(X)
		for _ in range(max_iterations):
			clusters = self.form_clusters(centroids, X)
			previous_centroids = centroids
			centroids = self.compute_centroids(clusters, X)
			if centroids == previous_centroids:
				break
		return (self.get_cluster_labels(clusters, X))
