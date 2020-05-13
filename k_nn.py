from __future__ import print_function, division
from utils.operations import euclidean_distance
import numpy as np
import math

class kNN():

	''' K nearest neighbors classifier.
    Parameters:
    -----------
    k : int
        Number of nearest neighbors that will determine the class of the 
        sample we want to predict.
    '''

	def __init__(self, k=5):
		self.k = k

	def vote(self, knn_labels):
		''' Return the most common class in the k nearest neighbors '''
		count = np.bincount(self.knn_labels)
		return (count.argmax())

	def predict(self, X_train, X_test, y_train):
		y_test = np.zeros(X_train.shape[0])
		for i, sample in enumerate(X_test):
			# Sort the indexes of the k nearest neighbors of the training set
			idx = np.argsort([euclidean_distance(sample, x) for x in X_train])[:self.k]
			# Based on the above indexes, find the groups to which belong the k nearest neighbors 
			knn_labels = np.array([y_train[i] for i in idx])
			# Label a particular sample as the most appearing label in the k nearest neighbors
			y_pred[i] = self.vote(knn_labels)
		return (y_pred)

