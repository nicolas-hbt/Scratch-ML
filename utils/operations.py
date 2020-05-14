from __future__ import division
import numpy as np
import math

def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    return (mse)

def euclidean_distance(x1, x2):
	''' Compute the euclidean (L2) distance between two vectors '''
	assert (len(x1) == len(x2)), 'Dimensions do not match'
	dist = 0
	for i in range(len(x1)) :
		dist += (x1[i] - x2[i])**2
	return (math.sqrt(dist))

def calculate_entropy(y):
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        proba = count / len(y)
        entropy += -proba * log2(proba)
    return (entropy)

def accuracy_score(y_true, y_pred):
    accu_score = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return (accu_score)

def compute_covariance_matrix(X, Y=None):
    ''' Covariance is a measure of the extent to which corresponding elements from TWO sets of ordered data move in the same direction '''
    if Y is None: # Y=None implicitly means we want to compute the variance, as we feed just one dataset into the function
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0)) # coeff : 1/(n-1) ==> unbiased estimate

    return (np.array(covariance_matrix, dtype=float))