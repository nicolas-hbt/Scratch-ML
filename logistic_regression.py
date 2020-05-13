from __future__ import print_function, division
import numpy as np
import math

class LogisticRegression():
	''' We use Gradient Descent for training here. An alternative is to use batch optimization by least squares method. '''

	def __init__(self, learning_rate=0.01):
		self.learning_rate = learning_rate

	def sigmoid(X):
		return (1 / (1 + np.exp(-X)))

	def fit(self, X, y, n_iterations=1000):
		self.X = X
		self.y = y 
		self.params = np.zeros(np.shape(X)[1])
		for i in range(n_iterations):
			y_pred = sigmoid(X.dot(self.params))
			self.params -= self.learning_rate * (y_pred - y).dot(X)

	def predict(self, X):
		y_pred = self.sigmoid(X.dot(self.param))
		return (y_pred)

