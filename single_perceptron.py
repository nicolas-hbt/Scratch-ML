from __future__ import print_function, division
import numpy as np
import math


class Perceptron():

	def __init__(self, n_iterations=1000, activation_function=Sigmoid, loss=SquareLoss, learning_rate=0.01):
		self.n_iter = n_iterations
		self.activ_func = activation_function()
		self.loss = loss()
		self.learning_rate = learning_rate

	def fit(self, X, y):
		n_samples, n_features = X.shape
		_, n_outputs = y.shape

		# Weight initialization
        self.W = np.random.uniform((-1.0/ math.sqrt(n_features)), (1.0/math.sqrt(n_features)), (n_features, n_outputs))
        self.b = np.zeros((1, n_outputs))

        for i in range(self.n_iter):
        	linear_output = X.dot(self.W) + self.b
        	y_pred = self.activ_func(linear_output)
        	# Error gradient wrt the inputs of the activation function
        	error_grad = self.loss.gradient(y, y_pred) * self.activ_func.gradient(linear_output)
        	grad_wrt_W = X.T.dot(error_grad)
            grad_wrt_b = np.sum(error_grad, axis=0, keepdims=True)
            # Update
            self.W  -= self.learning_rate * grad_wrt_W
            self.b -= self.learning_rate  * grad_wrt_b

	def predict(self, X):
		y_pred = self.activ_func(X.dot(self.W) + self.b)
        return (y_pred)