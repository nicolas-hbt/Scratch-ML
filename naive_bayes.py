from __future__ import division, print_function
import numpy as np
import math


class NaiveBayes():

    def fit(self, X, y):
    	self.X = X
    	self.y = y
    	self.classes = np.unique(y) # number of different classes
    	self.param = [] # empty list for future parameters to be computed (one set of params for each class)
    	for idx, c in enumerate(self.classes):
    		X_subset = X[np.where(c == y)] # select only rows of X where the value of y matches with the right class : c
    		self.param.append([])
    		for feature in X_subset.T :
    			params = {"mean" : feature.mean(), "variance" : feature.var()}
    			self.param[idx].append(params)


    def calculate_likelihood(self, mean, var, x):
    	''' Assuming a Gaussian distribution for the probability distribution P(X), we return the likelihood of data X given mean and var '''
    	eps = 1e-6 # Avoid division by zero
    	return ( (1.0/math.sqrt(2.0 * math.pi * var + eps)) * math.exp(-((x - mean)**2 / (2 * var + eps))))


    def calculate_prior(self, c):
    	'''The prior is simply the frequency of each class in the train set'''
    	freq = np.sum(self.y == c) / self.y.shape[0]
    	return (freq)


    def classify(self, sample):
    	''' Based on Bayes rule : P(Y|X) = P(X|Y)*P(Y)/P(X) '''
    	posteriors = []
    	for idx, c in enumerate(self.classes):
    		posterior = self.calculate_prior(c) # Initialize P(Y|X) as P(Y)
    		# This is based on the assumption that P(Y|X) = P(X|Y)*P(Y)
    		# Then, in the following, we take care of "P(X|Y)" and increment for as many features as we have 
    		for feature_val, params in zip(sample, self.param[idx]) :
    			likelihood = self.calculate_likelihood(params["mean"], params["variance"], feature_val)
    			posterior *= likelihood
    		posteriors.append(posterior)
    	return (self.classes[np.argmax(posteriors)]) # Return the class with maximum posterior probability


    def predict(self, X):
    	y_pred = [self.classify(x) for x in X]
    	return (y_pred)