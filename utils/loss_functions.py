from __future__ import division
from operations import accuracy_score
import numpy as np

class SquareLoss():
    def __init__(self, y, y_pred): 
    	self.y = y
    	self.y_pred = y_pred

    def loss(self, y, y_pred):
        return (0.5 * np.power((y_pred - y), 2))

    def gradient(self, y, y_pred):
        return (y_pred - y)


class CrossEntropy():
    def __init__(self, label, predict_proba): 
    	''' label : for example in a binary setting, label = 0 for man and label = 1 for woman
    	predict_proba : the predicted probability
    	You can think of label as "y_true" and predict_proba as "y_pred" '''
    	self.label = label
    	self.predict_proba = predict_proba

    def loss(self, label, predict_proba):
        # Avoid division by zero
        predict_proba = np.clip(predict_proba, 1e-15, 1 - 1e-15)
        return (-label * np.log(predict_proba) - (1 - label) * np.log(1 - predict_proba))

    def accuracy(self, label, predict_proba)
    	return accuracy_score(np.argmax(label, axis=1), np.argmax(predict_proba, axis=1))
    	# 'label' and 'predict_proba' being of dimensions (n_samples, n_classes)
    	# we basically pick the index for which the probability is maximum in each row

    def gradient(self, label, predict_proba):
        # Avoid division by zero
        predict_proba = np.clip(predict_proba, 1e-15, 1 - 1e-15)
        return (-(label / predict_proba) + (1 - label) / (1 - predict_proba))