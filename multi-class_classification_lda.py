from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

class MultiClassLDA():

    def __init__(self):
        self.W = None

    def fit_transform(self, X, y, n_components):
        X = X.values if type(X) is not np.ndarray else X
        n_features = np.shape(X)[1]
        classes = np.unique(y)
        n_classes = len(classes)

        ## STEP 1
        mean_vectors = []
        for cl in range(1, n_classes+1):
            mean_vectors.append(np.mean(X[y==cl], axis=0))
        S_W = np.zeros((n_features, n_features)) # within-class scatter matrix
        for cl, mv in zip(range(1, n_classes+1), mean_vectors): 
            class_sc_mat = np.zeros((n_features, n_features)) # scatter matrix for every class
            for x in X[y == cl] :
                x, mv = x.reshape(n_features,1), mv.reshape(n_features,1) # make column vectors 
                class_sc_mat += (x-mv).dot((x-mv).T)
            S_W += class_sc_mat

        ## STEP 2
        overall_mean = np.mean(X, axis=0)
        S_B = np.zeros((n_features, n_features)) # between-class scatter matrix
        for i,mean_vec in enumerate(mean_vectors):  
            n = X[y==i+1,:].shape[0] # number of observations for a given label. Start from 1.
            mean_vec, overall_mean = mean_vec.reshape(n_features,1), overall_mean.reshape(n_features,1) # make column vector
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

        ## STEP 3
        eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

        ## STEP 4
        # Sort the eigenvalues in descending order and keep only the first n_components
        sorted_idx = eigen_vals.argsort()[::-1]
        eigen_vals = eigen_vals[sorted_idx][:n_components]
        eigen_vecs = eigen_vecs[:, sorted_idx][:, :n_components]
        
        # Projection
        X_transformed = X.dot(eigen_vecs).real

        return (X_transformed)

    def plot_2D(self, X, y, title="2D Projection using LDA transformation"):
        labels = y
        X_transformed = self.fit_transform(X, y, n_components=2)
        X_LD1 = X_transformed[:, 0]
        X_LD2 = X_transformed[:, 1]
        plt.figure(figsize=(10,7))
        label_dict = {}
        keys = range(len(np.unique(y)))
        values = sorted(y.unique().tolist())
        for label in keys:
            label_dict[label] = values[label]
            plt.scatter(X_LD1[y==label+1], X_LD2[y==label+1], alpha=0.7, label=label_dict[label])
        leg = plt.legend(loc='upper right', fancybox=True)
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        if title: plt.title(title)
        plt.grid()
        plt.tight_layout
        plt.show()