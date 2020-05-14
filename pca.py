from __future__ import print_function, division
import numpy as np
from utils import compute_covariance_matrix

class PCA():

    def transform(self, X, n_components):
        covariance_matrix = compute_covariance_matrix(X)

        # eigen_vecs[:,i] corresponds to eigen_vals[i]
        eigen_vals, eigen_vecs = np.linalg.eig(covariance_matrix)

        # Sort the eigenvalues and corresponding eigenvectors in descending order
        # and only keep the first n_components
        idx = eigen_vals.argsort()[::-1]
        eigen_vals = eigen_vals[idx][:n_components]
        eigen_vecs = eigen_vecs[:, idx][:, :n_components]

        # Data projection onto the principal components
        X_transformed = X.dot(eigen_vecs)

        return (X_transformed)