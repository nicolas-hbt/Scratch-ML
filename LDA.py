class LDA():

    def __init__(self):
        self.W = None

    def transform(self, X, y):
        self.fit(X, y)
        X_transform = X.dot(self.W) # Projection
        assert X_transform.shape[1] == 2, "The matrix is not of right dimensions."
        return (X_transform)

    def fit(self, X, y):
        X = X.values if type(X) is not np.ndarray else X
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        ## STEP 1
        self.mean_vectors = []
        for cl in range(1, self.n_classes+1):
            self.mean_vectors.append(np.mean(X[y==cl], axis=0))
        dim = X.shape[1]
        self.S_W = np.zeros((dim, dim)) # within-class scatter matrix
        for cl, mv in zip(range(1, self.n_classes+1), self.mean_vectors): 
            class_sc_mat = np.zeros((dim, dim)) # scatter matrix for every class
            for x in X[y == cl] :
                x = x.reshape(dim,1)
                mv = mv.reshape(dim,1) # make column vectors
                class_sc_mat += (x-mv).dot((x-mv).T)
            self.S_W += class_sc_mat

        ## STEP 2
        overall_mean = np.mean(X, axis=0)
        self.S_B = np.zeros((dim, dim)) # between-class scatter matrix
        for i,mean_vec in enumerate(self.mean_vectors):  
            n = X[y==i+1,:].shape[0]
            mean_vec, overall_mean = mean_vec.reshape(dim,1), overall_mean.reshape(dim,1) # make column vector
            self.S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

        ## STEP 3
        self.eigen_vals, self.eigen_vecs = np.linalg.eig(np.linalg.inv(self.S_W).dot(self.S_B))
        for i in range(len(self.eigen_vals)):
            self.eigenvec_sc = self.eigen_vecs[:,i].reshape(dim,1)   

        ## STEP 4
        # Make a list of (eigenvalue, eigenvector) tuples
        self.eigen_pairs = [(np.abs(self.eigen_vals[i]), self.eigen_vecs[:,i]) for i in range(len(self.eigen_vals))]
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        self.eigen_pairs = sorted(self.eigen_pairs, key=lambda k: k[0], reverse=True)
        # Keep the first two linear discriminants
        self.W = np.hstack((self.eigen_pairs[0][1].reshape(dim,1), self.eigen_pairs[1][1].reshape(dim,1)))
        return (self.W)

        def predict(self, X):
