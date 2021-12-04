import numpy as np
import cProfile
from sklearn.metrics.pairwise import pairwise_distances

class OneCentroid:
    def __init__(self, root_class, alpha=0.95, distance="euclidean", verbose=False):
        self.root_class = root_class
        self.alpha = alpha
        self.distance = distance
        self.centroid = None
        self.distances_fit = None
        self.distances_pred = None
        self.cov_matrix = None
        self.inv_cov_matrix = None
        self.radius_value = None
    
    def fit(self, X, y):
        '''
        desc: This function trains OneCentroid on X (X here being one of original X's 10 classes)
        The goal of OneCentroid is to place one centroid at the barycenter of a class and then extend a radius around this centroid
        to fit the class samples and thus create a definition of the class (centroid=mean, radius=variance).
        '''
        #isolate root class
        X_root_class = X[np.where(np.isin(y, self.root_class))]
        #initialize radius_value
        self.radius_value = 0
        #Set centroïd to barycenter of desired class
        self.init_centroid(X_root_class)
        #compute convariance matrix
        self.cov_matrix = np.cov(X_root_class.T)
        #compute pseudo-inverse of convariance matrix (we use pseudo-inverse to account for potential negative values)
        self.inv_cov_matrix = np.linalg.pinv(self.cov_matrix)
        distances = []
        if(self.distance=="euclidean"):
            dist = pairwise_distances(X_root_class, self.centroid.reshape(1, -1), "euclidean")
            distances=dist[:, 0]
        elif(self.distance=="mahalanobis"):
            dist = pairwise_distances(X_root_class, self.centroid.reshape(1, -1), "mahalanobis", VI = self.inv_cov_matrix)
            distances=dist[:, 0]
        distances = np.sort(distances)
        self.distances_fit = distances #stores all distances to centroid in fitted sample
        self.radius_value = distances[int(len(distances)*self.alpha)]
           
    
    def init_centroid(self, X_root_class):
        '''
        desc: Compute barycenter of specified root class in y to set it as the starting point for the centroid
        '''
        centroid = np.mean(X_root_class, axis = 0)
        
        self.centroid = centroid
        return self.centroid
    
    def predict(self,X):
        '''
        Assign either 0 or 1 to each sample of X, 1 supposedly containing a large majority of the root class
        '''
        self.distances = []
        predictions = []
        distances = []
        for x in X:
            pred_class=0
            if(self.distance=="euclidean"):
                dist = np.linalg.norm(x - np.array(self.centroid))
                distances.append(dist)
            elif(self.distance=="mahalanobis"):
                dist = pairwise_distances(x.reshape(1, -1), self.centroid.reshape(1, -1), "mahalanobis", VI = self.inv_cov_matrix)
                distances.append(dist[0][0])
            if(dist < self.radius_value):
                pred_class = 1
            predictions.append(pred_class)
        self.distances_pred = distances #stores all distances to centroid in predicted sample
        return predictions