import sys

sys.path.append("..")
from headers.NCMClassifier import NCMClassifier
from headers.OneCentroid import OneCentroid
from numpy import inf
import numpy as np
import pandas as pd
import math as math
from sklearn.metrics import pairwise_distances
from numpy.random import multivariate_normal



class Node:

    def __init__(self, parent, is_leaf, min_samples_leaf, max_features, distance='euclidean', root_distance='mahalanobis', method_subclasses='sqrt', method_split='alea', root_class=None, alpha=0.95, mode_mean=None, mode_cov=None, mode_weight=None, mode_indexes=None, localdepth=None):
        """
        
        :param parent: object node() root node
        :param is_leaf: boolean,
        :param min_samples_leaf: int(), min samples to split into leaves
        :param max_features: int()
        :param distance: string()
        :param method_subclasses: string(), mathematical method choosed between { sqrt, log2, ...}
        :param method_split:string(), split method to split node in leaves
        :param root_class: int(), the class to isolate in case of a generative node
        :param alpha: int()
        """
        self.left_child = None
        self.right_child = None
        self.parent = parent
        self.splitting_clf = None
        self.left_subclasses = None
        self.right_subclasses = None
        self.is_leaf = is_leaf
        self.min_samples_leaf = min_samples_leaf
        self.distance = distance
        self.root_distance = root_distance
        self.method_subclasses = method_subclasses
        self.method_split = method_split
        self.root_class = root_class
        self.alpha = alpha
        self.mode_mean = mode_mean
        self.mode_cov = mode_cov
        self.mode_weight = mode_weight
        self.mode_indexes = mode_indexes
        self.max_features = max_features
        self.majority_class = None
        self.proportion_classes = None
        self.total_effectives = 0
        self.localdepth=localdepth

    def fit(self, X, y):
        """
        fit function: give and train data in each node created

        :param X: numpy.ndarray() input data to fit the node
        :param y: numpy.ndarray() input label to fit the node
        """
        k = np.unique(y)

        min_classes = 3
        if self.method_subclasses == 'sqrt':
            nb_subclasses = max(round(math.sqrt(len(k))), min_classes)
        elif self.method_subclasses == 'log2':
            nb_subclasses = max(round(math.log2(len(k))), min_classes)
        elif type(self.method_subclasses) == float:
            nb_subclasses = max(round(self.method_subclasses * (len(k))), min_classes)
        elif type(self.method_subclasses) == int:
            nb_subclasses = max(self.method_subclasses, min_classes)
        else:
            nb_subclasses = len(k)

        if len(k) <= min_classes or len(k) <= nb_subclasses:
            subclasses = k
        else:
            subclasses = np.random.choice(k, round(nb_subclasses), replace=False)

        sub_features = np.random.choice(np.arange(0, X.shape[1]), int(self.max_features), replace=False)
        idx_subclasses = np.where(np.in1d(y, subclasses))
 
        X_subclasses = X[np.array(idx_subclasses[0]), :]  # Recuperation de l'echantillon
        y_subclasses = y[idx_subclasses]

        self.splitting_clf = NCMClassifier(metric=self.distance, sub_features=sub_features)
        self.splitting_clf.fit(X_subclasses, y_subclasses)
        self.update_statistics(y)
        if len(k) <= 1:
            # When single leaf
            self.is_leaf = True
        else:
            # Generative node, fit classifier to isolate one class and send it to the left child, send the rest to the right child.
            if self.method_split == 'generative':
                self.splitting_clf = OneCentroid(root_class = self.root_class, alpha=self.alpha, distance=self.root_distance)
                self.splitting_clf.fit(X, y)
                self.left_subclasses = 1
                self.right_subclasses = 0
                
            # split randomly subclasses
            elif self.method_split == 'alea':
                np.random.shuffle(subclasses)
                self.left_subclasses, self.right_subclasses = np.array_split(subclasses, 2)

            # put the class with the most sample in the left child and the rest in the right child
            elif self.method_split == 'maj_class':
                filtered_proportions = pd.Series(self.proportion_classes).loc[subclasses]
                self.left_subclasses = filtered_proportions.idxmax()
                self.right_subclasses = subclasses[subclasses != self.left_subclasses]

            else:
                print("Method not defined.")

    def predict_split(self, X):
        """
        predict_split(X): return the index of data to the left or to the right
        :param X:numpy.ndarray() input data
        :return:
            - is_splittable: boolean
            - left_indexes:  numpy.ndarray()
            - right_indexes  numpy.ndarray()
        """
        is_splittable, left_indexes, right_indexes = False, None, None
        if not self.is_leaf:
            predictions = self.predict_splitting_function(X)
            left_indexes = np.where(np.in1d(predictions, self.left_subclasses))
            right_indexes = np.where(np.in1d(predictions, self.right_subclasses))

            # --------------------------- stopping criterion ( and in NCMTree.py  fct() build nodes)-------------------------------
            if len(left_indexes[0]) > self.min_samples_leaf and len(
                    right_indexes[0]) > self.min_samples_leaf:  # Critere d'arrêt : gini?
                is_splittable = True
        return is_splittable, left_indexes, right_indexes

    def predict_all(self, X):
        """
        :param X: numpy.ndarray()
        :return: pd.Series(), return proba of each classes.
        """
        index_array = np.array(np.arange(len(X)))  # store real indexes for retrieval of propagated samples
        pred = self.predict_propagate_proba(X, index_array)
        return pd.Series(pred).sort_index(ascending=True)

    def predict_propagate_proba(self, X, index_array):
        """
        :param X: numpy.ndarray()
        :param index_array:
        :return:
        """
        if not self.get_is_leaf():  # recursive tree descent
            prediction = self.predict_splitting_function(X)
            decision = np.isin(prediction, self.left_subclasses)
            left = np.where(decision)
            right = np.where(decision == False)
            if len(left[0]) > 0:
                left_prediction = self.left_child.predict_propagate_proba(X[left], index_array[left])
            else:
                left_prediction = {}
            if len(right[0]) > 0:
                right_prediction = self.right_child.predict_propagate_proba(X[right], index_array[right])
            else:
                right_prediction = {}
            left_prediction.update(right_prediction)
            return left_prediction
        else:
            return {x: self.proportion_classes for x in index_array}  # example : { 12:{ "1": 0.93, "4":0.07 } }
            # probability of each classes
            
    def predict_splitting_function(self, X):
        """
        :param X:
        :return:
        """
        return self.splitting_clf.predict(X)

    def update_statistics(self, y):
        """
        update probabilities of each classes in the node and store the total effectives
        """
        # if node has never been fitted
        if self.majority_class is None:
            self.total_effectives = len(y)
            proportions = pd.Series(y).value_counts()
            self.majority_class = proportions.index[0]
            self.proportion_classes = (np.round(proportions / proportions.sum(axis=0), 3)).to_dict()

    def depth(self):
        return max(self.left_child.depth() if self.left_child else 0,
                   self.right_child.depth() if self.right_child else 0) + 1

    def size(self):
        if self.is_leaf:
            return 1
        else:
            return self.left_child.size() + self.right_child.size() + 1

    def get_left_child(self):
        return self.left_child

    def get_right_child(self):
        return self.right_child

    def get_parent(self):
        return self.parent

    def get_is_leaf(self):
        return self.is_leaf

    def set_left_child(self, left_child):
        self.left_child = left_child

    def set_right_child(self, right_child):
        self.right_child = right_child

    def set_parent(self, parent):
        self.parent = parent

    def set_leaf(self, is_leaf):
        self.is_leaf = is_leaf

    def get_cardinality(self):
        """
        :return: 1, if it's a leaf or give numbers of child below the current node. int()
        """
        # TO CHECK
        if self.get_left_child().get_is_leaf():
            return 1
        else:
            return self.get_left_child().get_cardinality() + self.get_right_child().get_cardinality()

    def plot(self, X):
        print("########################################")
        if self.is_leaf:
            print(" LEAF : compute statitics")
        else:
            print("\t\t\t\tNODE ")
            classes = np.append(self.left_subclasses, self.right_subclasses)
            issplittable, left_indexes, right_indexes = self.predict_split(X)
            print("\t\t\t\tSELECTED CLASS : " + str(classes))

            if issplittable:
                print("LEFT NODE : " + str(self.left_subclasses))
                print("Size : " + str(len(left_indexes[0])))
                print("Proportion : " + str(round(len(left_indexes[0]) / len(X), 2) * 100))
                print("\t\t\t\t\t\t\tRIGHT NODE : " + str(self.right_subclasses))
                print("\t\t\t\t\t\t\tSize : " + str(len(right_indexes[0])))
                print("\t\t\t\t\t\t\tProportion : " + str(round(len(right_indexes[0]) / len(X), 2) * 100))
        print("########################################")
