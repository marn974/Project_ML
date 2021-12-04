import sys

import numpy as np
from scipy.special import softmax

sys.path.append("..")
from headers.utils import *
from math import log2
from math import sqrt
import pandas as pd
from src.Node import Node


class NCMTree:
    def __init__(self, max_depth=10, min_samples_split=2,
                 min_samples_leaf=1, random_state=None, debug=False, distance="euclidean", root_distance="mahalanobis",
                 method_subclasses="sqrt", method_split="alea", method_max_features="sqrt", root_class=None, alpha=0.95, mode_mean=None, mode_cov=None, mode_weight=None, nbgenlayer=0):
        """

        :param max_depth:
        :param min_samples_split:
        :param min_samples_leaf:
        :param random_state:
        :param debug:
        :param distance:
        :param method_subclasses:
        :param method_split:
        :param method_max_features:
        :param root_class:
        :param alpha:
        :param nbgenlayer:
        """
        self.method_max_features = method_max_features
        self.max_features = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.root = None
        self.depth = 1
        self.debug = debug
        self.cardinality = 0
        self.distance = distance
        self.root_distance = root_distance
        self.method_subclasses = method_subclasses
        self.method_split = method_split
        self.root_class = root_class
        self.alpha=alpha
        self.nbgenlayer = nbgenlayer

        if debug:
            print('=== Initialisation NCMTree ===')
            print('\t max_features: {}'.format(self.method_max_features))
            print('\t max_depth: {}'.format(self.max_depth))
            print('\t min_samples_split: {}'.format(self.min_samples_split))
            print('\t min_samples_leaf: {}'.format(self.min_samples_leaf))
            print('\t random_state: {}'.format(self.random_state))
            print('====================================')

    def fit(self, X, y, X_nobag, y_nobag):
        """
        :param X:
        :param y:
        :return:
        """
        # number of features selected
        if self.method_max_features == 'log2':
            self.max_features = int(log2(X.shape[1]))
            assert X.shape[1] != 0, 'Null dimension'
            assert self.max_features != 0, 'self.max_features = 0'

        elif self.method_max_features == 'sqrt':
            self.max_features = int(sqrt(X.shape[1]))
            assert X.shape[1] != 0, 'Null dimension'
            assert self.max_features != 0, 'self.max_features = 0'

        elif type(self.method_max_features) == float:
            self.max_features = int(self.method_max_features * (X.shape[1]))
            assert X.shape[1] != 0, 'Null dimension'
            assert self.max_features != 0, 'self.max_features = 0'

        elif type(self.method_max_features) == int:
            self.max_features = self.method_max_features
            assert X.shape[1] != 0, 'Null dimension'
            assert self.max_features != 0, 'self.max_features = 0'

        self.root = self.build_nodes(X, y, X_nobag, y_nobag, None, 0)

    def build_nodes(self, X, y, X_nobag, y_nobag, parent, localdepth=0):
        """
        recursive function for growing tree
        :param localdepth:
        :param X:
        :param y:
        :param parent:
        :return:
        """
        # If it's a generative layer, create a generative node
        if localdepth < self.nbgenlayer:
            current_node = Node(parent, False, self.min_samples_leaf, self.max_features, self.distance, self.root_distance, self.method_subclasses, 'generative', root_class=self.root_class, alpha=self.alpha, localdepth=localdepth)
        else:
            current_node = Node(parent, False, self.min_samples_leaf, self.max_features, self.distance, self.method_subclasses,
                            self.method_split, localdepth=localdepth)
        self.cardinality += 1
        if(current_node.method_split == "generative"): #if generative layer, use unbagged X and y, to improve perf
            current_node.fit(X_nobag, y_nobag)
        else:
            current_node.fit(X, y)
        split_possible, left_index, right_index = current_node.predict_split(X)

        # ------------stopping criterion (and in Node.py, fct() predict_split )------------
        if split_possible and localdepth < self.max_depth and len(X) > self.min_samples_split:
            if self.depth < localdepth:
                self.depth = localdepth
            # LEFT NODE
            left_child = self.build_nodes(X[left_index], y[left_index], X_nobag[left_index], y_nobag[left_index], current_node, localdepth+1)
            current_node.set_left_child(left_child)
            # RIGHT NODE
            right_child = self.build_nodes(X[right_index], y[right_index], X_nobag[right_index], y_nobag[right_index], current_node, localdepth+1)
            current_node.set_right_child(right_child)
        else:
            current_node.set_leaf(True)
        return current_node

    def predict(self, X):
        """

        :param X:
        :param proba:
        :return:
        """
        return self.root.predict_all(X)
