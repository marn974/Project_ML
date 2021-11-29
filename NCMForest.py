import time
import sys
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.feature_selection import SelectKBest
from src.NCMTree import NCMTree

sys.path.append("..")
from headers.utils import *
import numpy as np
import pandas as pd

class NCMForest:

    def __init__(self, n_classes, tree_multiplier = 1, max_depth=10, min_samples_split=2,
                 min_samples_leaf=5, method_max_features='sqrt', random_state=None, debug=False, method_split="alea",
                 method_subclasses="sqrt", distance ="euclidean", root_distance="mahalanobis", alpha=0.95, nbgenlayer=0):
        """

        :param n_trees:
        :param max_depth:
        :param min_samples_split:
        :param min_samples_leaf:
        :param method_max_features:
        :param random_state:
        :param debug:
        :param method_split:
        :param method_subclasses:
        :param distance:
        :param alpha:
        :param nbgenlayer:
        """
        self.tree_multiplier = tree_multiplier
        self.n_trees = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.method_max_features = method_max_features
        self.method_split = method_split
        self.random_state=random_state
        self.debug = debug
        self.method_subclasses = method_subclasses
        self.distance = distance
        self.root_distance = root_distance
        self.alpha=alpha
        self.n_classes=n_classes
        self.nbgenlayer = nbgenlayer
        self.gen_sub_features = None
        self.trees = None

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        start = time.time()
        self.n_trees = self.n_classes * self.tree_multiplier
        self.trees = np.array([NCMTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                       min_samples_leaf=self.min_samples_leaf,
                                       random_state=self.random_state, debug=False, method_split=self.method_split,
                                       method_subclasses = self.method_subclasses, method_max_features=self.method_max_features,
                                       distance = self.distance, root_distance = self.root_distance, root_class=i%self.n_classes, 
                                       alpha=self.alpha,
                                       nbgenlayer=self.nbgenlayer) for i in range(self.n_trees)])
                
        df_input = pd.DataFrame(X)
        df_input['y'] = y
        X_noboot, y_noboot = df_input.drop(['y'], axis=1).values, df_input['y'].values
        for i in range(self.n_trees):  # build trees
            print(i)
            #create two X and y, one not boostraped(created above, before loop) and the other boostraped. This is so we can have bagging for discriminant nodes only, as bagging on generative nodes ruins perf
            boot_df, oob_df = boostrap_oob(df_input)
            X_boot, y_boot = boot_df.drop(['y'], axis=1).values, boot_df['y'].values
            self.trees[i].fit(X_boot, y_boot, X_noboot, y_noboot)  # called fit in file  NCMTree.py
        end = time.time()

        if self.debug is True:
            print('')
            print('{} trees learned {} samples in {}sec'.format(self.n_trees, len(X), round(end-start,3)));
            print('')

    def predict(self, X, return_proba=False):
        """

        :param X:
        :param return_proba:
        :return:
        """
        trees_prediction_df = pd.DataFrame()

        for i in range(0, len(self.trees)):
            tree_prediction = self.trees[i].predict(X).values
            tree_prediction_df = pd.DataFrame(pd.DataFrame(tree_prediction).loc[:, 0].values.tolist())
            tree_prediction_df.fillna(0, inplace=True)
            tree_prediction_df['tree'] = i

            tuples = list(zip(tree_prediction_df.index,tree_prediction_df['tree']))
            tree_prediction_df = tree_prediction_df.drop(['tree'], axis=1)
            index = pd.MultiIndex.from_tuples(tuples, names=['sample_id', 'tree'])
            tree_prediction_df.index = index
            trees_prediction_df = pd.concat([trees_prediction_df, tree_prediction_df], ignore_index=False, sort=False)

        trees_pred_agg_df = trees_prediction_df.groupby(level=0).mean()
        trees_pred_agg_df['y_pred'] = trees_pred_agg_df.idxmax(axis=1)

        if return_proba:

            proba = trees_pred_agg_df.max(axis=1)

            return trees_pred_agg_df['y_pred'], proba
        else:
            return trees_pred_agg_df['y_pred']

    def score(self, X, y_true):
        """

        :param X:
        :param y_true:
        :return:
        """
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

    def __str__(self):
        return """=== Initialisation NCMForest === 
        \t n_trees: """+str(self.n_trees)+"""
        \t max_features: """+str(self.method_max_features)+"""
        \t max_depth: """+str(self.max_depth)+"""
        \t method_max_features: """+str(self.method_max_features)+"""
        \t distance : """+str(self.distance)+"""
        \t min_samples_split : """+str(self.min_samples_split)

