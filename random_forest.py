import pandas as pd
import numpy as np
from collections import Counter


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y_encoded):
        self.n_features = (
            min(X.shape[1] ,self.n_features) if self.n_features else X.shape[1]
        )
        self.root = self._grow_tree(X, y_encoded)

    def _grow_tree(self, X, y_encoded, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y_encoded))

        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y_encoded)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        best_feature, best_thresh = self._best_split(X, y_encoded, feat_idxs)

        left_idxs, right_idxs = self._split(X.iloc[:, best_feature], best_thresh)

        left_node = self._grow_tree(
            X.iloc[left_idxs, :].reset_index(drop=True),
            y_encoded.iloc[left_idxs].reset_index(drop=True),
            depth + 1,
        )
        right_node = self._grow_tree(
            X.iloc[right_idxs, :].reset_index(drop=True),
            y_encoded.iloc[right_idxs].reset_index(drop=True),
            depth + 1,
        )
        return Node(best_feature, best_thresh, left_node, right_node)

    def _best_split(self, X, y_encoded, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X.iloc[:, feat_idx]
            thresholds = np.unique(X_column)
            
            if len(thresholds) == 1:
                continue

            for thr in thresholds:
                gain = self._information_gain(y_encoded, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y_encoded, X_column, threshold):

        parent_entropy = self._entropy(y_encoded)

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y_encoded)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y_encoded.iloc[left_idxs]), self._entropy(
            y_encoded.iloc[right_idxs]
        )
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        information_gain = parent_entropy - child_entropy

        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column.values <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column.values > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y_encoded):
        if len(y_encoded) == 0:
            return 0
        
        y_encoded = np.asarray(y_encoded).flatten()
        if y_encoded.dtype != int:
            y_encoded = y_encoded.astype(int)
        hist = np.bincount(y_encoded)
        ps = hist / len(y_encoded)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y_encoded):
        counter = Counter(y_encoded)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X.values])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        X = pd.DataFrame(X)
        y = pd.Series(y)
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[idxs], y.iloc[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
