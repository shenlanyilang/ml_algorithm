import numpy as np
import sys
from collections import Counter

class Node(object):
    def __init__(self):
        self.father = None
        self.left = None
        self.right = None
        self.data_index = None
        self.split_feature = None
        self.split_value = None
        self.depth = None
        self.positive_prob = None

class ClassificationTree(object):
    def __init__(self, max_depth=4, min_leaves=8):
        self.max_depth = max_depth
        self.min_leaves = min_leaves
        self.feature_nums = None
        self.root = Node()
        self.root.depth = 0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1,1)
        self.train_x = X
        self.train_y = y
        self.sorted_feature = {}
        self.feature_nums = X.shape[1]
        for i in range(self.feature_nums):
            sorted_feature = sorted(list(set(self.train_x[:, i])))
            self.sorted_feature[i] = sorted_feature
        self.root.data_index = range(X.shape[0])
        self.root.positive_prob = sum(y) / len(y)
        self._feature_select(self.root)

    def chnt_cal(self, cat_list):
        counter = Counter(cat_list)
        chnt = 0
        nums = len(cat_list)
        for v in counter.values():
            chnt += -np.log(v/nums)*(v/nums)
        return chnt

    def _feature_select(self, node:Node):
        left_node = Node()
        right_node = Node()
        left_node.depth = node.depth + 1
        right_node.depth = node.depth + 1
        left_node.father = node
        right_node.father = node
        if left_node.depth > self.max_depth or right_node.depth > self.max_depth:
            return
        data_indx = node.data_index
        data_nums = len(data_indx)
        min_split_chnt = sys.maxint
        feature_select = None
        split_value = None
        split_success = False
        for f in range(self.feature_nums):
            #feature_values = list(set(self.train_x[data_indx,:], f))
            left_collect = []
            right_collect = []
            for split in sorted(self.sorted_feature[f]):
                for idx in data_indx:
                    if self.train_x[idx, f] < split:
                        left_collect.append((idx, self.train_y[idx]))
                    else:
                        right_collect.append((idx, self.train_y[idx]))
                if len(left_collect) < self.min_leaves or len(right_collect) < self.min_leaves:
                    continue
                split_chnt = len(left_collect)/data_nums*self.chnt_cal([j for i, j in left_collect]) \
                    + len(right_collect)/data_nums*self.chnt_cal([j for i, j in right_collect])
                if split_chnt < min_split_chnt:
                    split_success = True
                    min_split_chnt = split_chnt
                    feature_select = f
                    split_value = split
                    left_node.positive_prob = sum([j for i, j in left_collect]) / len(left_collect)
                    right_node.positive_prob = sum([j for i, j in right_collect]) / len(right_collect)
                    left_node.data_index = [i for i, j in left_collect]
                    right_node.data_index = [i for i, j in right_collect]
        if not split_success:
            return
        node.left = left_node
        node.right = right_node
        node.split_feature = feature_select
        node.split_value = split_value
        self.feature_select(left_node)
        self.feature_select(right_node)

    def predict_prob(self, x):
        node = self.root
        pred_prob = None
        while node.left and node.right:
            if x[node.split_feature] < node.split_value:
                node = node.left
            else:
                node = node.right
            pred_prob = node.positive_prob
        return pred_prob


class RegressionTree(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, x):
        pass