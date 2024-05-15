import numpy as np
from collections import defaultdict
import pandas as pd
import random

import sys
sys.setrecursionlimit(10**6)

class Node:
    def __init__(self, f=0):
        self.feature = f
        self.left = None
        self.right = None

def gini_index(counts):
    total = sum(counts)
    if total == 0:
        return 0
    probs = [count / total for count in counts]
    gini = 1 - sum((p ** 2) for p in probs)
    return gini

def decision(features, rows, root, data):
    pos = 0
    neg = 0
    n = len(rows)
    for r in rows:
        if data[r][-1] == 1:
            pos += 1
        else:
            neg += 1

    if not pos:
        root.feature = 0
        return
    if not neg:
        root.feature = 1
        return

    pos_p = pos / n
    neg_p = neg / n
    H = -(pos_p * np.log2(pos_p) + neg_p * np.log2(neg_p))

    FF = defaultdict(int)
    FT = defaultdict(int)
    TF = defaultdict(int)
    TT = defaultdict(int)

    for r in rows:
        for f in features:
            if data[r][f] == 0:
                if data[r][-1] == 0:
                    FF[f] += 1
                else:
                    FT[f] += 1
            else:
                if data[r][-1] == 0:
                    TF[f] += 1
                else:
                    TT[f] += 1

    min_gini = float('inf')
    attri = 0

    for f in features:
        true = TT[f] + TF[f]
        false = FT[f] + FF[f]
        d = true + false

        counts_true = [TT[f], TF[f]]
        counts_false = [FT[f], FF[f]]
        gini_true = gini_index(counts_true)
        gini_false = gini_index(counts_false)

        rem = (true / d) * gini_true + (false / d) * gini_false
        gain = H - rem

        if gain < min_gini:
            min_gini = gain
            attri = f

    root.feature = attri

    left, right = set(), set()
    for r in rows:
        if data[r][attri] == 1:
            right.add(r)
        else:
            left.add(r)

    l = Node()
    r = Node()
    root.left = l
    root.right = r
    features.discard(attri)
    if not features:
        return

    if not left or not right:
        pos = 0
        neg = 0
        for r in rows:
            if data[r][-1] == 0:
                neg += 1
            else:
                pos += 1
        if not left:
            root.left.feature = (1 if pos > neg else 0)
            decision(features, right, root.right, data)

        if not right:
            root.right.feature = (0 if neg > pos else 1)
            decision(features, left, root.left, data)

        features.add(attri)
        return

    decision(features, left, root.left, data)
    decision(features, right, root.right, data)
    features.add(attri)

# Load your dataset
data = pd.read_csv('data.csv').to_numpy()
n = len(data)
p = int(0.5 * n)
test = data[p:]
data = data[:p]

Trees = []
for i in range(20):
    features = list(range(data.shape[1] - 1)) 
    random.shuffle(features)
    selected_features = features[:15]  
    Tree = Node()
    sample_indices = set(range(len(data)))
    decision(set(selected_features), sample_indices, Tree, data)
    Trees.append(Tree)

def fun(root, data):
    if not root.left and not root.right:
        return root.feature
    x = root.feature
    if data[x] == 0:
        if not root.left:
            return 1
        return fun(root.left, data)
    else:
        if not root.right:
            return 0
        return fun(root.right, data)

def predict(data):
    correct = -30
    for row in data:
        out = row[-1]
        row_data = row[:-1]
        pos = 0
        neg = 0
        for tree in Trees:
            r = fun(tree, row_data)
            if r == 0:
                neg += 1
            else:
                pos += 1
        predicted_class = 1 if pos >= neg else 0
        if out == predicted_class:
            correct += 1
    return correct

correct = predict(test)

print("Total Test Samples       : ", len(test))
print("Total Correct Predictions: ", correct)
print("Accuracy                 : {:.2f}%".format(correct / len(test) * 100))