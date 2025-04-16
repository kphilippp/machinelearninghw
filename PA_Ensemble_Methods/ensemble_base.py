# Keven Diaz - KXD210034
# Kevin Philip - KXP210063

# Decision tree functions pulled from Keven and Tina Nguyen PA2

# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np

#Import Sklearn Libraries for comparison
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import pandas as pd

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    partitions = {}
    for i, value in enumerate(x):
        if value not in partitions:     # make sure the value is unique
            partitions[value] = []
        partitions[value].append(i)

    return partitions


def entropy(y, weights=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z. 
    Include the weights of the boosted examples if present

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    counts = np.bincount(y)
    prob = counts / len(y)
    entropy = -np.sum(prob * np.log2(prob + 1e-10))    # 1e-10 prevents log(0)
    return entropy


def mutual_information(x, y, weights=None):
    """
    
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)

    Compute the weighted mutual information for Boosted learners
    """

    H_y = entropy(y)
    partitions = partition(x)

    H_y_given_x = 0
    for _, indices in partitions.items():
        H_y_given_x += len(y[indices]) / len(y) * entropy(y[indices])

    mutual_info = H_y - H_y_given_x
    return mutual_info


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, weights=None):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    if attribute_value_pairs is None:
        attribute_value_pairs = [(attr, val) for attr in range(
            x.shape[1]) for val in np.unique(x[:, attr])]

    # Base cases
    if len(np.unique(y)) == 1:
        return y[0]
    if len(attribute_value_pairs) == 0 or depth >= max_depth:
        return np.bincount(y).argmax()

    # Find best split
    best_pair = None
    best_info_gain = -1
    for pair in attribute_value_pairs:
        attr, val = pair
        feature = (x[:, attr] == val).astype(int)
        info_gain = mutual_information(feature, y)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_pair = pair

    if best_pair is None or best_info_gain <= 0:
        return np.bincount(y).argmax()

    attr, attr_val = best_pair
    feature = (x[:, attr] == attr_val).astype(int)

    remaining_pairs = [p for p in attribute_value_pairs if p[0] != attr]

    # Create tree
    tree = {}
    for branch_value in [0, 1]:
        indices = np.where((x[:, attr] == attr_val) == branch_value)[0]
        if len(indices) == 0:
            tree[(attr, attr_val, branch_value == 1)] = np.bincount(y).argmax()
        else:
            tree[(attr, attr_val, branch_value == 1)] = id3(
                x[indices], y[indices], remaining_pairs, depth + 1, max_depth
            )

    return tree


def bootstrap_sampler(x, y, num_samples,  weights=None, ensemble_method="bagging"):
    """
    This method sample the dataset where some data points are duplicated 
    """
    if ensemble_method == "bagging":
        indices = np.random.choice(len(x), num_samples, replace=True)
        return x[indices], y[indices]
    else:
        sample_indices = np.random.choice(num_samples, num_samples, p=weights)
        return (x[sample_indices], y[sample_indices])
        


def bagging(x, y, max_depth, num_trees):
    """
    Implements bagging of multiple id3 trees where each tree trains on a boostrap sample of the original dataset
    """
    
    """
    What this method does is keeps each tree in an array called ensemble
    Then for the number of trees you specify, you want to
    - Get a bootstrapped sample set based on the original data
    - train a decision tree using the id3 algorithm and the data from step 1
    - then after train, add it to our list of trees

    """

    ensemble = []
    for _ in range(num_trees):
        bootstrappedX, bootstrappedY = bootstrap_sampler(x,y, len(x))
        trainedTree = id3(bootstrappedX, bootstrappedY, max_depth=max_depth)
        ensemble.append((1, trainedTree))

    return ensemble


def boosting(x, y, max_depth, num_stumps):

    """
    Implements an adaboost algorithm using the id3 algorithm as a base decision tree
    """
    """
    This function first sets the number of examples we have and the weight for each example to 1
    Then for each weak learned we bootstrap the dataset based on the weights 
       of each example (higher weights have a higher chance of getting picked)
    Then we train a tree on that data and get our predictions
    We then compute error which is the sum of the misclassified examples weights
    We say the weight of the tree or alpha is 1/2 * the log of 1-error / error + small number
    Then we can update the weights where the correct examples get a smaller weight while
        the misclassified examples get a bigger weight, more chance of getting bootstrapped
    """

    n = len(x)
    w = np.ones(n) / n
    ensemble = []

    for _ in range(num_stumps):
        strappedX, strappedY = bootstrap_sampler(x, y, n, w, ensemble_method="boosting")

        tree = id3(strappedX, strappedY, max_depth=max_depth)
        y_pred = np.array([predict_example(xi, tree) for xi in x])

        err = np.sum(w * (y_pred != y))
        if err == 0:
            alpha = 1
        else:
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))

        w *= np.exp(-alpha * y * (2 * y_pred - 1))
        w /= np.sum(w)
        ensemble.append((alpha, tree))

    return ensemble



def predict_example_ens(x, h_ens):
    """
    Predicts the classification label for a single example x using a combination of weighted trees
    Returns the predicted label of x according to tree
    """

    score = 0
    for alpha, h in h_ens:
        pred = predict_example(x, h)
        vote = 1 if pred == 1 else -1
        score += alpha * vote
    return 1 if score >= 0 else 0



def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    return np.mean(y_true != y_pred)


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))
            
            
def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # if the tree is just a leaf
    if not isinstance(tree, dict):
        return tree

    for (attr_idx, attr_val, true_branch), subtree in tree.items():
        if (x[attr_idx] == attr_val) == true_branch:
            return predict_example(x, subtree)
        

def split_data(filename):

    M = np.genfromtxt(f'{filename}.train', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt(f'{filename}.test', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    return Xtrn, ytrn, Xtst, ytst

def dt_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives

    return TN, FP, FN, TP

if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    visualize(decision_tree)

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))


    # Experiments with Ensemble Learning

    monk_datasets = ['monks-1', 'monks-2', 'monks-3']

    for dataset in monk_datasets:
        print(f"\nPart e: Ensemble Learning for {dataset} ---------------------------------")
        Xtrn, ytrn, Xtst, ytst = split_data(dataset)
    
        # Baggin 💯 
        for depth in [3, 5]:    
            for numTrees in [10, 20]:
                print(f"\n\nBagging: Depth={depth}, Trees={numTrees}")
                h_ens = bagging(Xtrn, ytrn, max_depth=depth, num_trees=numTrees)
                y_pred = np.array([predict_example_ens(x, h_ens) for x in Xtst]) 
                TN, FP, FN, TP = dt_confusion_matrix(ytst, y_pred)
                print(f"\nTN: {TN}, FP: {FP}\nFN : {FN}, TP: {TP}")
 
        # Boostin 💪
        for depth in [1, 2]:
            for k in [20, 40]:
                print(f"\n\nBoosting: Depth={depth}, Stumps={k}")
                h_ens = boosting(Xtrn, ytrn, max_depth=depth, num_stumps=k)
                y_pred = np.array([predict_example_ens(x, h_ens) for x in Xtst])  
                TN, FP, FN, TP = dt_confusion_matrix(ytst, y_pred)                
                print(f"\nTN: {TN}, FP: {FP}\nFN : {FN}, TP: {TP}")
                


# Scikit-Learn Part A ------------------------------------------------------------
    print("\nPart C: Scikit-Learn Ensemble Comparison -----------------")
    for dataset in monk_datasets:
        print(f"\nDataset: {dataset}")
        Xtrn, ytrn, Xtst, ytst = split_data(dataset)
        
        # Bagging Experiments
        for depth in [3,5]:
            for num_trees in [10,20]:
                clf = BaggingClassifier(
                    estimator=DecisionTreeClassifier(max_depth=depth),
                    n_estimators=num_trees,
                    bootstrap=True
                )
                clf.fit(Xtrn, ytrn)
                y_pred = clf.predict(Xtst)
                print(f"\nBagging (depth={depth}, trees={num_trees})")
                print(confusion_matrix(ytst, y_pred))
                
        # Boosting Experiments
        for depth in [1,2]:
            for num_stumps in [20,40]:
                clf = AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=depth),
                    n_estimators=num_stumps
                )
                clf.fit(Xtrn, ytrn)
                y_pred = clf.predict(Xtst)
                print(f"\nBoosting (depth={depth}, stumps={num_stumps})")
                print(confusion_matrix(ytst, y_pred))