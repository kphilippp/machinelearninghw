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


#Tina Nguyen NBN210002
#Keven Diaz KXD210034

import numpy as np

from sklearn.metrics import confusion_matrix
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
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


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    counts = np.bincount(y)
    prob = counts / len(y)
    entropy = -np.sum(prob * np.log2(prob + 1e-10))    # 1e-10 prevents log(0)
    return entropy


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    H_y = entropy(y)
    partitions = partition(x)

    H_y_given_x = 0
    for _, indices in partitions.items():
        H_y_given_x += len(y[indices]) / len(y) * entropy(y[indices])

    mutual_info = H_y - H_y_given_x
    return mutual_info


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
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
        print(
            '+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


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


def learning_curves(Xtrn, ytrn, Xtst, ytst, depths=range(1, 11)):
    train_errors = []
    test_errors = []

    for depth in depths:
        tree = id3(Xtrn, ytrn, max_depth=depth)

        # Compute training error
        ytrn_pred = [predict_example(x, tree) for x in Xtrn]
        trn_error = compute_error(ytrn, ytrn_pred)

        # Compute test error
        ytst_pred = [predict_example(x, tree) for x in Xtst]
        tst_error = compute_error(ytst, ytst_pred)

        train_errors.append(trn_error)
        test_errors.append(tst_error)

    return depths, train_errors, test_errors


def plotting_curves(depths, train_errors, test_errors, title):
    plt.plot(depths, train_errors, label="Training Error", linestyle="--")
    plt.plot(depths, test_errors, label="Test Error")
    plt.xlabel("Tree Depth")
    plt.ylabel("Error")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def dt_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives

    return TP, FP, FN, TN


if __name__ == '__main__':

    monks_datasets = ["monks-1", "monks-2", "monks-3"]

# Part A: Learning curves ------------------------------------------------
    for dataset in monks_datasets:
        print(f"Part a: learning {dataset}")

        Xtrn, ytrn, Xtst, ytst = split_data(dataset)

        depths, train_errors, test_errors = learning_curves(
            Xtrn, ytrn, Xtst, ytst)

        plotting_curves(depths, train_errors, test_errors,
                        f"Learning Curve for {dataset}")

# Part B: Weak learners -------------------------------------------------
    Xtrn, ytrn, Xtst, ytst = split_data("monks-1")

    for depth in [1, 2]:
        print(f"\nPart b: Decision Tree (Depth={depth})")

        decision_tree = id3(Xtrn, ytrn, max_depth=depth)
        visualize(decision_tree)

        y_pred = np.array([predict_example(x, decision_tree) for x in Xtst])

        TP, FP, FN, TN = dt_confusion_matrix(ytst, y_pred)

        print("\nConfusion Matrix:")
        # print confusion matrix in 2x2 grid
        print(np.array([[TN, FP], [FN, TP]]))

# Part C: scikit-learn --------------------------------------------------
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(Xtrn, ytrn)
    y_pred = decision_tree.predict(Xtst)
    sklearn_confusion_matrix = np.array(confusion_matrix(ytst, y_pred))
    print("\nPart c: confusion matrix using scikit-learn:")
    print(sklearn_confusion_matrix)

    tree_data = export_graphviz(decision_tree)
    graph = graphviz.Source(tree_data)
    graph.render("decision_tree")
    graph.view()  

# Part D: UC Irvine Data Set ----------------------------------------------
    cryo_data_df = pd.read_csv("Cryotherapy.csv")

    cryo_columns = ['sex', 'age', 'Time', 'Number_of_Warts', 'Type', 'Area', 'Result_of_Treatment']
    cryo_data_df = cryo_data_df.loc[:, cryo_columns]

    cryo_features = ['sex', 'age', 'Time', 'Number_of_Warts', 'Type', 'Area']
    cryo_X = cryo_data_df.loc[:, cryo_features]
    cryo_Y = cryo_data_df.loc[:, ['Result_of_Treatment']]

    continuous_features = ['age', 'Time', 'Number_of_Warts', 'Area']

    for feature in continuous_features:
        mean_value = cryo_X[feature].mean()
        cryo_X[feature] = (cryo_X[feature] > mean_value).astype(int)

    cryo_X_trn, cryo_X_tst, cryo_y_trn, cryo_y_tst = train_test_split(cryo_X, cryo_Y, random_state=0, train_size=0.75)
    
    # part b: weak learners for UCI dataset
    for depth in [1, 2]:
        print(f"\nPart d: Cryotherapy Decision Tree (Depth={depth})")

        cryo_decision_tree = id3(cryo_X_trn.to_numpy(), cryo_y_trn.to_numpy().flatten(), max_depth=depth)
        visualize(cryo_decision_tree)

        cryo_y_pred = np.array([predict_example(x, cryo_decision_tree) for x in cryo_X_tst.to_numpy()])

        TP, FP, FN, TN = dt_confusion_matrix(cryo_y_tst.to_numpy().flatten(), cryo_y_pred)

        print("\nConfusion Matrix:")
        # print confusion matrix in 2x2 grid
        print(np.array([[TN, FP], [FN, TP]]))
        
    # part c: scikit-learn for UCI dataset
    cryo_decision_tree = DecisionTreeClassifier()
    cryo_decision_tree.fit(cryo_X_trn, cryo_y_trn)
    cryo_y_pred = cryo_decision_tree.predict(cryo_X_tst)
    sklearn_confusion_matrix = np.array(confusion_matrix(cryo_y_tst, cryo_y_pred))
    print("\nPart d: Cryotherapy confusion matrix using scikit-learn:")
    print(sklearn_confusion_matrix)
    
    cryo_tree_data = export_graphviz(cryo_decision_tree)
    graph = graphviz.Source(cryo_tree_data)
    graph.render("cryo_decision_tree")
    graph.view()  