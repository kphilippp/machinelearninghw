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
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    
    partitions = {}

    for unique_num in np.unique(x):
        partitions[unique_num] = []

    for index, value in enumerate(x):
        partitions[value].append(index)

    return partitions

def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    
    unique_labels = np.unique(y)
    total_labels = len(y)

    label_counts = {key: 0 for key in unique_labels}
    for i in y:
        label_counts[i] += 1

    probabilities = {}
    for label in unique_labels:
        probabilities[label] = (label_counts[label]/total_labels)

    probabilities = np.array(list(probabilities.values()))
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE

    base_entropy = entropy(y)

    partitions = partition(x)

    conditional_entropy = 0.0
    for unique_x in np.unique(x):
        y_vals = []
        for each_index in partitions[unique_x]:
            y_vals.append(y[each_index])
        
        subset_entropy = entropy(y_vals)
        weight = (len(partitions[unique_x]) / len(x))
        conditional_entropy += weight * subset_entropy

    return base_entropy - conditional_entropy

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

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    
    # PURITY CHECK
    # if all labels the same, then no need to split just return the label as leaf node
    if len(set(y)) == 1:
        return y[0]

    # IF THE ATTR_VALUE_PAIRS HAVENT EVEN BEEN INITIALIZED
    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for attr in range(x.shape[1]):
            for val in np.unique(x[:, attr]):
                attribute_value_pairs.append((attr, val))

    # EXHAUSTED ATTRIBUTES OR MAX DEPTH REACHED
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        #return majority leaf node
        return np.bincount(y).argmax()



    # FIND NEXT BEST NODE TO SPLIT ON THEN
    # your gonna go through your attr val pairs
    # for each, you wanna get a vector of 0's and 1's where 0 means the value in the column != the val in the pair and 1 otherwise
    # you wanna pass that into mutual information and that will give you a "gain", which you should keep track of as well as the actual pair that gives you that
    best_gain = -1
    best_pair = None

    for pair in attribute_value_pairs:
        attr, val = pair
        test = (x[:, attr] == val).astype(int)
        gain = mutual_information(test, y)
        if (gain > best_gain):
            best_gain = gain
            best_pair = pair
        
    if best_pair is None or best_gain <= 0:
        return np.bicount(y).argmax()


    # REMOVE THAT NODE FROM LIST SO WE DONT USE IT AGAIN
    new_attr_val_pairs = []
    for pair in attribute_value_pairs:
        if pair != best_pair:
            new_attr_val_pairs.append(pair)


    # NOW YOU NEED TO RECURSIVELY CALL ID3
    # BUT YOU NEED TO ONLY PASS IN THE DATA OF THE RESPECTIVE ATTRIBUTES OF THE BRANCH
    # so, if the node means an attr is false, then only the data that has that attribute false should be passed in.
    # same thing for the y_values and the attribute list should be the new attribute list
    # increase the depth also when passing in

    best_attr, best_val = best_pair
    
    true_x = []
    true_y = []
    false_x = []
    false_y = []
    for index, val in enumerate((x[:, best_attr] == best_val).astype(int)):
        if val == 1:
            true_x.append(x[index])
            true_y.append(y[index])
        else:
            false_x.append(x[index])
            false_y.append(y[index])
        
    subtree_left = id3(np.array(true_x), np.array(true_y), new_attr_val_pairs, depth + 1, max_depth)
    subtree_right = id3(np.array(false_x), np.array(false_y), new_attr_val_pairs, depth + 1, max_depth)

    tree = {}
    tree[(best_attr, best_val, True)] = subtree_left
    tree[(best_attr, best_val, False)] = subtree_right

    return tree

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE

    if not isinstance(tree, dict):
        return tree

    for (attr, val, branch), subtree in tree.items():
        if (x[attr] == val) == branch:
            return predict_example(x, subtree)

    return None

def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    error = np.mean(y_true != y_pred)
    return error

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

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true ==1) & (y_pred ==1))
    TN = np.sum((y_true ==0) & (y_pred ==0))
    FP = np.sum((y_true ==0) & (y_pred ==1))
    FN = np.sum((y_true ==1) & (y_pred ==0))

    return TP, FP, FN, TN

if __name__ == '__main__':

    # A ---------------------------------------------------------------------------------------------------------------------------

    filenames = ["monks-1","monks-2","monks-3"]

    for file in range(3):
        # Load the training data
        M = np.genfromtxt(f'./{filenames[file]}.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]

        # Load the test data
        M = np.genfromtxt(f'./{filenames[file]}.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]

        # train tree for depth 1 to 10
        # get training error
        # get testing error
        test_errors = []
        train_errors = []
        for d in range(1, 11):
            # Learn a decision tree of depth 
            decision_tree = id3(Xtrn, ytrn, max_depth=d)
            #visualize(decision_tree)

            # Compute predictions and error on the training set
            y_pred_train = [predict_example(x, decision_tree) for x in Xtrn]
            trn_err = compute_error(ytrn, y_pred_train)
            train_errors.append(trn_err)
            
            # Compute predictions and error on the test set
            y_pred_test = [predict_example(x, decision_tree) for x in Xtst]
            tst_err = compute_error(ytst, y_pred_test)
            test_errors.append(tst_err)
        
       

        plt.figure()
        plt.plot(list(range(1,11)), train_errors, marker='o', label='Training Error')
        plt.plot(list(range(1,11)), test_errors, marker='x', label='Test Error')
        plt.xlabel('Tree Depth')
        plt.ylabel('Error')
        plt.title(f'Learning Curves for {filenames[file]}')
        plt.legend()
        plt.show()

    # B ---------------------------------------------------------------------------------------------------------------------------------

    M = np.genfromtxt(f'./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt(f'./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    



    print("Part B\nDepth 1\n\n")
    decision_tree = id3(Xtrn, ytrn, max_depth = 1)
    visualize(decision_tree)
    y_pred = np.array([predict_example(x, decision_tree) for x in Xtst])
    TP, FP, FN, TN = confusion_matrix(ytst, y_pred)
    print("\nConfusion Matrix for Depth: 1\n")
    print("             Positive    Negative ")
    print(f" Positive\t{TP}\t    {FN}    ")
    print(f" Negative\t{FP}\t    {TN}    ")

    scikitTree1 = DecisionTreeClassifier(max_depth=1)
    scikitTree1.fit(Xtrn, ytrn)

    y_preds = scikitTree1.predict(Xtst)
    TP, FP, FN, TN = confusion_matrix(ytst, y_pred)
    print("\nScikit-Learn's Confusion Matrix for Depth: 1\n")
    print("             Positive    Negative ")
    print(f" Positive\t{TP}\t    {FN}    ")
    print(f" Negative\t{FP}\t    {TN}    ")


    print("Depth 2\n\n")
    decision_tree = id3(Xtrn, ytrn, max_depth = 2)
    visualize(decision_tree)
    y_pred = np.array([predict_example(x, decision_tree) for x in Xtst])
    TP, FP, FN, TN = confusion_matrix(ytst, y_pred)
    print("\nConfusion Matrix for Depth: 2\n")
    print("             Positive    Negative ")
    print(f" Positive\t{TP}\t    {FN}    ")
    print(f" Negative\t{FP}\t    {TN}    ")

    scikitTree2 = DecisionTreeClassifier(max_depth=2)
    scikitTree2.fit(Xtrn, ytrn)


    y_preds = scikitTree2.predict(Xtst)
    TP, FP, FN, TN = confusion_matrix(ytst, y_pred)
    print("\nScikit-Learn's Confusion Matrix for Depth: 2\n")
    print("             Positive    Negative ")
    print(f" Positive\t{TP}\t    {FN}    ")
    print(f" Negative\t{FP}\t    {TN}    ")

    # C ---------------------------------------------------------------------------------------------------------------------------------

    scikitTree = DecisionTreeClassifier()
    scikitTree.fit(Xtrn, ytrn)
    dot_data = export_graphviz(scikitTree)
    graph = graphviz.Source(dot_data)
    graph.render("monks1_decision_tree")

    # D ---------------------------------------------------------------------------------------------------------------------------------

    # Define converters for the class labels:
    label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    converters = {4: lambda s: label_map[s.decode('utf-8')]}  # Assuming the class label is column index 4

    data = np.genfromtxt('./iris/iris.data', delimiter=',', converters=converters, dtype=float)
    data = np.array([list(row) for row in data])
    y = data[:, 4].astype(int)
    X = data[:, :4]

    X_discrete = np.copy(X)
    for i in range(X.shape[1]):
        col_mean = np.mean(X[:, i])
        X_discrete[:, i] = (X[:, i] > col_mean).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X_discrete, y, random_state=0, train_size = .75)

    tree1 = id3(X_train, y_train, max_depth=1)
    print("\nDecision Tree for Iris Dataset Max Depth 1\n")
    visualize(tree1)

    y_pred = np.array([predict_example(x, tree1) for x in X_test])
    TP, FP, FN, TN = confusion_matrix(y_test, y_pred)

    print("\n\nConfusion Matrix for Max Depth 1\n")
    print("             Positive    Negative ")
    print(f" Positive\t{TP}\t    {FN}    ")
    print(f" Negative\t{FP}\t    {TN}    ")


    tree2 = id3(X_train, y_train, max_depth=2)
    print("\n\n\nDecision Tree for Iris Dataset Max Depth 2\n")
    visualize(tree2)

    y_pred = np.array([predict_example(x, tree2) for x in X_test])
    TP, FP, FN, TN = confusion_matrix(y_test, y_pred)

    print("\n\nConfusion Matrix for Max Depth 2\n")
    print("             Positive    Negative ")
    print(f" Positive\t{TP}\t    {FN}    ")
    print(f" Negative\t{FP}\t    {TN}    ")
