# logistic_regression.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Nikhilesh Prabhakar (nikhilesh.prabhakar@utdallas.edu),
# Athresh Karanam (athresh.karanam@utdallas.edu),
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing a simple version of the 
# Logistic Regression algorithm. Insert your code into the various functions 
# that have the comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. 




import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle 


class SimpleLogisiticRegression():
    """
    A simple Logisitc Regression Model which uses a fixed learning rate
    and Gradient Ascent to update the model weights
    """
    def __init__(self):
        self.w = []
        pass

        
    def initialize_weights(self, num_features):
        #DO NOT MODIFY THIS FUNCTION
        w = np.zeros((num_features))
        return w


    def compute_loss(self,  X, y):
        """
        Compute binary cross-entropy loss for given model weights, features, and label.
        :param w: model weights
        :param X: features
        :param y: label
        :return: loss   
        """

        # You have w weights, X features, and y label for each example.
        # Compute z which is the dot product. Since X is 2D and we have many examples, z will be an array of z's
        # Then we should get our prediction for each z 
        # clipping to avoid the log 0 problem
        # To compute loss
        #   - you need to compute - (1/N) * SUM 1toN( true label * log (predicted label) + (1- truelabel) * (log( 1- predicted label )) )
        #   - n

        # INSERT YOUR CODE HERE
        z = np.dot(X, self.w)
        predictions = self.sigmoid(z)

        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss

        
    
    def sigmoid(self, val):

        """
        Implement sigmoid function
        :param val: Input value (float or np.array)
        :return: sigmoid(Input value)
        """

        # The input is an arrya of z's which is the dot product of the weights and the features which is an array of
        # The returned value is an array of predictions because np.exp vectorizes

        # INSERT YOUR CODE HERE
        return 1 / ( 1 + np.exp(-val) )


    def gradient_ascent(self, w, X, y, lr):

        """
        Perform one step of gradient ascent to update current model weights. 
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        Update the model weights
        """

        # @param X: already biased features

        # what needs to happen
        # calculate z vector resulting from sigmoid which is a vector of our predictions
        # calculate error
        # compute gradient
        # update weights

        # INSERT YOUR CODE HERE

        predictions = self.sigmoid(np.dot(X, w))   # p subscript i
        error = y - predictions                    # error 
        gradient = np.dot(X.T, error)              # 
        new_weights = w + ( lr * gradient )
        return new_weights



    def fit(self,X, y, lr=0.1, iters=100, recompute=True):
        """
        Main training loop that takes initial model weights and updates them using gradient descent
        :param w: model weights
        :param X: features
        :param y: label
        :param lr: learning rate
        :param recompute: Used to reinitialize weights to 0s. If false, it uses the existing weights Default True

        NOTE: Since we are using a single weight vector for gradient ascent and not using 
        a bias term we would need to append a column of 1's to the train set (X)

        """
        # ex:
        # param X (no bias) = [[000001],[000010],[000100],...]
        # param y = [1, 0, 1, 1, 0, 0, 0, 0 ...]

        # what should happen 
        # we need to put a bias on our features and then initialize the weights
        # then we can start gradient ascent with that for the iteration count


        # INSERT YOUR CODE HERE
        ones_col = np.ones((X.shape[0], 1))
        X_biased = np.hstack((ones_col, X))
        if(recompute):
            self.w = self.initialize_weights(X_biased.shape[1])
        for _ in range(iters):
            # INSERT YOUR CODE HERE
            self.w = self.gradient_ascent(self.w, X_biased, y, lr)


    def predict_example(self, w, x):
        """
        Predicts the classification label for a single example x using the sigmoid function and model weights for a binary class example
        :param w: model weights
        :param x: example to predict
        :return: predicted label for x
        """
        # INSERT YOUR CODE HERE
        x_bias = np.insert(x, 0, 1)
        probability = self.sigmoid(np.dot(w, x_bias))
        if probability >= 0.5:
            return 1
        else:
            return 0



    def compute_error(self, y_true, y_pred):
        """
        Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
        :param y_true: true label
        :param y_pred: predicted label
        :return: error rate = (1/n) * sum(y_true!=y_pred)
        """
        # INSERT YOUR CODE HERE
        return np.mean(y_true != y_pred)




if __name__ == '__main__':

    # Load the training data
    M = np.genfromtxt('./data/monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./data/monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    lr =  SimpleLogisiticRegression()
    
    lowestTestError = float('inf')
    lowestTrainError = float('inf')
    best_iters = None
    best_a = None

    # Part 1) Compute Train and Test Errors for different number of iterations and learning rates
    for iter in [10, 100,1000,10000]:
        for a in [0.01,0.1, 0.33]:
            # INSERT CODE HERE
            lr.fit(Xtrn, ytrn, a, iter, True)

            # after done training lets get errors
            # what needs to be done
            #  - format the features so we can have proper dot product with weights since we did that when training
            #  - compute z vector for example vector
            #  - compute p hat probabilities using the z vector and sigmoid function
            #  - using probs, classify the examples 
            #  - compute error between true labels and our labels


           
            train_predictions = []
            test_predictions = []
           

            for i in range(len(Xtrn)):
                train_predictions.append(lr.predict_example(lr.w, Xtrn[i]))
            for i in range(len(Xtst)):
                test_predictions.append(lr.predict_example(lr.w, Xtst[i]))


            train_error = lr.compute_error(ytrn, train_predictions)
            test_error = lr.compute_error(ytst, test_predictions)

            print("Iter #: {}, Learning Rate: {} Train Error: {}, Test Error: {}".format(iter, a, train_error, test_error))            
            if train_error < lowestTrainError:
                best_a = a
                best_iters = iter
                lowestTrainError = train_error
            if test_error < lowestTestError:
                lowestTestError = test_error


    print("Lowest Train Error: {:.4f}\nLowest Test Error: {:.4f}\nBest Learning Rate: {:.4f}Best Iteration Count: {:.4f}".format(lowestTrainError, lowestTestError, best_a, best_iters))



    # Part 2) Retrain Logistic Regression on the best parameters and store the model as a pickle file
    # what needs to be done 
    #   - retrain model on those params
    #   - use Pythons pickle module to save model 
    #   - Open a file (using a name that might include your NETID) in binary write mode.
    #   - Dump your model object (lr) into this file.

    # INSERT CODE HERE 
    lr.fit(Xtrn, ytrn, best_a, best_iters, True)

    # Code to store as pickle file
    netid = 'kxp210063'
    file_pi = open('{}_lr.obj'.format(netid), 'wb')  #Use your NETID
    pickle.dump(lr, file_pi)



    # Part 3) Compare your model's performance to scikit-learn's LR model's default parameters 
    # what it should do
    #   - initialize Log Regression Object
    #   - train model
    #   - scikit predictions
    #   - calculate error for both training and testing
    # INSERT CODE HERE
    scikit_logreg = LogisticRegression()
    scikit_logreg.fit(Xtrn, ytrn)

    sk_train_preds = scikit_logreg.predict(Xtrn)
    sk_test_preds = scikit_logreg.predict(Xtst)

    sk_train_error = np.mean(sk_train_preds != ytrn)
    sk_test_error = np.mean(sk_test_preds != ytst)

    print("Scikit-learn Logistic Regression\n Train Error: {:.4f}, Test Error: {:.4f}"
        .format(sk_train_error, sk_test_error))



    # Part 4) Plot curves on train and test loss for different learning rates. Using recompute=False might help
    # what you need to do
    #   - for each learning rate, you need to reinitialize the model, new weights with small iterations (soley to reset weights)
    #   - create empty lists to store train loss, test loss, and iterations
    #   - train the model incremenetally 
    #   - for each batch, train for a set number of iterations
    #   - compute the losses, when passing in data, make sure it has bias so it is all the same format
    #   - basically what this is doing is that at a certain iteration count, well compute the count and record the loss in that moment, we'll continue 


    for j, a in enumerate([0.01,0.1, 0.33]):
        lr.fit(Xtrn, ytrn, lr=a, iters=1)
        # INSERT CODE HERE

        # Prepare the biased features
        Xtrn_bias = np.hstack((np.ones((Xtrn.shape[0], 1)), Xtrn))
        Xtst_bias = np.hstack((np.ones((Xtst.shape[0], 1)), Xtst))

        # Record initial losses at iteration 0
        iteration_v = [0]
        train_loss_v = [lr.compute_loss(Xtrn_bias, ytrn)]
        test_loss_v = [lr.compute_loss(Xtst_bias, ytst)]


        for i in range(10):
            lr.fit(Xtrn, ytrn, lr=a, iters=100,recompute=False)
            # INSERT CODE HERE

            train_loss = lr.compute_loss(Xtrn_bias, ytrn)
            test_loss = lr.compute_loss(Xtst_bias, ytst)

            iteration_v.append(100*(i+1))
            train_loss_v.append(train_loss)
            test_loss_v.append(test_loss)



        plt.plot(iteration_v, train_loss_v, label=f'Train Loss: {a}')
        plt.plot(iteration_v, test_loss_v, label=f'Test Loss: {a}')   
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training vs Test Loss Curves = ' + str(a))
        plt.legend()
        plt.show()




