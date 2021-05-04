# Important note: you do not have to modify this file for your homework.

import numpy as np


def train_and_predict_logreg(train_matrix, train_labels, test_matrix, learning_rate):
    """Train a logistic regression model using gradient descent with the given learning rate
    and predict the resulting labels on a test set.

    Args:
        train_matrix: A numpy array containing the word counts for the train set
        train_labels: A numpy array containing the spam or not spam labels for the train set
        test_matrix: A numpy array containing the word counts for the test set
        learning_rate: The learning rate to use for gradient descent

    Return:
        The predicted labels for each message
    """
    theta = logreg_train(train_matrix, train_labels, learning_rate)
    return logreg_predict(theta, test_matrix)


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad

def calc_loss(X, Y, theta):
    probs = 1. / (1 + np.exp(-X.dot(theta)))
    return np.mean(-Y * np.log(probs) - (1 - Y) * np.log(1 - probs))

def logreg_train(X, Y, learning_rate=0.00001, max_iter=1000, verbose=False):
    """Train a logistic regression model using gradient descent with the given learning rate
    and return the learned parameters

    Args:
        X: the design matrix where each row corresponds to a data point and 
           each column corresponds to a feature
        Y: the vector containing labels (0 or 1) for each data point
        learning_rate: the learning rate for GD
        max_iter: maximum number of iterations of GD

    Returns:
        theta: the final logistic regression parameters found by GD
    """
    theta = np.zeros(X.shape[1])
    loss = calc_loss(X, Y, theta)

    i = 0
    while True:
        i += 1
        prev_loss = loss
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        loss = calc_loss(X, Y, theta)
        if verbose:
            print('[lr=%f] Finished %d iterations, current loss is %f' % (learning_rate, i, loss))
        if np.isnan(loss):
            return theta
        if i >= max_iter:
            return theta

def logreg_predict(theta, X):
    """A function that takes in a theta parameter and design matrix X
    and returns the predictions.

    Args:
        theta: the theta parameter for Logistic Regression
        X: the design matrix where each row corresponds to a data point and 
           each column corresponds to a feature

    returns:
        Y_hat: the vector containing the predictions
    """
    probs = 1. / (1 + np.exp(-X.dot(theta)))
    return probs > 0.5

