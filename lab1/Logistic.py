import pandas as pd
import numpy as np


class LogisticRegression:

    def __init__(self, penalty="l2", gamma=1, fit_intercept=True, plot=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.penalty = penalty
        self.gamma = gamma
        self.plot = plot

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        return 1.0 / (1.0 + np.exp(-x))

    def grad(self, X, y):
        regularization = 0
        if self.penalty == 'l1':
            regularization = self.gamma * \
                np.sign(self.beta) / self.beta.shape[0]
        elif self.penalty == 'l2':
            regularization = self.gamma * self.beta / self.beta.shape[0]
        return (X.T @ (self.sigmoid(X @ self.beta) - y)) + regularization

    def hessian(self, X):
        regularization = 0
        if self.penalty == 'l1':
            regularization = 0
        elif self.penalty == 'l2':
            regularization = self.gamma * np.diag(np.ones((self.beta.shape[0],1)).flatten()) / self.beta.shape[0]
        p1 = self.sigmoid(X @ self.beta).flatten()
        return (X.T @ np.diag(p1) @ np.diag(1-p1) @ X) + regularization

    def loss(self, X, y):
        regularization = 0
        if self.penalty == 'l1':
            regularization = self.gamma * \
                np.sum(np.abs(self.beta)) / self.beta.shape[0]
        elif self.penalty == 'l2':
            regularization = self.gamma * \
                (self.beta.T @ self.beta) / (2*self.beta.shape[0])
        return (np.sum(np.log(1 + np.exp(X @ self.beta))) - (X @ self.beta).T @ y + regularization)[0][0]

    def iteration(self, X, y, lr=None, pattern='newton'):
        if pattern == 'newton':
            return (1 if lr == None else lr) * (np.linalg.pinv(self.hessian(X)) @ self.grad(X, y))
        elif pattern == 'gd':
            return (0.005 if lr == None else lr) * self.grad(X, y)

    def fit(self, X, y, lr=None, tol=1e-7, max_iter=1e7, pattern='gd'):
        """
        Fit the regression coefficients via gradient descent or other methods 
        """
        loss_table = []
        self.beta = np.zeros((X.shape[1], 1))
        before = self.loss(X, y)
        self.beta -= self.iteration(X, y, lr, pattern)
        after = self.loss(X, y)
        loss_table.append(before)
        loss_table.append(after)
        iter = 1
        while iter < max_iter and np.absolute(before-after) > tol:
            before = after
            self.beta -= self.iteration(X, y, lr, pattern)
            after = self.loss(X, y)
            iter += 1
            loss_table.append(after)
        times = np.arange(0, iter + 1)
        loss_table = np.array(loss_table).flatten()
        if self.plot:
            from matplotlib import pyplot as plt
            plt.title('loss curve of training')
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.plot(times, loss_table)
            plt.show()

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        result = self.sigmoid(X @ self.beta)
        for i in range(result.shape[0]):
            if result[i] > 0.5:
                result[i] = 1
            else:
                result[i] = 0
        return result
