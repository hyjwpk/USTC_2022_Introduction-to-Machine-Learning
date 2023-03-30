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
            regularization = self.gamma * \
                np.diag(
                    np.ones((self.beta.shape[0], 1)).flatten()) / self.beta.shape[0]
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
        return result


class SVM:
    def __init__(self, dim, C=1, plot=True):
        """
        You can add some other parameters, which I think is not necessary
        """
        self.dim = dim
        self.C = C  # 优化目标中损失函数前面的参数，设置太大会导致模型震荡难以收敛
        self.plot = plot  # 是否输出图像

    def grad(self, X, y):
        temp = 1 - (X @ self.w) * y
        # 在max内部的值大于0时，才计算对应项的导数
        flag = np.array(np.frompyfunc(lambda x: 1.0 if x >
                        0 else 0.0, 1, 1)(temp), np.float64)
        return self.w - (self.C * (y.T @ (flag * X))).T

    def loss(self, X, y):
        temp = 1 - (X @ self.w) * y
        flag = np.array(np.frompyfunc(lambda x: 1.0 if x >
                        0 else 0.0, 1, 1)(temp), np.float64)
        return (0.5 * (self.w.T @ self.w) + self.C * np.sum((flag * temp).flatten()))[0][0]

    def iteration(self, X, y):
        # 步长较大时会反复震荡，因此将步长设为0.00001
        return 0.00001 * self.grad(X, y)

    def fit(self, X, y, tol=1e-2, max_iter=1e2):
        """
        Fit the coefficients via your methods
        """
        self.w = np.zeros((self.dim, 1))
        loss_table = []
        before = after = 0
        iter = 0
        # 通过前后两次迭代的损失函数值的差值来判断是否收敛
        while iter == 0 or (iter < max_iter and np.absolute(before - after) > tol):
            before = after
            self.w -= self.iteration(X, y)
            after = self.loss(X, y)
            iter += 1
            loss_table.append(after)
        if self.plot:
            from matplotlib import pyplot as plt
            plt.title('loss curve of training')
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.plot(np.arange(1, iter + 1), np.array(loss_table).flatten())
            plt.show()

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        # 计算预测值，并映射到±1
        return np.array(np.frompyfunc(lambda x: x, 1, 1)(X @ self.w), np.float64)


def predictOvR(X, model1, model2, model3, model4):
    result1 = model1.predict(X)
    result2 = model2.predict(X)
    result3 = model3.predict(X)
    result4 = model4.predict(X)

    def predict(x, result1=result1, result2=result2, result3=result3, result4=result4):
        y1 = result1[x]
        y2 = result2[x]
        y3 = result3[x]
        y4 = result4[x]
        result = 0
        max = 0
        if y1 > max:
            max = y1
            result = 0
        if y2 > max:
            max = y2
            result = 1
        if y3 > max:
            max = y3
            result = 2
        if y4 > max:
            max = y4
            result = 3
        return result
    result = np.frompyfunc(predict, 1, 1)(range(X.shape[0])).reshape(-1, 1)
    result = np.array(result, dtype=np.float64)
    return result
