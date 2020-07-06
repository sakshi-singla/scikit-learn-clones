import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def normalize(X):  # creating standard variables here (u-x)/sigma
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:, j])
        s = np.std(X[:, j])
        X[:, j] = (X[:, j] - u) / s


def MSE(X, y, B, lmbda):
    e = (y - (X.dot(B)))
    return (e.T).dot(e)


def loss_gradient(X, y, B, lmbda):
    return -(X.T).dot(y-(X.dot(B)))


def loss_ridge(X, y, B, lmbda):
    b0 = np.mean(y)
    B[0] = b0
    e = (y - (X.dot(B)))
    return (e.T).dot(e) + lmbda*((B.T).dot(B))


def loss_gradient_ridge(X, y, B, lmbda):
    b0 = np.mean(y)
    B[0] = b0
    e = (y - (X.dot(B)))
    return -(X.T).dot(e) + lmbda*B


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_likelihood(X, y, B, lmbda):
    XB = np.dot(X,B)
    return -1*np.sum(y * XB - np.log(1 + np.exp(XB)))


def log_likelihood_gradient(X, y, B, lmbda):

    return -1*np.dot((X.T),y - sigmoid(np.dot(X,B)))


# NOT REQUIRED but to try to implement for fun
def L1_log_likelihood(X, y, B, lmbda):
    pass


# NOT REQUIRED but to try to implement for fun
def L1_log_likelihood_gradient(X, y, B, lmbda):
    """
    Must compute \beta_0 differently from \beta_i for i=1..p.
    \beta_0 is just the usual log-likelihood gradient
    # See https://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html
    # See https://stackoverflow.com/questions/38853370/matlab-regularized-logistic-regression-how-to-compute-gradient
    """
    pass


def minimize(X, y, loss, loss_gradient,
             eta=0.00001, lmbda=0.0,
             max_iter=1000, addB0=True,
             precision=0.00000001):
    "Here are various bits and pieces you might want"
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")

    if addB0:
        # add column of 1s to X
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])

    B = np.random.random_sample(size=(p+1, 1)) * 2 - 1  # make between [-1,1)

    prev_B = B
    cost = 9e99
    step = 0
    eps = 1e-5  # prevent division by 0

    prevLoss = loss(X, y, prev_B, lmbda)
    h = np.zeros((p+1, 1))
    for i in range(max_iter):
        # print(loss_gradient)
        h += loss_gradient(X, y, B, lmbda) ** 2
        adjusted_grad = loss_gradient(X, y, B, lmbda) / (eps + np.sqrt(h))
        # prev_B = B
        B = B - eta * adjusted_grad
        newLoss=loss(X, y, B, lmbda)

        if abs(newLoss-prevLoss) < precision:
            return B
        prevLoss = newLoss
    return B

class LinearRegression621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          MSE,
                          loss_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class RidgeRegression621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_ridge,
                          loss_gradient_ridge,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class LogisticRegression621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        prob = sigmoid(np.dot(X, self.B))
        retArr = np.where(prob >= 0.5, 1, 0)
        return retArr

    def fit(self, X, y):
        self.B = minimize(X, y,
                          log_likelihood,
                          log_likelihood_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


# NOT REQUIRED but to try to implement for fun
class LassoLogistic621:
    pass
