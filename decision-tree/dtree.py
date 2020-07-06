import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from collections import Counter


class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        return self.prediction


class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss  # loss function; either np.std or gini

    def bestsplit(self, X, y):
        best = dict()
        best['col'] = -1
        best['split'] = -1
        best['loss'] = self.loss(y)
        k = 11
        p = X.shape[1]
        n = X.shape[0]
        for i in range(p):
            Xcol = X[:, i]
            candidates = Xcol
            if len(Xcol) > k:
                candidates = np.random.choice(Xcol, k, replace=False)
            for split in candidates:

                yl = y[Xcol <= split]
                yr = y[Xcol > split]

                if (len(yl) == 0) or (len(yr) == 0):
                    continue
                l = ((len(yl)*self.loss(yl))+(len(yr)*self.loss(yr)))/n

                if l == 0:
                    return i, split
                if l < best['loss']:
                    best['col'] = i
                    best['loss'] = l
                    best['split'] = split
        return best['col'], best['split']

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """

        self.root = self.fit_(X, y)


    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """

        if len(X) < self.min_samples_leaf:
            return self.create_leaf(y)
        col, split = self.bestsplit(X, y)

        if col == -1:
            return self.create_leaf(y)

        Xleft = X[X[:, col] <= split]
        Xright = X[X[:, col] > split]
        yleft = y[X[:, col] <= split]
        yright = y[X[:, col] > split]

        lchild = self.fit_(Xleft, yleft)
        rchild = self.fit_(Xright, yright)

        return DecisionNode(col, split, lchild, rchild)


    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """

        retArray = np.array([])
        for row in X_test:
            retArray = np.append(retArray, [self.root.predict(row)])
        return retArray


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.std)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"

        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


def gini(array):
    "Return the gini impurity score for values in y"
    """Calculate the Gini coefficient of a numpy array."""
    counter_values = Counter(array)
    total = len(array)
    retValue = 1
    for key in counter_values:
        retValue = retValue - (counter_values[key]/total)**2
    return retValue


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)
    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)


    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, stats.mode(y)[0])
