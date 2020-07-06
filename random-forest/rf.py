import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from dtree import ClassifierTree621, RegressionTree621

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.uniqueEl = 0
        self.oob_idxs = []


    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """

        self.trees = []
        self.uniqueEl = len(np.unique(y))
        for i in range(self.n_estimators):

            X_, y_ = resample(X, y, n_samples=len(X))

            if self.__class__ == RandomForestRegressor621:
                self.trees.append(RegressionTree621().fit(X_, y_, self.max_features, self.min_samples_leaf))
            if self.__class__ == RandomForestClassifier621:
                self.trees.append(ClassifierTree621(self.min_samples_leaf).fit(X_, y_, self.max_features, self.min_samples_leaf))

            if self.oob_score:
                Z = np.column_stack((X, y))
                Z_ = np.column_stack((X_, y_))

                ind = []
                for i in range(Z.shape[0]):
                    if (any((Z_[:] == Z[i]).all(1))) == False:
                        ind.append(i)
                self.oob_idxs.append(ind)

        if self.oob_score:
            self.oob_score_ = self.compute_oob_score(X, y)


class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        leafArray = np.array([])

        for row in X_test:
            num = 0
            deno = 0
            for tree in self.trees:
                leaf = tree.leaf(row)
                num = num+(leaf.prediction*leaf.n)
                deno = deno+leaf.n
            leafArray = np.append(leafArray, [num/deno])
        return leafArray

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)

    def compute_oob_score(self, X, y):
        n = X.shape[0]
        oob_counts = np.zeros(n)
        oob_preds = np.zeros(n)
        j=0
        for tree in self.trees:
            leafsizes = np.zeros(n)
            leafpredictions = np.zeros(n)
            for i in self.oob_idxs[j]:
                leaf = tree.leaf(X[i])
                leafsizes[i] = leaf.n
                leafpredictions[i] = leaf.prediction
            j = j+1
            oob_preds += np.multiply(leafsizes, leafpredictions)
            oob_counts += leafsizes
        oob_avg_preds = oob_preds[oob_counts > 0]/oob_counts[oob_counts > 0]
        return r2_score(y[oob_counts > 0], oob_avg_preds)


class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []

    def predict(self, X_test) -> np.ndarray:
        leafArray = np.array([])
        for row in X_test:
            counts = np.zeros(self.uniqueEl)
            for t in self.trees:
                leaf = t.leaf(row)
                counts[leaf.prediction] += leaf.n
            leafArray = np.append(leafArray, [np.argmax(counts)])
        return leafArray


    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def compute_oob_score(self, X, y):
        j = 0
        n = X.shape[0]
        k = self.uniqueEl

        oob_counts = np.zeros(n)
        oob_preds = np.zeros((n, k))

        for tree in self.trees:
            leafsizes = np.array([])
            tpred = np.array([])
            for i in self.oob_idxs[j]:
                leaf = tree.leaf(X[i])
                leafsizes = np.append(leafsizes, [leaf.n])
                tpred = np.append(tpred, [leaf.prediction])
            oob_preds[self.oob_idxs[j], tpred.astype(int)] += leafsizes
            oob_counts[self.oob_idxs[j]] += 1
            j = j+1

        oob_votes = np.array([])
        for i in np.where(oob_counts > 0)[0]:
            oob_votes = np.append(oob_votes, [np.argmax(oob_preds[i])])
        return accuracy_score(y[oob_counts > 0], oob_votes)
