import numpy as np
import operator

# Euclidean Distance Caculator
def calcDist(x, y):
    return np.linalg.norm(x - y, axis=1)


def kmeans(X: np.ndarray, k: int, centroids=None, tolerance=1e-2):
    n, p = X.shape

    if centroids == 'kmeans++':
        CentroidIdxs = np.random.choice(range(n), 1)
        remainingIdxs = np.arange(n)
        for i in range(k - 1):
            remainingIdxs = remainingIdxs[~np.isin(remainingIdxs, CentroidIdxs)]
            dict1 = {}
            for i in remainingIdxs:
                distances = calcDist(X[i], X[CentroidIdxs])
                idxMin = np.argmin(distances)
                dict1[i] = distances[idxMin]
            CentroidIdxs = np.append(CentroidIdxs, max(dict1.items(), key=operator.itemgetter(1))[0])
        centroids = X[CentroidIdxs, :]
    else:
        idx = np.random.choice(range(n), k, replace=False)
        centroids = X[idx, :]

    clusters = []
    for i in range(0, k):
        clusters.append([])

    flag = True
    while flag:
        prevCentroids = centroids
        for j in range(n):
            distances = calcDist(X[j], centroids)
            cluster = np.argmin(distances)
            clusters[cluster].append(j)

        for i in range(k):
            cluster = clusters[i]
            newCentroid = np.sum(X[cluster, :], axis=0) / float(len(cluster))
            centroids[i] = newCentroid
        if np.linalg.norm(prevCentroids.flatten() - centroids.flatten()) <= tolerance:
            flag = False

    return centroids, clusters
