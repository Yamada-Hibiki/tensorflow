import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
# Scale function makes it into range from -1 to 1, digits.data ==> all of the features
data = scale(digits.data)
# Label
y = digits.target

# k = len(np.unique(y)  # Dynamic way
k = 10
# decide the amount of instances and features
samples, features = data.shape


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


# number of centroids, choose the initial centroids randomly,
# It generates the centroids 10 times
clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)

