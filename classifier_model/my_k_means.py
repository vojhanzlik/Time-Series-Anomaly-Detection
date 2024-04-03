import warnings

import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2
from sklearn.cluster import KMeans


def classify_kmeans(data, n_clusters):
    """
    Function that performs the Kmeans clustering algorithm and returns the means and
    labels of given points

    Args:
        data : list[]
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier_class = KMeans(n_clusters=n_clusters, n_init=10)
        classifier_class.fit(data)
        centroids = classifier_class.cluster_centers_
        labels = classifier_class.labels_
        clusters = [[i for i, x in enumerate(labels) if x == num] for num in range(max(labels) + 1)]
        return centroids, clusters

def get_cluster_params(means, points, labels):
    
    ##assert len(means) == len(labels), "Number of centroids does not match number of classes"
    clusters_params = [[] for i in range(len(means))]
    #
    if not isinstance(points, np.ndarray):
            points = np.array(points)
    for i in range(len(means)):
        pts = points[labels[i]].T
        cov_mat = np.cov(pts)
        clusters_params[i] = (means[i], cov_mat)
    #    
    return clusters_params

def upscale(array):
    return array[np.newaxis, :] if len(np.shape(array)) < 2 else array

class GaussianClassifier():
    def __init__(self, cluster_params) -> None:
        self.cluster_params = cluster_params
    def classify_to_cluster(self, sample, confidence = 0.95):
        if not isinstance(sample, np.ndarray):
            sample = np.array(sample)
        probs = np.zeros(len(self.cluster_params))
        for i in range(len(self.cluster_params)):
            mu, covar = self.cluster_params[i]
            mu = upscale(mu)
            n_dim = len(mu)
            chi2value = chi2.ppf(confidence, df = n_dim)
            difference_vector = sample - mu
            difference_vector = upscale(difference_vector)
            mahalonobis_distance = difference_vector @ inv(covar) @ difference_vector.T
            probs[i] = mahalonobis_distance <= chi2value
        return probs
    
    def classify(self, sample, confidence = 0.95):
        return np.sum(self.classify_to_cluster(sample, confidence)) > 0

