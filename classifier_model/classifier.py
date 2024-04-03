import pickle

import numpy as np
import pandas as pd
from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging
from tslearn.utils import to_time_series_dataset

from classifier_model.dtw_barycenter import get_distance, compute_dtw_dists
from classifier_model.my_k_means import classify_kmeans, get_cluster_params, GaussianClassifier
from classifier_model.plotting import plot_samples


class LabeledSignal:
    def __init__(self, signal, time, label):
        self.signal = signal
        self.time = time
        self.label = label

    def size(self):
        return np.shape(self.signal)

    def __repr__(self) -> str:
        df = pd.DataFrame(self.signal)
        return f"""Signal: {df}, label: {self.label}"""


class FeatureClassifier1:
    def __init__(self, treshold=0.995, n_clusters=None) -> None:
        self.barycenter_dtw = None
        self.barycenter_euclid = None
        self.treshhold = treshold
        self.classifier = None
        self.n_clusters = n_clusters
        self.method = None

    def naive_fit(self, training_signals, n_clusters=1, method=[1, 5], vis=False):
        self.method = method
        correct_signals_np_array = [sample.signal for sample in training_signals if sample.label == True]
        wrong_signals_np_array = [sample.signal for sample in training_signals if sample.label == False]
        training_set = to_time_series_dataset(correct_signals_np_array)
        self.barycenter_dtw = [dtw_barycenter_averaging(training_set)]
        self.barycenter_euclid = [euclidean_barycenter(training_set)]
        self.n_clusters = n_clusters
        self.method = method
        c, w, b, eb = compute_dtw_dists(correct_signals_np_array,
                                        wrong_signals_np_array,
                                        self.barycenter_dtw,
                                        self.barycenter_euclid,
                                        method=method)
        if vis:
            plot_samples(c, w, b, eb, method)
        data_to_fit = np.array(c).T
        centroids, clusters = classify_kmeans(data_to_fit, n_clusters)
        cluster_params = get_cluster_params(centroids, data_to_fit, clusters)
        self.classifier = GaussianClassifier(cluster_params)

    def set_treshold(self, treshold) -> None:
        self.treshhold = treshold

    def __repr__(self) -> str:
        return f"""
        Feature Classifier:
            Number of clusters: {self.n_clusters}
            Treshhold: {self.treshhold}
            Number of dtw barycenters: {len(self.barycenter_dtw)}
            Number of euclidean barycenters: {len(self.barycenter_euclid)}
            """

    def return_distance_to_closest(self, signal):
        return [min([get_distance(signal, i, self.method[0]) for i in self.barycenter_dtw]),
                min([get_distance(signal, i, self.method[1]) for i in self.barycenter_euclid])]

    def predict(self, signal):
        if self.classifier is not None:
            signal_metrics = self.return_distance_to_closest(signal)
            return self.classifier.classify(signal_metrics, self.treshhold)
        else:
            print("ERROR: Cannot predict, ml_classifier is not trained yet!")

    def load_params(self, path):
        with open(path, 'rb') as f:
            loaded_object = pickle.load(f, fix_imports=True)
        self.barycenter_dtw = loaded_object.barycenter_dtw
        self.barycenter_euclid = loaded_object.barycenter_euclid
        self.classifier = loaded_object.classifier
        self.treshhold = loaded_object.treshhold

        self.method = loaded_object.method
        self.n_clusters = loaded_object.n_clusters

    def save_params(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
