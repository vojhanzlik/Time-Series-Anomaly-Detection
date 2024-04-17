from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging
from tslearn.utils import to_time_series_dataset
from scipy.stats import chi2
import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime
from metrics.dtw_barycenter import compute_dtw_dists, get_distance
from visual import plot_samples
from support.my_k_means import classify_kmeans, get_cluster_params, GaussianClassifier
from utils import transform_pd_to_npy


class featureClassifier():
    def __init__(self, treshold = 0.995, n_clusters = None, method = [1,5]) -> None:
        self.barycenter_dtw = None
        self.barycenter_euclid = None
        self.treshhold = treshold
        self.classifier = None
        self.n_clusters = n_clusters
        self.method = method

    def naive_fit(self, training_signals, n_clusters = 1, method = [1, 5], vis = False):
        self.method = method
        correct_signals_np_array = [transform_pd_to_npy(sample[0]) for sample in training_signals if sample[1] == True]
        wrong_signals_np_array = [transform_pd_to_npy(sample[0]) for sample in training_signals if sample[1] == False]
        training_set = to_time_series_dataset(correct_signals_np_array)
        self.barycenter_dtw = [dtw_barycenter_averaging(training_set)]
        self.barycenter_euclid = [euclidean_barycenter(training_set)]
        self.n_clusters = n_clusters
        self.method = method
        c, w, b, eb = compute_dtw_dists(correct_signals_np_array,
                                        wrong_signals_np_array,
                                        self.barycenter_dtw,
                                        self.barycenter_euclid,
                                        method = method)
        if vis:
            plot_samples(c, w, b, eb, method)
        print(c)
        data_to_fit = np.array(c)
        centroids, clusters = classify_kmeans(data_to_fit, n_clusters)
        cluster_params = get_cluster_params(centroids, data_to_fit, clusters)
        self.classifier = GaussianClassifier(cluster_params)
        print(self.classifier)
    


    def known_cluster_fit(self, training_signals, n_clusters = 4, method = [1, 5], vis = True, keep_barycenters = False):
        if self.method != method:
            keep_barycenters = False
            raise Exception(f"New method {method} does not equal the state of the art method {self.method}.")
        if not keep_barycenters:
            self.barycenter_dtw = []
            self.barycenter_euclid = []
        correct_signals = [[sig.signal for i, sig in enumerate(training_signals) if ((i+1)%n_clusters == j and sig.label == True)] for j in range(n_clusters)]
        for i, cluster in enumerate(correct_signals):
            training_set = to_time_series_dataset(cluster)
            self.barycenter_dtw.append(dtw_barycenter_averaging(training_set))
            self.barycenter_euclid.append(euclidean_barycenter(training_set))
            print(f"INFO: Computed barycenter: {i + 1} out of: {n_clusters}.")
        corsig = [i.signal for i in training_signals if i.label == True]
        wrosig = [i.signal for i in training_signals if i.label == False]
        c, w, b, eb = compute_dtw_dists(corsig,
                                        wrosig,
                                        self.barycenter_dtw,
                                        self.barycenter_euclid,
                                        method = method)
        if vis:
            plot_samples(c, w, b, eb, method)
        data_to_fit = np.array(c).T
        centroids, clusters = classify_kmeans(data_to_fit, n_clusters)
        cluster_params = get_cluster_params(centroids, data_to_fit, clusters)
        print(cluster_params)
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
    def return_distance_to_closest(self, signal, partial = False):
        if not partial:
            dist = np.array([min([get_distance(signal, i, self.method[0]) for i in self.barycenter_dtw]),
                    min([get_distance(signal, i, self.method[1]) for i in self.barycenter_euclid])])
        else:
            signal_len = np.shape(signal)[0]
            dist = np.array([min([get_distance(signal, i[:min(signal_len, np.shape(i)[0]), :], self.method[0]) for i in self.barycenter_dtw]),
                    min([get_distance(signal, i[:min(signal_len, np.shape(i)[0]), :], self.method[1]) for i in self.barycenter_euclid])])
        return dist

    
    def predict_partial_signal(self, signal_):
        signal = transform_pd_to_npy(signal_)
        if self.classifier is not None:
            signal_len = np.shape(signal)[0]
            bar_len = max([np.shape(bar)[0] for bar in [*self.barycenter_dtw, *self.barycenter_euclid]])
            signal_metrics = self.return_distance_to_closest(signal, partial = True) * (bar_len/signal_len)
            #print(signal_metrics, end=" ")
            return self.classifier.classify(signal_metrics, self.treshhold)
            if signal_len < bar_len * 0.9:
                pass    
            else:
                return self.predict(signal_)
        else:
            print("ERROR: Cannot predict, classifier is not trained yet!")

    def predict(self, signal):
        signal_ = transform_pd_to_npy(signal)
        if self.classifier is not None:
            signal_metrics = self.return_distance_to_closest(signal_)
            return self.classifier.classify(signal_metrics, self.treshhold)
        else:
            print("ERROR: Cannot predict, classifier is not trained yet!")

    
    def load_params(self, path):
        with open(path, 'rb') as f:
            loaded_object = pickle.load(f)
        self.barycenter_dtw = loaded_object.barycenter_dtw
        self.barycenter_euclid = loaded_object.barycenter_euclid
        self.classifier = loaded_object.classifier
        self.treshhold = loaded_object.treshhold
        self.method = loaded_object.method
        self.n_clusters = loaded_object.n_clusters

    def save_params(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
