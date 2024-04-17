import numpy as np
import pandas as pd
from tslearn.utils import to_time_series_dataset
import matplotlib.pyplot as plt
from visual import plot_one_6dim_signal, plot_6dim_signal_dataset
from utils import incremental_mean_update, incremental_variance_update, transform_pd_to_npy, get_swich_points, set_signals_to_same_length




class deviationClassifier():
    def __init__(self, n_dim, sensitivity, anom_tresh = 12, learning_treshold = 0.95):
        self.n_dim = n_dim
        self.anomaly_treshold = anom_tresh
        self.sensitivity = sensitivity
        self.magnitude = 0
        self.learning_treshold = learning_treshold
        self.mean_ts = None
        self.std_ts = None
    
    def show_params(self):
        plot_one_6dim_signal(self.mean_ts, self.mean_ts, self.std_ts, sensitivity=self.sensitivity, anomaly_highlight = False)
    
    
    def fit_whole_supervised_dataset(self, training_dataset):
        correct_signals = [i[0] for i in training_dataset if i[1] == True]
        correct_ds = to_time_series_dataset(correct_signals)
        mean_ts = np.nanmean(correct_ds, axis = 0)
        std_ts = np.nanstd(correct_ds, axis = 0)
        self.mean_ts = transform_pd_to_npy(mean_ts)
        self.std_ts = transform_pd_to_npy(std_ts)
        self.magnitude = len(correct_signals)
    
    def fit_incremental_dataset(self, training_dataset, vis = False, sens = 3):
        correct_signals = [i[0] for i in training_dataset if i[1] == True]
        for i, sample in enumerate(correct_signals):
            sample_len = np.shape(sample)[0]
            if self.mean_ts is not None:
                self.mean_ts, sample_ = set_signals_to_same_length(self.mean_ts, sample)
                self.std_ts, sample_ = set_signals_to_same_length(self.std_ts, sample)
            else:
                self.mean_ts = transform_pd_to_npy(sample)
                self.std_ts = np.zeros_like(self.mean_ts)
                if vis:
                    sample_ = transform_pd_to_npy(sample)
                    #plot_one_6dim_signal(sample_, self.mean_ts, self.std_ts, sensitivity=sens, anomaly_highlight = False)
                continue
            mean_ts, std_ts, sample_ = (transform_pd_to_npy(self.mean_ts),
                                        transform_pd_to_npy(self.std_ts),
                                        transform_pd_to_npy(sample))
            mean_ts, std_ts, sample_ = mean_ts[:sample_len, :], std_ts[:sample_len, :], sample_[:sample_len, :]
            old_mean = mean_ts
            mean_ts = incremental_mean_update(mean_ts, sample_, self.magnitude)
            self.mean_ts[:sample_len, :] = mean_ts
            std_ts = incremental_variance_update(std_ts, sample_, self.magnitude, old_mean, mean_ts)
            self.std_ts[:sample_len, :] = std_ts
            sample_ = transform_pd_to_npy(sample_)
            if vis:
                plot_one_6dim_signal(sample_, self.mean_ts, self.std_ts, sensitivity=sens, anomaly_highlight = False, real_len = sample_len)
                #plot_6dim_signal_dataset([(transform_pd_to_npy(sig), True) for sig in correct_signals[:i]])
            self.magnitude += 1

        
    def predict_full_signal(self, sample, vis = True, save_fig = None, learn_from_signal = True):
        if self.magnitude == 0:
            print("WARN: The classifier is not trained, aborting classification!")
            if learn_from_signal:
                self.mean_ts = sample
                self.std_ts = np.zeros_like(sample)
                self.magnitude = 1
            return
        sample = transform_pd_to_npy(sample)
        sample_len = np.shape(sample)[0]
        mean_len = np.shape(self.mean_ts)[0]
        self.mean_ts, sample = set_signals_to_same_length(self.mean_ts, sample)
        diff = np.abs(sample[:sample_len, :] - self.mean_ts[:sample_len, :])
        anomalies = diff > np.abs(self.sensitivity * self.std_ts[:sample_len, :])
        anomalies_sum = np.sum(anomalies, axis=1)
        ls = get_swich_points(anomalies_sum)
        anomalies = transform_pd_to_npy(anomalies)
        anomalies_sw_points = [get_swich_points(anomalies[:, i]) for i in range(self.n_dim)]
        if vis:
            plot_one_6dim_signal(sample, self.mean_ts, self.std_ts,
                                 self.sensitivity, anomalies = ls, anomalies_all = anomalies_sw_points, save_fig=save_fig, real_len = sample_len)
        is_anomaly = np.sum(anomalies) > self.anomaly_treshold
        if not is_anomaly and learn_from_signal:
            old_mean = self.mean_ts
            self.mean_ts = incremental_mean_update(self.mean_ts, sample, self.magnitude)
            self.std_ts = incremental_variance_update(self.std_ts, sample, self.magnitude, old_mean, self.mean_ts)
            self.magnitude += 1
        return is_anomaly

    def predict_partial_signal(self, sample, vis = True, save_fig = None, learn_from_signal = True):
        if self.magnitude == 0:
            print("WARN: The classifier is not trained, aborting classification!")
            if learn_from_signal and self.learning_treshold < (sample_len/mean_len):
                self.mean_ts = sample
                self.std_ts = np.zeros_like(sample)
                self.magnitude = 1
            return
        sample_len = np.shape(sample)[0]
        mean_len = np.shape(self.mean_ts)[0]
        self.mean_ts, sample = set_signals_to_same_length(self.mean_ts, sample)
        diff = np.abs(sample[:sample_len, :] - self.mean_ts[:sample_len, :])
        anomalies = diff > np.abs(self.sensitivity * self.std_ts[:sample_len, :])
        anomalies_sum = np.sum(anomalies, axis=1)
        ls = get_swich_points(anomalies_sum)
        anomalies = transform_pd_to_npy(anomalies)
        anomalies_sw_points = [get_swich_points(anomalies[:, i]) for i in range(self.n_dim)]
        if vis:
            sample = transform_pd_to_npy(sample)
            plot_one_6dim_signal(sample[:sample_len, :], self.mean_ts[:sample_len, :], self.std_ts[:sample_len, :],
                                 self.sensitivity, anomalies = ls, anomalies_all = anomalies_sw_points, save_fig=save_fig, real_len = sample_len)
        is_anomaly = np.sum(anomalies) > self.anomaly_treshold * (sample_len/mean_len)
        if not is_anomaly and learn_from_signal and self.learning_treshold < (sample_len/mean_len):
            old_mean = self.mean_ts
            self.mean_ts = incremental_mean_update(self.mean_ts, sample, self.magnitude)
            self.std_ts = incremental_variance_update(self.std_ts, sample, self.magnitude, old_mean, self.mean_ts)
            self.magnitude += 1
        return not is_anomaly

    def save_params(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load_params(self, path):
        with open(path, 'rb') as f:
            loaded_object: deviationClassifier = pickle.load(f)
            self.n_dim = loaded_object.n_dim
            self.anomaly_treshold = loaded_object.anomaly_treshold
            self.sensitivity = loaded_object.sensitivity
            self.magnitude = loaded_object.magnitude
            self.learning_treshold = loaded_object.learning_treshold
            self.mean_ts = loaded_object.mean_ts
            self.std_ts = loaded_object.std_ts
    
    def __repr__(self):
        return f"""n-Sigma anomaly detector:
                    n: {self.sensitivity}
                    Number of training signals: {self.magnitude}"""
    