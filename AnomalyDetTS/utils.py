import numpy as np
import pandas as pd
import torch

def transform_pd_to_npy(dataframe):
    return dataframe.to_numpy() if isinstance(dataframe, pd.DataFrame) else dataframe

def transform_pd_to_tensor(dataframe):
    return torch.Tensor(dataframe.to_numpy()) if isinstance(dataframe, pd.DataFrame) else dataframe

def set_signals_to_same_length(signal1, signal2):
    signal1_len = np.shape(signal1)[0]
    signal2_len = np.shape(signal2)[0]
    signal1_, signal2_ = signal1, signal2
    if signal1_len > signal2_len:
        signal2_ = np.pad(signal2, ((0, signal1_len - signal2_len), (0, 0)), 'constant')
    elif signal1_len < signal2_len:
        signal1_ = np.pad(signal1, ((0, signal2_len - signal1_len), (0, 0)), 'constant')
    return signal1_, signal2_


def incremental_mean_update(cur_mean, new_sample, n_samples):
    n_samples = n_samples if n_samples > 0 else 1
    if cur_mean is None:
        return new_sample
    new_mean = cur_mean + (new_sample - cur_mean)/n_samples
    return transform_pd_to_npy(new_mean)

def incremental_variance_update(cur_variance, new_sample, n_samples, cur_mean, new_mean):
    n_samples = n_samples if n_samples > 0 else 1
    if cur_variance is None:
        return np.zeros_like(cur_mean)
    new_var = np.sqrt((n_samples * cur_variance**2 + (new_sample - cur_mean) * (new_sample - new_mean))/n_samples)
    return transform_pd_to_npy(new_var)

def load_data(path : str, id_name : str = 'meas_id', id : list[int] | None = None,
              feature_column_start : int = 3, resampling_fcn = None,
              supervised : bool = True):
    sequences : list= []
    raw_sequence = pd.read_csv(path)
    if id is not None:
        raw_sequence = raw_sequence[raw_sequence[id_name].isin(id)]
    FEATURE_COLUMNS = raw_sequence.columns.to_list()[feature_column_start:]
    for _, group in raw_sequence.groupby("idx"):
        sequence_features = group[FEATURE_COLUMNS] if resampling_fcn is None else resampling_fcn(group[FEATURE_COLUMNS])
        label = list(set(group["label"]))[0] if supervised else None
        sequences.append((sequence_features, label))
    return sequences

def get_swich_points(arr):
    arr = np.array(arr)
    padded_arr = np.pad(arr, (1, 1), mode='constant', constant_values=0)
    diffs = np.diff(padded_arr.astype(bool).astype(int))
    start_indices = np.where(diffs == 1)[0]
    end_indices = np.where(diffs == -1)[0]
    ret = np.array([start_indices, end_indices]).T
    return ret