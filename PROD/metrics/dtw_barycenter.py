import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tslearn.barycenters import dtw_barycenter_averaging, softdtw_barycenter, euclidean_barycenter
from tslearn.utils import to_time_series_dataset
from tslearn.metrics import dtw as ts_dtw
from tslearn.metrics import dtw_path_from_metric as m_dtw
import pickle


METHODS_LABEL_DICT = {0 : "Signals as a distance matrix",
                      1 : "DTW-aligned SAD distance (L1 norm)",
                      2 : "DTW-aligned euclidean distance (L2 norm)",
                      3 : "Barycenter as a distance matrix",
                      4 : "DTW-aligned Cosine distance",
                      5 : "Unaligned euclidean distance (L2 norm)",
                      6 : "Unaligned SAD distance (L1 norm)",
                      None : ""}


def compute_hard_barycenter(samples, path_to_save_hard):
    """
    Function, that computes DBA algorithm and finds hard DTW barycenter
    for signals in samples.
    Args:
        samples : list[np.ndarray] - list of one or multidimensional np arrays
                    representing signals. the signals can be of different length
                    along the 0th axis, but must be the same length along the 1st
                    axis. (0th axis represents time)
        path_to_save_hard: str - path to save the output file.
    
    Returns:
        barycenter_hard : np.ndarray - barycenter of given signals.
    """
    X = to_time_series_dataset(samples)
    barycenter_hard = dtw_barycenter_averaging(X)
    print("hard computed")
    np.save(path_to_save_hard, barycenter_hard)
    return barycenter_hard


def get_distance(signal1, signal2, method = 2):
    """
    Returns a distance between two signals
    args:
    
        singal1 : np.ndarray - array representing the signal n x m where
                              n represents time and m is number of dimensions

        singal2 : np.ndarray - array representing the signal n x m where
                              n represents time and m is number of dimensions
                    
        method : int - represents the method used for distance measurement:
                    0 : signal1 is used as a distance matrix TODO: ask what that means
                    1 : L1 norm is used between two DTW-aligned arrays == Manhattan distance
                    2 : L2 norm is used between two DTW-aligned arrays == Euclidean distance
                    3 : signal2 is used as a distance matrix TODO: ask what that means
                    4 : cosine distance of DTW-aligned signals is used - defined by cosine similarity
                    5 : Unaligned Euclidean distance is used
                    6 : Unaligned Manhattan distance is used
                    None: returns dist = 0
        returns:
            dist : int - distance between the two signals using specified metric

    """
    if method == 0: 
        return m_dtw(signal1, signal2, metric="precomputed")[1]
    elif method == 1:
        return m_dtw(signal1, signal2, metric="l1")[1]
    elif method == 2:
        return ts_dtw(signal1, signal2)
    elif method == 3:
        return m_dtw(signal2, signal1, metric="precomputed")[1]
    elif method == 4:
        return m_dtw(signal2, signal1, metric="cosine")[1]
    elif method == 5:
        maxlen = min(np.shape(signal1)[0], np.shape(signal2)[0])
        signal1_m, signal2_m = signal1[:maxlen, :], signal2[:maxlen, :]
        return np.sum(np.linalg.norm(signal1_m - signal2_m, axis=0))
    elif method == 6:
        maxlen = min(np.shape(signal1)[0], np.shape(signal2)[0])
        signal1_m, signal2_m = signal1[:maxlen, :], signal2[:maxlen, :]
        return np.sum(np.abs(signal1_m - signal2_m))
    elif method == None:
        return 0
    else:
        raise Exception(f"WARN: Invalid metric to use distance selected. Method selected: {method}")
        return None

def compute_dtw_dists(correct_sample_list : list, wrong_sample_list : list, barycenters : list, ebars : list, method = [2,5]):
    """
    Computes two metrics for successfull and wrong singals with respect to two different barycenter sets.
    Args:
        correct_sample_list : list[np.ndarray] - signals of successfully measured processes
        wrong_sample_list : list[np.ndarray] - signals of wrongly measured processes
        barycenters : list[np.ndarray] - list of signals to which the measurments are compared via method[0]
        ebars : list[np.ndarray] - list of signals to which the measurements are compared via method[1]
        method : list[int1, int2] - two numbers which specify the method used for comparrison see function "get_distance"
    
    Returns:
        correct_points : list[list[int], list[int]] - two lists of measurements correct_points[0] = measurements via method[0],
                                                                                correct_points[1] = measurements via method[1] 
        wrong_points : list[list[int], list[int]] - two lists of measurements wrong_points[0] = measurements via method[0],
                                                                              wrong_points[1] = measurements via method[1]
        bar : list[list[int], list[int]] - list of mutual distances between barycenters from set 1
        ebar : list[list[int], list[int]] - list of mutual distances between barycenters from set 2
    """
    correct_points, wrong_points = [],[]
    for i in range(len(ebars)):
        orig_shape = np.shape(ebars[i])[1]
        ebars[i] = ebars[i][np.logical_not(np.isnan(ebars[i]))]
        ebars[i] = ebars[i].reshape((len(ebars[i])//orig_shape), orig_shape)
    for i in correct_sample_list:
        coords = execute_method(i, barycenters, ebars, method)
        correct_points.append(*coords)   
    for i in wrong_sample_list:
        coords = execute_method(i, barycenters, ebars, method)
        wrong_points.append(*coords)
    bar = return_relative_distance_of_barycenters(barycenters, method)
    ebar = return_relative_distance_of_barycenters(ebars, method)
    print(correct_points)
    return correct_points, wrong_points, bar, ebar


def execute_method(signal, comparrison_set1, comparrison_set2, method = [2, 5]):
    """
    Compares the signal to both comparisson sets via selected metrics and returns
    the selected distance to closest set

    Args:
        singal : np.ndarray - array representing the signal n x m where
                              n represents time and m is number of dimensions
        comparrison_set1 : list[np.ndarray] - array of signals to compare the signal
                              via the method[0] (see function get_distance)
                              all signals must have the same number of dimensions as
                              the "signal" variable
        comparrison_set2 : list[np.ndarray] - array of signals to compare the signal
                              via the method[1] (see function get_distance)
                              all signals must have the same number of dimensions as
                              the "signal" variable
        method : list[int1|None, int2|None] - selects the metric to be used
                              (see function get_distance)
    """
    points = []
    dist1, dist2 = np.inf, np.inf
    for j in range(len(comparrison_set1)):
        dist1 = min(get_distance(signal, comparrison_set1[j], method[0]), dist1)
        if comparrison_set2:
            dist2 = min(get_distance(signal, comparrison_set2[j], method[1]), dist2)
        else:
            dist2 = 0
    points.append([dist1, dist2])
    return points

def get_mean_std(signal):
    """
    Returns mean and standard deviation for given signal.
    Args:
        singal : np.ndarray - array representing the signal n x m where
                              n represents time and m is number of dimensions
    Returns:
        mean : np.ndarray - mean along the 0th axis (mean for all dimensions) 1 x m vector
        std : np.ndarray - standard deviation for all m dimensions ()
    """
    return np.mean(signal, axis=0), np.std(signal, axis=0)

def transform_signal(signal, mean, std):
    """
    Performs a "normalization" transformation on a given signal using given mean and standard
    deviation
    Args:
        singal : np.ndarray - array representing the signal n x m where
                              n represents time and m is number of dimensions
    """
    return (signal - mean)/std

def get_barycenters(hard_path, soft_path):
    """
    Loads and returns precomputed barycenters and adds np.arange as the time dimension
    Args:
        hard_path : str - path to hard barycenter
        soft_path : str - path to soft barycenter
    Returns:
        hard : np.ndarray - hard barycenter
        soft : np.ndarray - soft barycenter
    """
    hard = np.load(hard_path)
    soft = np.load(soft_path)
    lenh = np.arange(np.shape(hard)[0])[:, np.newaxis]
    lens = np.arange(np.shape(soft)[0])[:, np.newaxis]
    hard = np.concatenate((lenh, hard), axis = 1)
    soft = np.concatenate((lens, soft), axis = 1)
    return hard, soft


def return_relative_distance_of_barycenters(barycenters, method = [2, 5]):
    """
    Computes a distance among multiple barycenters via given metric

    Args:
        barycenters : list[np.ndarray] - list of barycenters
        method : list[int1|None, int2|None] - selected metric (see function get_distance)
    
    Returns:
        bar: list[list[int], list[int]] - list of distances by specified metrics
                            bar[0] metric specified in method[0] is used
                            bar[1] metric specified in method[1] is used
    """
    barx = []
    bary = []
    for i in range(len(barycenters)):
        for j in range(i, len(barycenters)):
            barx.append(get_distance(barycenters[i], barycenters[j], method[0]))
            bary.append(get_distance(barycenters[i], barycenters[j], method[1]))
    return [barx, bary]
