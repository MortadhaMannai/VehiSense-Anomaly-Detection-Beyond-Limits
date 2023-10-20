#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Patricia Windler, Tim Bohne

import argparse
from typing import Tuple, Dict, List

import numpy as np
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesResampler

from cluster import create_processed_time_series_dataset, load_data, dir_path
from clustering_application import load_data as load_data_measurements
from config import knn_config, cluster_config
from oscillogram_classification import preprocess


def create_training_data(path: str, norm: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates the training dataset from the specified path.

    :param path: path to training data
    :param norm: normalization method to be applied to each sample
    :return: tuple of training samples and labels
    """
    create_processed_time_series_dataset(path, norm)
    train_x, train_y = load_data()
    train_x = preprocess.interpolation(train_x)
    return train_x, train_y


def create_test_data(path: str, norm: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates the test dataset from the specified path.

    :param path: path to test data
    :param norm: normalization method to be applied to each sample
    :return: tuple of test samples and labels
    """
    create_processed_time_series_dataset(path, norm)
    test_x, test_y, rec_ids = load_data_measurements()
    test_x = preprocess.interpolation(test_x)
    return test_x, test_y, rec_ids


def resample_voltage_values(voltage_values: np.ndarray) -> np.ndarray:
    """
    Resamples the given voltage values (time series) so that they reach the target size (sz - size of output TS).
    We need to reduce the length of the TS due to runtime and memory requirements.

    :param voltage_values: time series to be resampled
    :return: resampled voltage values
    """
    print("original TS size:", len(voltage_values[0]))
    voltage_values = TimeSeriesResampler(
        sz=len(voltage_values[0]) // knn_config.knn_config["resampling_divisor"]
    ).fit_transform(voltage_values)
    print("after down sampling:", len(voltage_values[0]))
    return voltage_values


def knn_classification(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
                       rec_ids: np.ndarray) -> Tuple[np.ndarray, Dict[str, Tuple[List, List]]]:
    """
    Performs k-nearest-neighbor (KNN) classification.

    :param train_x: training voltage signals
    :param train_y: training labels
    :param test_x: test voltage signals
    :param test_y: test labels
    :param rec_ids: measurement IDs for samples
    :return: array of predicted labels and classification result dictionary
    """
    print("perform k-nearest-neighbor (KNN) classification..")
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=knn_config.knn_config["n_neighbors"], weights="distance")
    fitted_knn = knn.fit(train_x, train_y)
    pred_y = fitted_knn.predict(test_x)
    classification_by_rec_id = {}
    for i in range(len(test_y)):
        if rec_ids[i] in classification_by_rec_id:
            classification_by_rec_id[rec_ids[i]][0].append(pred_y[i])
            classification_by_rec_id[rec_ids[i]][1].append(test_y[i])
        else:
            classification_by_rec_id[rec_ids[i]] = [pred_y[i]], [test_y[i]]
    return pred_y, classification_by_rec_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='knn-classifier for sub-ROI patches')
    parser.add_argument('--norm', action='store', type=str, required=True,
                        help='normalization method: %s' % cluster_config.cluster_config["normalization_methods"])
    parser.add_argument('--train_path', type=dir_path, required=True, help='path to the training data to be processed')
    parser.add_argument('--test_path', type=dir_path, required=True, help='path to the test data to be processed')
    args = parser.parse_args()

    x_train, y_train = create_training_data(args.train_path, args.norm)
    x_test, y_test, measurement_ids = create_test_data(args.test_path, args.norm)
    x_train = resample_voltage_values(x_train)
    y_pred, classification_per_measurement_id = knn_classification(x_train, y_train, x_test, y_test, measurement_ids)

    print("ground truth:", y_test)
    print("prediction:", y_pred)

    assert len(y_test) == len(y_pred)
    accuracy = (y_test == y_pred).sum() / len(y_test)
    print("accuracy:", accuracy)
    print("classification for each measurement: ")
    for key, value in classification_per_measurement_id.items():
        print("measurement", key)
        print("prediction:", value[0])
        print("ground truth:", value[1])
