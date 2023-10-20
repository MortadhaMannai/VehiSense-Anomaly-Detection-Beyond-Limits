#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import argparse
import os
from typing import List, Tuple, Dict

import joblib
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw, soft_dtw

from cluster import create_processed_time_series_dataset
from config import cluster_config
from training_data import TrainingData


def compute_distances(sample: np.ndarray, clustering_model: TimeSeriesKMeans) -> List:
    """
    Computes the distance between the provided sample and each cluster of the specified model using the configured
    metric.

    :param sample: new sample to be assigned to the closest cluster
    :param clustering_model: "trained" clustering model
    :return: computed cluster distances
    """
    if cluster_config.cluster_application_config["metric"] == "DTW":
        return [dtw(sample, cluster_centroid) for cluster_centroid in clustering_model.cluster_centers_]
    elif cluster_config.cluster_application_config["metric"] == "SOFT_DTW":
        return [soft_dtw(sample, cluster_centroid) for cluster_centroid in clustering_model.cluster_centers_]
    else:
        # default option is DTW
        return [dtw(sample, cluster_centroid) for cluster_centroid in clustering_model.cluster_centers_]


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the test samples.

    :return: test samples (voltage values, labels, measurement IDs)
    """
    data = TrainingData(np.load(cluster_config.cluster_application_config["data"], allow_pickle=True))
    measurement_ids = np.loadtxt(cluster_config.cluster_application_config["measurement_ids"], delimiter=',', dtype=str)

    if measurement_ids.shape == ():
        # convert scalar value to a 1D array
        measurement_ids = np.array([measurement_ids])

    x_test = data[:][0]
    y_test = data[:][1]
    np.random.seed(cluster_config.cluster_application_config["seed"])
    idx = np.random.permutation(len(x_test))
    return x_test[idx], y_test[idx], measurement_ids[idx] if len(y_test) > 0 else ()


def determine_best_matching_cluster_for_sample(sample: np.ndarray, clustering_model: TimeSeriesKMeans) -> int:
    """
    Determines the best matching cluster (smallest distance to centroid) for the specified sample.

    :param sample: sample to determine cluster for
    :param clustering_model: 'trained' clustering model
    :return: index of best-matching sample
    """
    distances = compute_distances(sample, clustering_model)
    # select the best-matching cluster for the new sample
    return int(np.argmin(distances))


def dir_path(path: str) -> str:
    """
    Returns path if it's valid, raises error otherwise.

    :param path: path to be checked
    :return: feasible path or error
    """
    if os.path.isfile(path) or os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")


def cluster_test_samples(x_test: np.ndarray, y_test: np.ndarray, measurement_ids: np.ndarray,
                         trained_model: TimeSeriesKMeans, ground_truth: np.ndarray) -> Dict[str, Tuple[List, List]]:
    """
    Clusters the test samples, i.e., assigns new samples to predetermined clusters.

    :param x_test: voltage value samples
    :param y_test: labels for samples
    :param measurement_ids: measurement IDs of samples
    :param trained_model: clustering model (predetermined clusters)
    :param ground_truth: ground truth labels of samples
    :return dictionary containing clustering (classification) results
            scheme: rec_id: [[prediction], [ground_truth]]
    """
    classification_per_measurement_id = {}

    for i in range(len(x_test)):
        test_sample = x_test[i]
        print("test sample excerpt:", test_sample[:15])
        best_matching_cluster = determine_best_matching_cluster_for_sample(test_sample, trained_model)
        print("best matching cluster for new sample:", best_matching_cluster,
              "(", ground_truth[best_matching_cluster], ")")
        best_cluster = ground_truth[best_matching_cluster]

        # ground truth provided?
        if y_test[i] is not None:
            test_sample_ground_truth = y_test[i]
            print("ground truth:", test_sample_ground_truth)

            # if the ground truth matches the most prominent label in the cluster, it's a success
            d = {i: best_cluster.count(i) for i in np.unique(best_cluster)}
            most_prominent_entry = max(d, key=d.get)

            if test_sample_ground_truth == most_prominent_entry:
                print("SUCCESS: ground truth (", test_sample_ground_truth,
                      ") matches most prominent entry in cluster (", most_prominent_entry, ")")
            else:
                print("FAILURE: ground truth (", test_sample_ground_truth,
                      ") does not match most prominent entry in cluster (", most_prominent_entry, ")")
            print("-------------------------------------------------------------------------")

            if measurement_ids[i] in classification_per_measurement_id:
                classification_per_measurement_id[measurement_ids[i]][0].append(most_prominent_entry)
                classification_per_measurement_id[measurement_ids[i]][1].append(test_sample_ground_truth)
            else:
                classification_per_measurement_id[measurement_ids[i]] = (
                    [most_prominent_entry], [test_sample_ground_truth]
                )
    return classification_per_measurement_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assign new samples to predetermined clusters')
    parser.add_argument('--samples', type=dir_path, required=True, help='path to the samples to be assigned')
    args = parser.parse_args()
    create_processed_time_series_dataset(args.samples)

    # load saved clustering model from file
    model, y_pred, ground_truth_labels = joblib.load(cluster_config.cluster_application_config["model"])
    test_x, test_y, rec_ids = load_data()
    clustering_res = cluster_test_samples(test_x, test_y, rec_ids, model, ground_truth_labels)

    for key, value in clustering_res.items():
        print("measurement:", key)
        print("prediction:", value[0])
        print("ground truth:", value[1])
