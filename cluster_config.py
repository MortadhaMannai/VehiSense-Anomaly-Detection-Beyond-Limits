#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

cluster_config = {
    "seed": 42,
    "number_of_clusters": 7,  # for the battery voltage signal (sub-ROIs)
    "n_label": 5,  # number of integers that appear as labels in the file names of the patches
    "n_init": 50,
    "max_iter": 500,
    "max_iter_barycenter": 500,
    "resampling_divisor": 10,
    "interpolation_target": "MIN",  # other options are 'MAX' and 'AVG'
    "small_val": 0.0000001,
    "normalization_methods": ["none", "z_norm", "min_max_norm", "dec_norm", "log_norm"]
}

cluster_application_config = {
    "model": "trained_models/dba_km.pkl",
    "data": "data/patch_data.npz",
    "measurement_ids": "data/patch_measurement_ids.csv",
    "metric": "DTW",
    "seed": 42
}
