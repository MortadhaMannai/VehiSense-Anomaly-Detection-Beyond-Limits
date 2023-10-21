#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

params = {
    "tuning_method": "random",
    "n_runs_in_sweep": 3
}

sweep_config = {
    "batch_size": {
        "values": [2, 16]
    },
    "learning_rate": {
        "values": [0.01, 0.0001]
    },
    "optimizer": {
        "value": "keras.optimizers.Adam"
    },
    "epochs": {
        "value": 3
    },
    "model": {
        "values": ["FCN"]
    },
    "loss_function": {
        "value": "binary_crossentropy"
    },
    "accuracy_metric": {
        "value": "binary_accuracy"
    },
    "trained_model_path": {
        "value": "best_model.h5"
    },
    "save_best_only": {
        "value": True
    },
    "monitor": {
        "value": "val_loss"
    },
    "ReduceLROnPlateau_factor": {
        "value": 0.5
    },
    "ReduceLROnPlateau_patience": {
        "value": 20
    },
    "ReduceLROnPlateau_min_lr": {
        "value": 0.0001
    },
    "EarlyStopping_patience": {
        "value": 50
    },
    "validation_split": {
        "value": 0.2
    }
}
