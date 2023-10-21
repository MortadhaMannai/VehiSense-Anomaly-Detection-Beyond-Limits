#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

hyperparameter_config = {
    "batch_size": 4,
    "learning_rate": 0.001,
    "optimizer": "keras.optimizers.Adam",
    "epochs": 30,
    "model": "FCN",
    "loss_function": "binary_crossentropy",  # sparse_categorical_crossentropy
    "accuracy_metric": "binary_accuracy",  # sparse_categorical_accuracy
    "trained_model_path": "best_model.h5",
    "save_best_only": True,
    "monitor": "val_loss",
    "ReduceLROnPlateau_factor": 0.5,
    "ReduceLROnPlateau_patience": 20,
    "ReduceLROnPlateau_min_lr": 0.0001,
    "EarlyStopping_patience": 50,
    "validation_split": 0.2
}
