#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne

import argparse
import os
from typing import Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from wandb.keras import WandbCallback

import models
from config import api_key, run_config
from training_data import TrainingData


def set_up_wandb(wandb_config: dict) -> None:
    """
    Setup for 'weights and biases'.

    :param wandb_config: configuration to be used
    """
    if wandb.run is None:
        wandb.login(key=api_key.wandb_api_key)
        wandb.init(project="Oscillogram Classification", config=wandb_config)


def visualize_n_samples_per_class(x: np.ndarray, y: np.ndarray) -> None:
    """
    Iteratively visualizes one sample per class as long as the user enters '+'.

    :param x: sample series
    :param y: corresponding labels
    """
    plt.figure()
    classes = np.unique(y, axis=0)
    samples_by_class = {c: x[y == c] for c in classes}

    for sample in range(len(samples_by_class[classes[0]])):
        key = input("Enter '+' to see another sample per class\n")
        if key != "+":
            break
        for c in classes:
            plt.plot(samples_by_class[c][sample], label="class " + str(c))
        plt.legend(loc="best")
        plt.show()
        plt.close()


def train_keras_model(model: keras.models.Model, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                      y_val: np.ndarray) -> None:
    """
    Trains the specified 'Keras' model on the specified data.

    :param model: model to be trained
    :param x_train: training data samples
    :param y_train: training data labels
    :param x_val: validation data samples
    :param y_val: validation data labels
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            wandb.config["trained_model_path"],
            save_best_only=wandb.config["save_best_only"],
            monitor=wandb.config["monitor"]
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=wandb.config["monitor"],
            factor=wandb.config["ReduceLROnPlateau_factor"],
            patience=wandb.config["ReduceLROnPlateau_patience"],
            min_lr=wandb.config["ReduceLROnPlateau_min_lr"]
        ),
        keras.callbacks.EarlyStopping(
            monitor=wandb.config["monitor"],
            patience=wandb.config["EarlyStopping_patience"],
            verbose=1
        ),
        WandbCallback()
    ]
    optimizer = eval(wandb.config["optimizer"])(learning_rate=wandb.config["learning_rate"])
    model.compile(optimizer=optimizer, loss=wandb.config["loss_function"], metrics=[wandb.config["accuracy_metric"]])
    history = model.fit(
        x_train,
        y_train,
        batch_size=wandb.config["batch_size"],
        epochs=wandb.config.epochs,
        callbacks=callbacks,
        validation_split=wandb.config["validation_split"],
        validation_data=(x_val, y_val),
        verbose=1,
    )
    plot_training_and_validation_loss(history)


def train_model(model: Union[keras.models.Model, RandomForestClassifier, MLPClassifier, DecisionTreeClassifier],
                x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) \
        -> Union[keras.models.Model, RandomForestClassifier, MLPClassifier, DecisionTreeClassifier]:
    """
    Trains the selected model (classifier).

    :param model: model to be trained
    :param x_train: training samples
    :param y_train: corresponding training labels
    :param x_val: validation samples
    :param y_val: corresponding validation labels
    :return trained model
    """
    print("training model:\ntotal training samples:", len(x_train))
    for c in np.unique(y_train, axis=0):
        assert len(x_train[y_train == c]) > 0
        print("training samples for class", str(c), ":", len(x_train[y_train == c]))
    print("total validation samples:", len(x_val))
    for c in np.unique(y_val, axis=0):
        assert len(x_val[y_val == c]) > 0
        print("validation samples for class", str(c), ":", len(x_val[y_val == c]))

    x_train = np.squeeze(x_train)
    x_val = np.squeeze(x_val)
    print("train shape:", x_train.shape)
    print("val shape:", x_val.shape)
    # interpolation, padding, etc., is already performed in preprocessing
    assert x_train.shape[1] == x_val.shape[1]

    if 'keras' in str(type(model)):
        train_keras_model(model, x_train, y_train, x_val, y_val)
        return model
    else:
        print("neurons of input layer:", len(x_train[0]))
        print("neurons of output layer:", len(np.unique(y_train)))
        return model.fit(x_train, y_train)


def plot_training_and_validation_loss(history: keras.callbacks.History) -> None:
    """
    Plots the learning curves.

    :param history: training history
    """
    metric = wandb.config["accuracy_metric"]
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()


def evaluate_model_on_test_data(
        x_test: np.ndarray, y_test: np.ndarray,
        model: Union[keras.models.Model, RandomForestClassifier, MLPClassifier, DecisionTreeClassifier],
        x_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    Evaluates the trained model on the specified test data.

    :param x_test: test samples
    :param y_test: test labels
    :param model: trained model to be evaluated
    :param x_train: trainings samples
    :param y_train: training labels
    """
    print("evaluating model:\ntotal test samples:", len(x_test))
    for c in np.unique(y_test, axis=0):
        assert len(x_test[y_test == c]) > 0
        print("test samples for class", str(c), ":", len(x_test[y_test == c]))

    if 'keras' in str(type(model)):
        # should be the same, but read from file
        model = keras.models.load_model(wandb.config["trained_model_path"])
        expected_feature_vector_len = model.layers[0].output_shape[0][1]
    else:
        expected_feature_vector_len = model.n_features_in_
    # test samples should match model input length (assured via preprocessing)
    assert x_test.shape[1] == expected_feature_vector_len
    # shuffle test data
    idx = np.random.permutation(len(x_test))
    x_test = x_test[idx]
    y_test = y_test[idx]

    if 'keras' in str(type(model)):
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print("test accuracy:", test_acc)
        print("test loss:", test_loss)
        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
    else:
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)
        # accuracy score -> set of labels predicted for a sample must exactly match the corresponding set of labels
        print("----------------------------------------------------------------------------")
        print("accuracy on training data:", accuracy_score(y_train, y_pred_train))
        print("accuracy on test data:", accuracy_score(y_test, y_pred_test))
        print("----------------------------------------------------------------------------")
        # precision -> ability of the classifier not to label as positive a sample that is negative
        print("precision:", precision_score(y_test, y_pred_test))
        print("CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred_test))


def perform_consistency_check(train_data: TrainingData, val_data: TrainingData, test_data: TrainingData,
                              z_train: List[str], z_val: List[str], z_test: List[str]) -> None:
    """
    Performs a consistency check for the provided data:
        - all three (train, val, test) should either provide feature info or not
        - if the data consists of feature vectors, all three sets should contain
          exactly the same features in the same order

    The underlying assumption is that we should always extract the same features for training, testing,
    and finally when applying the trained model.

    :param train_data: training dataset
    :param val_data: validation dataset
    :param test_data: test dataset
    :param z_train: features of the training dataset (or empty)
    :param z_val: features of the validation dataset (or empty)
    :param z_test: features of the test dataset (or empty)
    """
    print("performing consistency check..")
    # equal number of dimensions (either all have feature info or none)
    assert len(train_data[:]) == len(val_data[:]) == len(test_data[:])
    # equal number of considered features
    assert len(z_train) == len(z_val) == len(z_test)
    # check whether all features are the same (+ same order)
    for i in range(len(z_train)):
        assert z_train[i] == z_val[i] == z_test[i]
    print("consistency check passed..")


def prepare_data(train_data_path: str, val_data_path: str, test_data_path: str, keras_model: bool, vis_samples: bool) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares the data for the training / evaluation process.

    :param train_data_path: path to read training data from
    :param val_data_path: path to read validation data from
    :param test_data_path: path to read test data from
    :param keras_model: whether the data is prepared for a keras model
    :param vis_samples: whether to display option to visualize samples
    :return: (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    z_train = z_val = z_test = []
    # avoid calls with .csv data
    assert '.npz' in train_data_path and '.npz' in val_data_path and '.npz' in test_data_path

    data = TrainingData(np.load(train_data_path, allow_pickle=True))
    x_train = data[:][0]
    y_train = data[:][1]
    if len(data[:]) == 3:
        z_train = data[:][2]
    if vis_samples:
        visualize_n_samples_per_class(x_train, y_train)

    if keras_model:
        # generally applicable to multivariate time series
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    idx = np.random.permutation(len(x_train))  # shuffle training set
    x_train = x_train[idx]
    y_train = y_train[idx]

    val_data = TrainingData(np.load(val_data_path, allow_pickle=True))  # read validation data
    x_val = val_data[:][0]
    y_val = val_data[:][1]
    if len(val_data[:]) == 3:
        z_val = data[:][2]

    test_data = TrainingData(np.load(test_data_path, allow_pickle=True))  # read test data
    x_test = test_data[:][0]
    y_test = test_data[:][1]
    if len(test_data[:]) == 3:
        z_test = test_data[:][2]

    perform_consistency_check(data, val_data, test_data, z_train, z_val, z_test)
    return x_train.astype('float32'), y_train, x_val.astype('float32'), y_val, x_test.astype('float32'), y_test


def train_procedure(train_path: str, val_path: str, test_path: str,
                    hyperparameter_config: dict = run_config.hyperparameter_config, vis_samples: bool = True):
    """
    Initiates the training and evaluation procedures.

    :param train_path: path to training data
    :param val_path: path to validation data
    :param test_path: path to test data
    :param hyperparameter_config: hyperparameter specification
    :param vis_samples: whether to display option to visualize samples
    """
    keras_model = hyperparameter_config["model"] in ["FCN", "ResNet"]
    if keras_model:
        set_up_wandb(hyperparameter_config)

    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data(
        train_path, val_path, test_path, keras_model, vis_samples
    )
    model = models.create_model(x_train.shape[1:], len(np.unique(y_train)), architecture=hyperparameter_config["model"])

    if 'keras' in str(type(model)):
        keras.utils.plot_model(model, to_file="img/model.png", show_shapes=True)

    trained_model = train_model(model, x_train, y_train, x_val, y_val)
    evaluate_model_on_test_data(x_test, y_test, trained_model, x_train, y_train)


def file_path(path: str) -> str:
    """
    Returns path if it's valid, raises error otherwise.

    :param path: path to be checked
    :return: feasible path or error
    """
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid file")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training model with time series data..')
    parser.add_argument('--train_path', type=file_path, required=True)
    parser.add_argument('--val_path', type=file_path, required=True)
    parser.add_argument('--test_path', type=file_path, required=True)
    args = parser.parse_args()

    train_procedure(args.train_path, args.val_path, args.test_path)
