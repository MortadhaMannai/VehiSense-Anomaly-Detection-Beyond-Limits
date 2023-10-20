#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tim Bohne, Patricia Windler

from typing import Tuple, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras


def create_fcn_model(input_shape: Tuple, num_classes: int) -> keras.models.Model:
    """
    Defines the CNN (FCN) architecture to be worked with.

    :param input_shape: shape of the input layer
    :param num_classes: number of unique classes to be considered
    :return: CNN (FCN) model
    """
    # input shape -> number of data points per sample
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    if num_classes == 2:  # binary classification
        # Since the architecture is equivalent with both one output neuron (decision boundary 0.5, sigmoid) and two
        # output neurons (𝑎𝑟𝑔𝑚𝑎𝑥, softmax) for binary classification, we use the less computationally complex version
        # with one neuron.
        output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)
    else:  # multi-class classification
        output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def create_resnet_model(input_shape: Tuple, num_classes: int) -> keras.models.Model:
    """
    Defines the ResNet architecture to be worked with.

    :param input_shape: shape of the input layer
    :param num_classes: number of unique classes to be considered
    :return: ResNet model
    """
    n_feature_maps = 64
    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    # FINAL
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
    output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap_layer)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def create_random_forest_model() -> RandomForestClassifier:
    """
    Defines the 'Random Forest' architecture to be worked with.

    :return: 'Random Forest' classifier
    """
    return RandomForestClassifier(verbose=1, n_estimators=5000, max_depth=1000, random_state=42)


def create_mlp_model() -> MLPClassifier:
    """
    Defines the MLP architecture to be worked with.

    :return: MLP classifier
    """
    return MLPClassifier(verbose=1, random_state=1, max_iter=100000, n_iter_no_change=500, batch_size=32,
                         hidden_layer_sizes=(16,))


def create_decision_tree_model() -> DecisionTreeClassifier:
    """
    Defines the 'Decision Tree' architecture to be worked with.

    :return: 'Decision Tree' classifier
    """
    return DecisionTreeClassifier(criterion='entropy')


def create_model(input_shape: Tuple, num_classes: int, architecture: str = "FCN") \
        -> Union[keras.models.Model, RandomForestClassifier, MLPClassifier, DecisionTreeClassifier]:
    """
    Initiates model generation based on the specified architecture, input shape, and number of classes.

    :param input_shape: shape of the input layer
    :param num_classes: number of unique classes to be considered
    :param architecture: architecture to be generated (default FCN)
    :return: created model
    """
    if architecture == "FCN":
        return create_fcn_model(input_shape, num_classes)
    elif architecture == "ResNet":
        return create_resnet_model(input_shape, num_classes)
    elif architecture == "RandomForest":
        return create_random_forest_model()
    elif architecture == "MLP":
        return create_mlp_model()
    elif architecture == "DecisionTree":
        return create_decision_tree_model()
    else:
        raise ValueError("Unknown model architecture: " + architecture)
