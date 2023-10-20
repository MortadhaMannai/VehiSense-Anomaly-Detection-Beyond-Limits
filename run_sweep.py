#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Patricia Windler, Tim Bohne

import argparse
import os

import wandb

import train
from config import sweep_config, api_key
from train import file_path


def main(train_path: str, val_path: str, test_path: str) -> None:
    def call_training_procedure() -> None:
        """
        Wrapper for the training procedure.
        """
        with wandb.init():
            train.train_procedure(
                train_path, val_path, test_path, hyperparameter_config=wandb.config, vis_samples=False
            )

    os.environ["WANDB_API_KEY"] = api_key.wandb_api_key
    config = {'method': sweep_config.params["tuning_method"], 'parameters': sweep_config.sweep_config}
    sweep_id = wandb.sweep(config, project="Oscillogram Classification")
    wandb.agent(sweep_id, call_training_procedure, count=sweep_config.params["n_runs_in_sweep"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hyper parameter tuning with "weights & biases sweep"')
    parser.add_argument('--train_path', type=file_path, required=True)
    parser.add_argument('--val_path', type=file_path, required=True)
    parser.add_argument('--test_path', type=file_path, required=True)
    args = parser.parse_args()
    main(args.train_path, args.val_path, args.test_path)
