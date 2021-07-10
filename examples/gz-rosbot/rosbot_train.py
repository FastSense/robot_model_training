#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import yaml
import wandb
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import numpy as np

from rosbot_model import RosbotModel
from rosbot_dataset import RosbotDataset
from mppi_train.trainer import Trainer
from mppi_train.utils import parse_config, init_wandb
"""
launch example:

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="2" 
./rosbot_train.py -cfg /home/kostya_fs/code/ML-Training/rosbot-gazebo-model/train_configs/gz-rosbot_1.yaml

"""

def parse_args():
    """
    Parse arguments from the command line
    Return:
        args: (dict) contains command line arguments
    """
    args = {}
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', action='store', dest='cfg_file',
                        required=False, help='Yaml config for training')
    parser.add_argument('-name', action='store', dest='wandb_name',
                        required=False, help='Run name for wandb', default=None)
    cli_args = parser.parse_args()
    args['cfg_file'] = cli_args.cfg_file
    args['wandb_name'] = cli_args.wandb_name
    return args

def parse_datasets(path):
    """
    Parses all dataset directories, saves each to a list
    Args:
        :path: (str) path to directory with data files
    Return:
        :list_of_datasets: (list) list of RobotDatasets
    """
    list_of_datasets = list()
    for traj_data in os.listdir(path):
        data_x_path = path + traj_data + "/state.csv"
        data_u_path = path + traj_data + "/control.csv"
        data_t_path = path + traj_data + "/time.csv"
        data_k_path = path + traj_data + "/kinetic_model_state.csv"
        list_of_datasets.append(
            RosbotDataset(
                data_x_path=data_x_path,
                data_u_path=data_u_path,
                data_t_path=data_t_path,
                data_k_path=data_k_path
            )
        )
    return list_of_datasets

def parse_all_datasets(path, exclude=[]):
    """
    Parses all dataset directories, saves each to a list
    Args:
        :path: (str) path to directory with data files
    Return:
        :list_of_datasets: (list) list of RobotDatasets
    """
    list_of_datasets = list()
    for traj_type in os.listdir(path):
        if traj_type not in exclude:
            list_of_datasets += parse_datasets(path + traj_type + '/')
    return list_of_datasets


def plot_vel_and_ctrl_distribution(list_of_datasets: list):
    """
    Args:
        :list_of_datasets: list of RosbotDataset
    Return:
        :fig: (matplotlib.figure.Figure) velocities and control distribution graph
    """
    with torch.no_grad():
        all_state = torch.tensor([])
        all_control = torch.tensor([])
        for dataset in list_of_datasets:
            all_state = torch.cat([all_state, dataset.data_x])
            all_control = torch.cat([all_control, dataset.data_u])
        data_for_visualization = torch.cat([all_state[:, 3:], all_control[:, :]], 1).cpu().numpy() 
        data_for_visualization = pd.DataFrame(
            data_for_visualization,
            columns = ["V", "W", "U_V", "U_W"]
        )
    fig, ax = plt.subplots()
    graph = sns.pairplot(data_for_visualization[::100], plot_kws={'alpha':0.1})
    # graph.savefig("plots/vel_and_ctrl_distribution.png")
    return fig


def main():
    """
    
    """
    # parse args
    args = parse_args()
    # parse config
    config = parse_config(args["cfg_file"])
    use_wandb = False
    if args['wandb_name'] is not None:
        # init wandb
        use_wandb = True
        init_wandb(args['wandb_name'], config)

    # set random seed equal 0
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    # choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # declare trainer
    trainer = Trainer()

    # define model
    rosbot_model = RosbotModel(        
        n_layers=config.layers_num,
        hidden_size=config.hidden_size,
        activation_function=config.activation_function,
        learning_rate=config.learning_rate,
        model_type=config.model_type
    )
    # move model to devices
    rosbot_model = rosbot_model.to(device) 
    
    # define datasets
    train_data = parse_all_datasets(config.train_data_path, exclude=config.exclude_train_data.split()) 
    val_data = parse_all_datasets(config.val_data_path)
    test_data = parse_all_datasets(config.test_data_path)

    graph = plot_vel_and_ctrl_distribution(train_data)
    if use_wandb:
        wandb.log({'velocities and control distribution': wandb.Image(plt)})

    # Testing the integration function (update state) in the RosbotModel

    # test_integration_data = parse_all_datasets("/home/kostya_fs/code/rosbot_gazebo_datasets/DT_01/")
    # trainer.test_integration(
    #     rosbot_model,
    #     test_integration_data, 
    #     device
    # )

    # Neural network training
    trainer.fit(
        rosbot_model,
        train_data, 
        val_data, 
        config.num_epochs, 
        config.batch_size, 
        config.rollout_size,
        config.main_metric,
        device,
        use_wandb,
        config.plot_trajectories
    )

    # Testing a neural network
    trainer.evaluate(
        rosbot_model,
        test_data,
        device,
        plot_trajectories=config.plot_trajectories,
        use_wandb=use_wandb,
        save_to_csv=config.save_to_csv
    )

    # save pytorch model
    if use_wandb:
        torch.save(rosbot_model.state_dict(), wandb.run.dir + "/model.pt")



if __name__ == "__main__":
    main()
