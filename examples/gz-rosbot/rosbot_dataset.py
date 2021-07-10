#!/usr/bin/python3
# -*- coding: utf-8 -*-
#licence removed for brevity

import torch
from pathlib import Path
import numpy as np
import math
import pandas as pd
import os

class RosbotDataset:
    """
    Base class for describing a dataset
    Contains a dataset and methods for working with it.

    For Rosbot each trajecory smaple = dataset
    """
    def __init__(
        self,
        data_x_path,
        data_u_path,
        data_t_path,
        data_k_path,
        device='cpu'
    ):
        """
        Args:
            :data_x_path: (str) Path to the csv file of the robot's state
            :data_u_path: (str) Path to the csv file of the robot's control
            :data_t_path: (str) Path to the csv file of the time
            :data_k_path: (str) Path to the csv file of the kinetic model state
            :device: (str) cuda / cpu
        Attributes:
            :data_t: (torch.tensor of shape [num_samples, 1]) timestamp sequence
            :data_x: (torch.tensor of shape [num_samples, 5]) robot state sequence
            :data_u: (torch.tensor of shape [num_samples, 2]) control sequence
            :data_k: (torch.tensor of shape [num_samples, 5]) kinetic model state sequence
        """
        super().__init__()
        
        self.data_x  = self.parse_robot_state(data_x_path)  
        self.data_u  = self.parse_control(data_u_path)
        self.data_t = self.parse_time(data_t_path)
        self.data_k  = self.parse_robot_state(data_k_path)
        self.device  = device
    
    def __len__(self):
        """
        Returns the size of the dataset
        """
        return len(self.data_x)

    def parse_robot_state(self, data_x_path):
        """
        Parse the csv file of the robot's state
        Args:
            :data_x_path: (str) path to the file
        Return:
            :data_x: (torch.tensor of shape [time (num of samples), robot_state])
            robot_state: [X, Y, YAW, V, W]
        """
        data = pd.read_csv(data_x_path).values
        T = len(data)
        data_x = np.zeros([T, 5])
        for i in range(T):
            line = [float(j) for j in data[i][0].split(" ")]
            data_x[i] = line
        return torch.tensor(data_x, dtype=torch.float)

    def parse_control(self, data_u_path):
        """
        Parse the csv file of the robot's state
        Args:
            :data_u_path: (str) path to the file
        Return: 
            :data_u: torch.tensor of shape [time (num of samples), control]
            control: [U_V, U_W]
        """
        data = pd.read_csv(data_u_path).values
        T = len(data)
        data_u = np.zeros([T, 2])
        for i in range(T):
            line = [float(j) for j in data[i][0].split(" ")]
            data_u[i] = line
        return torch.tensor(data_u, dtype=torch.float)

    def parse_time(self, data_t_path):
        """
        Parse the csv file of the time
        Args:
            :data_t_path (str): path to the file
        Return: 
            :data_t (torch.tensor of shape [time (num of samples), 1])
        """
        data = pd.read_csv(data_t_path).values
        T = len(data)
        data_t = np.zeros([T, 1])
        for i in range(T):
            line = data[i][0]
            data_t[i] = line
        return torch.tensor(data_t, dtype=torch.float)
            



def RosbotDataset_test():
    """
    Dummy tests for RosbotDataset
    """
   
    try:
        data_x_path = "/home/kostya_fs/code/rosbot_gazebo_datasets/train/traj=polygon-max_v=2.0-max_w=2.0/state.csv"
        data_u_path = "/home/kostya_fs/code/rosbot_gazebo_datasets/train/traj=polygon-max_v=2.0-max_w=2.0/control.csv"
        data_t_path = "/home/kostya_fs/code/rosbot_gazebo_datasets/train/traj=polygon-max_v=2.0-max_w=2.0/time.csv"
        test_dataset = RosbotDataset(
            data_x_path=data_x_path,
            data_u_path=data_u_path,
            data_t_path=data_t_path
        )
        print(" 1 OK")
    except Exception as e:
        print(" 1 FAIL")
        raise e

    try:
        global_path = "/home/kostya_fs/code/rosbot_gazebo_datasets/train/"
        list_of_datasets = list()
        for traj_data in os.listdir(global_path):
            data_x_path = global_path + traj_data + "/state.csv"
            data_u_path = global_path + traj_data + "/control.csv"
            data_t_path = global_path + traj_data + "/time.csv"
            list_of_datasets.append(
                RosbotDataset(
                    data_x_path=data_x_path,
                    data_u_path=data_u_path,
                    data_t_path = data_t_path
                )
            )
        if (len(list_of_datasets) == len(os.listdir(global_path))):
            print(" 2 OK")
    except Exception as e:
        print("2 FAIL")
        raise e



def main():
    """
    
    """
    RosbotDataset_test()


if __name__ == "__main__":
    main() 


# RosbotDataset from states.csv, control.csv, time.csv