#!/usr/bin/python3
# -*- coding: utf-8 -*-
# licence removed for brevity

import os
import torch
import numpy as np
import pandas as pd


class LinkManDataset:
    """
    Base class for describing a dataset
    Contains a dataset and methods for working with it.

    For LinkMan each trajecory smaple = dataset
    """

    def __init__(
        self,
        data_x_path,
        data_u_path,
        data_t_path,
        device='cpu'
    ):
        """
        Args:
            :data_x_path: (str) Path to the csv file of the robot's state
            :data_u_path: (str) Path to the csv file of the robot's control
            :data_t_path: (str) Path to the csv file of the time
            :device: (str) cuda / cpu
        Attributes:
            :data_t: (torch.tensor of shape [num_samples, 1]) timestamp sequence
            :data_x: (torch.tensor of shape [num_samples, 5]) robot state sequence
            :data_u: (torch.tensor of shape [num_samples, 2]) control sequence
        """
        super().__init__()

        self.data_x = self.parse_robot_state(data_x_path)
        self.data_u = self.parse_control(data_u_path)
        self.data_t = self.parse_time(data_t_path)
        self.device = device

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
        data = pd.read_csv(data_x_path, delimiter=' ')
        data = data[['x', 'y', 'yaw', 'v_x', 'w_z']].values
        return torch.tensor(data, dtype=torch.float)

    def parse_control(self, data_u_path):
        """
        Parse the csv file of the robot's state
        Args:
            :data_u_path: (str) path to the file
        Return:
            :data_u: torch.tensor of shape [time (num of samples), control]
            control: [U_V, U_W]
        """
        data = pd.read_csv(data_u_path, delimiter=' ').values
        return torch.tensor(data, dtype=torch.float)

    def parse_time(self, data_t_path):
        """
        Parse the csv file of the time
        Args:
            :data_t_path (str): path to the file
        Return:
            :data_t (torch.tensor of shape [time (num of samples), 1])
        """
        data = pd.read_csv(data_t_path, delimiter=' ').values
        return torch.tensor(data, dtype=torch.float)


def LinkManDataset_test():
    """
    Dummy tests for RosbotDataset
    """

    try:
        data_x_path = "/home/kostya/work/metalbot/ros2_ws/src/utils/logger/dataset_sy/around_table_left/robot_state.csv"
        data_u_path = "/home/kostya/work/metalbot/ros2_ws/src/utils/logger/dataset_sy/around_table_left/control.csv"
        data_t_path = "/home/kostya/work/metalbot/ros2_ws/src/utils/logger/dataset_sy/around_table_left/time.csv"
        test_dataset = LinkManDataset(
            data_x_path=data_x_path,
            data_u_path=data_u_path,
            data_t_path=data_t_path
        )
        print(test_dataset.data_x[0:20])
        print(" 1 OK")
    except Exception as e:
        print(" 1 FAIL")
        raise e

    try:
        global_path = "/home/kostya/work/metalbot/ros2_ws/src/utils/logger/dataset_sy/"
        list_of_datasets = list()
        for traj_data in os.listdir(global_path):
            data_x_path = global_path + traj_data + "/robot_state.csv"
            data_u_path = global_path + traj_data + "/control.csv"
            data_t_path = global_path + traj_data + "/time.csv"
            list_of_datasets.append(
                LinkManDataset(
                    data_x_path=data_x_path,
                    data_u_path=data_u_path,
                    data_t_path=data_t_path
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
    LinkManDataset_test()


if __name__ == "__main__":
    main()
