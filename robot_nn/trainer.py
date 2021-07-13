
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
 
class Trainer:
    """
    A class providing tools for training a dynamic robot model
    """
    def __init__(self, device='cpu'):
        """
        Args:
            :device (str) - cuda / cpu 
        """
        self.device = device

    def fit(
        self, model, train_data: list,
        val_data: list, epochs_num: int,
        batch_size: int, rollout_size: int,
        main_metric: str, device: str, use_wandb: bool, 
        plot_trajectories: bool, save_to_csv=False
        ):
        """
        Function for training a neural network model

        Args:
            :model: (torch.nn.Module) Robot model 
            :train_data: (list) list of train RobotDatasets
            :val_data: (list) list of validation RobotDatasets
            :epochs_num: (int) number of epochs
            :batch_size: (int) batch size
            :rollout_size: (int) rollout length (number of samples) used in predict_multi_step
            :main_metric: (str) the name of the key metric by which the best model is selected
            :device: (str) placement device cuda / cpu
            :use_wandb: (bool) flag that wandb logging is used
        """

        best_main_metric = None  
        best_state_dict = None  
        rollout_size_max = rollout_size
        history = { 
            'loss': [],
            'val_loss': []
        } 

        optimizer = model.get_optimizer()
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

        loss_fn = model.get_loss_fn() 
        bar = tqdm(range(epochs_num))      # progress bar
        for epoch in bar:
            train_loss = 0
            val_loss  = 0
            model.train()
            rollout_size = int(min((2 * epoch) / epochs_num, 1.0) * rollout_size_max)
            rollout_size = 2 if rollout_size < 2 else rollout_size

            N_iters = sum([len(ds) for ds in train_data]) // batch_size + 1
            for _ in range(N_iters):
                batch_x, batch_u, batch_dt = self.sample_train_batch(train_data, batch_size, rollout_size)
                batch_x = batch_x.to(device)
                batch_u = batch_u.to(device)
                batch_dt = batch_dt.to(device)

                predicted_traj = self.predict_multi_step(
                    model,
                    batch_x[:, 0, :],     # [batch_size, robot_state]
                    batch_u[:, 0:-1, :],  # [batch_size, rollout_size-1, control]
                    batch_dt,             # [batch_size, rollout_size-1, 1] 
                    rollout_size
                )
                loss = loss_fn(predicted_traj, batch_x)

                # Do backpropagation
                loss.backward()
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.cpu().detach().numpy() / N_iters
            
            # switch model to eval mode
            model.eval()

            with torch.no_grad():
                custom_metrics_results = dict()
                for i in range(len(val_data)):
                    traj = val_data[i]
                    batch_x = traj.data_x[None]
                    batch_u = traj.data_u[None]
                    batch_dt = self.calculate_delta_time(traj.data_t[None])

                    batch_x = batch_x.to(device)
                    batch_u = batch_u.to(device)
                    batch_dt = batch_dt.to(device)

                    predicted_traj = self.predict_multi_step(
                        model,
                        batch_x[:, 0, :],
                        batch_u[:, 0:-1, :],
                        batch_dt,
                        len(traj.data_x)
                    )

                    val_loss, custom_metrics_results = self.calc_loss_and_metrics(
                        model, traj, predicted_traj, val_loss,
                        custom_metrics_results, device=device,
                        dataset_type="val", name=i,
                        plot_trajectories=plot_trajectories,
                        use_wandb=use_wandb,
                        save_to_csv=save_to_csv
                    )
                val_loss = val_loss / len(val_data)

                history['loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                for key in custom_metrics_results:
                    if key not in history.keys():
                        history[key] = list()
                    history[key].append(np.array(custom_metrics_results[key]).mean())

                if best_main_metric is None or history[main_metric][-1] < best_main_metric:
                    best_main_metric = history[main_metric][-1]
                    best_state_dict = copy.deepcopy(model.state_dict())

            bar.set_postfix(
                train_loss=train_loss,
                val_loss=val_loss, 
                main_metric=history[main_metric][-1]
            )
            if use_wandb:
                wandb.log({key:history[key][-1] for key in history})
        
        print("best_main_metric = {}".format(best_main_metric))
        model.load_state_dict(best_state_dict)       

    def evaluate(
        self,
        model,
        test_data: list,
        device: str,
        plot_trajectories=False,
        use_wandb=False,
        save_to_csv=False
        ):
        """
        Provides the data for neural networks to make prediction with.
        The data has not been used for training before and has not been validated.
        Trajectory plots are made using received predictions

        Args:
            :model: (RobotModel) Robot model
            :test_data: (list of RobotDatasets) list of test datasets
            :device: (str) cuda / cpu
            :plot_trajectories: (bool) if true, plot validation trajectories
            :use_wandb: (bool) if true then saves trajectory graphs to wandb
            :save_to_csv: (bool) if true then saves trrajectory data to csv-files
        """
        model.eval()
        loss_fn = model.get_loss_fn()
        test_loss = 0
        custom_metrics_results = {}
        for i in range(len(test_data)):
            traj = test_data[i]
            batch_x = traj.data_x[None]  
            batch_u = traj.data_u[None] 
            batch_dt = self.calculate_delta_time(traj.data_t[None])

            batch_x = batch_x.to(device)
            batch_u = batch_u.to(device)
            batch_dt = batch_dt.to(device)

            predicted_traj = self.predict_multi_step(
                model,
                batch_x[:, 0, :],
                batch_u[:, 0:-1, :],
                batch_dt,
                len(traj.data_x)
            )


            test_loss, custom_metrics_results = self.calc_loss_and_metrics(
                model, traj, predicted_traj, test_loss,
                custom_metrics_results, device=device, dataset_type="test", name=i,
                plot_trajectories=plot_trajectories,
                use_wandb=use_wandb,
                save_to_csv=save_to_csv
            )

        if use_wandb:
            test_loss = test_loss / len(test_data)
            wandb.run.summary["test_loss"] = test_loss
            df = pd.DataFrame(data=custom_metrics_results, dtype=np.dtype("float"))
            table = wandb.Table(dataframe=df)
            wandb.log({"TEST Metrics": table})

            for metric in df:
                wandb.run.summary["test mean {}".format(metric)] = df[metric].mean()

    def calc_loss_and_metrics(
        self,
        model,
        traj,
        predicted_traj,
        loss=0.,
        custom_metrics_results={},
        device='cpu',
        dataset_type=" ",
        name='0',
        plot_trajectories=False,
        use_wandb=False,
        save_to_csv=False
        ):
        """
        The function of calculating the 
        error (loss) and custom metrics of the model
        
        Args:
            :model: (RobotModel) Robot model
            :traj: (RobotDataset) Single trajectory dataset
            :predicted_traj: (torch.tensor of shape [batch size, num_samples, robot_state])
                the trajectory predicted by the neural network
            :loss: (float) Current loss value
            :custom_metrics_results: (dict) dictionary with custom metrics
            :device: (str) cuda or cpu
            :dataset_type: (str) Dataset type, such as test or validation,
                 will be used in the filename
            :name: (str or int) Traj name or number
            :plot_trajectories: (bool) if true, plot validation trajectories
            :use_wandb: (bool) if true then saves trajectory graphs to wandb
            :save_to_csv: (bool) if true then saves trrajectory data to csv-files
        Return:
            :loss: (float) loss value + initial loss value
            :custom_metrics_results: (dict) dictionary with custom metrics
        """
        ground_truth = traj.data_x[None]  
        ground_truth = ground_truth.to(device)
        loss_fn = model.get_loss_fn()

        loss += loss_fn(predicted_traj, ground_truth).cpu().detach().numpy()
        
        if plot_trajectories:
            fig = model.plot_trajectories(
                predicted_traj[0].cpu().detach().numpy(),
                traj
            )
            if use_wandb:
                wandb.log({'{} graph {}'.format(dataset_type, name): wandb.Image(fig)})
        
        metrics = model.calc_metrics(predicted_traj, ground_truth)
        for key in metrics: 
            if key not in custom_metrics_results.keys():
                custom_metrics_results[key] = list()
            custom_metrics_results[key].append(metrics[key])

        if save_to_csv:
            if use_wandb:
                path_ = wandb.run.dir + '/{}_{}'.format(dataset_type, name)
            else:
                path_ = "/{}".format(name)
            model.save_predict_to_csv( 
                predicted_traj[0].cpu().detach().numpy(),
                traj,
                path_
            )

        return loss, custom_metrics_results

    def sample_train_batch(self, data : list, batch_size: int, rollout_size:  int):
        """
        Selects from a random dataset, a rollout size segment.
        Then makes a batch from all segments.
        
        Args:
            :data: (list) list of RobotDatasets
            :batch_size: (int) batch size 
            :rollout_size: (int) rollout length (number of samples) used in predict_multi_step
        Retrun:
            :batch_x: (torch.tensor) Robot State batch 
            :batch_u: (torch.tensor) control batch
            :batch_dt: (torch.tensor) time delta batch
        """
        batch_x_data_size = data[0].data_x.shape[1] # int
        batch_u_data_size = data[0].data_u.shape[1] # int
        batch_t_data_size = 1

        # [Batch size, rollout_size, robot state]
        batch_x = torch.zeros([batch_size, rollout_size, batch_x_data_size])
        # [Batch size, rollout_size, control]
        batch_u = torch.zeros([batch_size, rollout_size, batch_u_data_size])
        # [Batch size, rollout_size, 1]
        batch_t = torch.zeros([batch_size, rollout_size, batch_t_data_size])
        
        for i in range(batch_size):
            # random dataset from all datasets
            n = np.random.randint(0,len(data))                         
            # random point in dataset     
            m = np.random.randint(0,len(data[n].data_x) - rollout_size)
            # slice of data
            batch_x[i] = data[n].data_x[m: m + rollout_size]
            batch_u[i] = data[n].data_u[m: m + rollout_size]
            batch_t[i] = data[n].data_t[m: m + rollout_size]

        batch_dt = self.calculate_delta_time(batch_t)
        return batch_x, batch_u, batch_dt, 


    # PREDICT_MULTI_STEP
    def predict_multi_step(self, model, init_state, batch_u, batch_dt, rollout_size):
        """
        Gives the neural network a sequence of true control and the initial state of the robot.
        Then makes a prediction of the robot's state rollout_size times.
        The initial (given) position is used only for the first time,
        then the predictions of the neural network are used.
        The output is a trajectory. 
        
        Args:
            :init_state: (torch.tensor of shape [batch sizes, robot_state]) robot state tensor
            :batch_u: (torch.tensor of shape [batch sizes, time, control]) control tensor
            :batch_dt: (torch.tensor of shape [batch sizes, time, 1]) delta time tensor
        Return:
            :predicted_traj: (torch.tensor of shape [batch size, num_samples, robot_state])
                the trajectory predicted by the neural network
        """
        predicted_traj = [init_state]                  # shape [batch size, 1, robot_state]
        for i in range(0, rollout_size - 1):
            input_state = predicted_traj[i]            # shape [batch size, robot_state]
            control = batch_u[:, i, :]                 # shape [batch size, control]
            dt = batch_dt[:, i, :]                     # shape [batch size, 1]
            prediction = model.update_state(input_state, control, dt=dt)
            predicted_traj.append(prediction)

        predicted_traj = torch.cat([pred[:, None] for pred in predicted_traj], 1)                                       
        return predicted_traj

    def calculate_delta_time(self, time_seq):
        """
        Function for calculating the time delta

        Args:
            :time_seq: torch tensor of shape [batch_size, num_samples, 1]
        Return:
            :delta time: torch tensor of shape [batch_size, num_samples - 1, 1]
        """
        return time_seq[:, 1:, :] - time_seq[:, 0:-1, :]   

    def test_integration(self, model, input_datasets, device, plot_path=None, use_wandb=False):
        """
        Function for testing the integration function in the robot model.
        The function is like predict_multi_step,
        but instead of predicting the speed, it evaluates to a true value.

        Args:
            :model: (RobotModel) Robot model
            :input_datasets: (RobotDataset) Single trajectory dataset
            :device: (str) cuda or cpu
            :plot_path: (str) the path to the directory where you want to save the plots
            :use_wandb: (bool) if true then saves trajectory graphs to wandb
        """

        for j in range(len(input_datasets)):
            input_dataset = input_datasets[j]
            robot_state = input_dataset.data_x[None]
            delta_time = self.calculate_delta_time(input_dataset.data_t[None])
            batch_u = input_dataset.data_u[None]

            robot_state = robot_state.to(device)
            batch_u = batch_u.to(device)
            delta_time = delta_time.to(device)

            predicted_traj = [robot_state[:, 0, :]]
            
            for i in range(0, len(input_dataset.data_x) - 1):
                input_state = predicted_traj[i]
                control = batch_u[:, i, :]                 # shape [batch size, control]
                dt = delta_time[:, i, :]
                prediction = model.update_state(input_state, control, dt=dt, gt_velocities=robot_state[:, i, 3:])
                predicted_traj.append(prediction)
            
            predicted_traj = torch.cat([pred[:, None] for pred in predicted_traj], 1)                                       
            fig = model.plot_trajectories(
                predicted_traj[0].cpu().detach().numpy(),
                input_dataset
            )

            if use_wandb:
                wandb.log({'test_integration graph {}'.format(j): wandb.Image(fig)})

            if plot_path is not None:
                plt.savefig("{}/{}".format(plot_path, j))



