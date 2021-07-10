#!/usr/bin/python3
import os
import math
import torch
import torch.nn as nn
# default NN model
import matplotlib.pyplot as plt
from mppi_train.defualt_nn_model import DefaultModel
import numpy as np

class RosbotModelLoss(nn.Module):
    def __init__(self):
        super(RosbotModelLoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predict, ground_trurh): 
        """
        Defines the computation performed at every call.
        Args:
            predict (torch.tensor of shape [batch, time, robot_state]): predicted trajectory
            ground_trurh (torch.tensor of shape [batch, time, robot_state]): ground truth trajectory
        Return:
            Calculated MSEloss by X and Y coordinates
        """
        return self.loss_fn(predict[:,:,0:2],  ground_trurh[:,:,0:2])




class RosbotModel(nn.Module):
    def __init__(self,
        n_layers=2,
        hidden_size=64,
        activation_function='elu',
        learning_rate = 0.002,
        model_type = "semilinear"
    ):
        """
        Args:
            :n_layers (int): number of layers
            :hidden_size (int): number of neurons
            :activation_function (str): activation function (elu or relu)
            :learning_rate (float): neural network learning rate parameter
            :model_type (str): nonlinear / linear / semilinear
        Attributes:
            :model (torch.nn..Module): Dynamic robot model based on a neural network
            :optim_lr: Neural network learning rate parameter
        """
        super(RosbotModel, self).__init__()

        self.model_type = model_type
        self.optim_lr = learning_rate

        if self.model_type == "linear":
            self.model = DefaultModel(
                5,
                6,
                n_layers,
                hidden_size,
                activation_function
            )
        elif self.model_type == "semilinear":
            self.model = DefaultModel(
                5,
                4,
                n_layers,
                hidden_size,
                activation_function
            )
            self.b_model = DefaultModel(
            5,
            2,
            n_layers,
            hidden_size,
            activation_function
            ) 
        elif self.model_type == "nonlinear":
            self.model = DefaultModel(
                5,
                2,
                n_layers,
                hidden_size,
                activation_function
            )

    def get_optimizer(self):
        """
        Creates and returns an optimizer for a neural network
        """
        return torch.optim.Adam(self.parameters(), lr=self.optim_lr)
    
    def get_loss_fn(self):
        """
        Creates and returns a loss function for a neural network
        """
        return RosbotModelLoss()

    def get_initial_state(self):
        """
        Creates and returns the initial state of the model
        """
        return torch.zeros([2])


    def update_state(self, state, control, dt=0.033, gt_velocities=None):
        """
        Args:
            :state (torch.tensor of shape [batch, 5]
              batch of [x, y, yaw, v, w]
            :control (torch.tensor of shape [batch, 2]): batch of [u_v, u_w]
            :dt (torch.tensor of shape [batch, 1] or float): time delta
        Return:
            :next_state (torch.tensor of shape [batch_size, 5]):
                batch of [x_next, y_next, yaw_next, v_next, w_next]
        """

        if isinstance(dt, float):
            dt = dt * torch.ones(state.shape[0], device=state.device)[:, None]
        
        x_curr, y_curr, yaw_curr =  state[:,0:1], state[:,1:2], state[:,2:3] # shape [batch, 1]
        v_w_curr = state[:,3:]                                               # shape [batch, 2]

        inp = torch.cat([v_w_curr, control, dt], 1)                          # shape [batch, 5]
        if gt_velocities is None:
            predicted_velocities = self(inp)                                 # shape [batch, 2]
        else:
            predicted_velocities = gt_velocities

        v, w = predicted_velocities[:, 0:1], predicted_velocities[:, 1:] # shape [batch, 1]
        yaw = yaw_curr + w * dt                                              # shape [batch, 1]
        mask = (yaw > math.pi) * (2 * math.pi)                               # shape [batch, 1]
        yaw -= mask                                                          # shape [batch, 1]
        mask = (yaw < -math.pi) * (2 * math.pi)                              # shape [batch, 1]
        yaw += mask                                                          # shape [batch, 1]
        x = x_curr + v * torch.cos(yaw) * dt                                 # shape [batch, 1]
        y = y_curr + v * torch.sin(yaw) * dt                                 # shape [batch, 1]

        next_state= torch.cat([
            x,
            y,
            yaw,
            predicted_velocities
        ], 1)
        
        return next_state                                                    # shape [batch, 5]

    def forward(self, inp): 
        """
        Defines the computation performed at every call.
        Args:
            :inp: (torch.tensor of shape [batch, n_inputs]) input tensor
        """
        # TODO abs input or input**2
        if self.model_type == "linear":
            v_w = inp[:,:2]                                                  # shape [batch, 2]
            uv_uw = inp[:,2:4]                                               # shape [batch, 2]
            alphas = self.model(inp**2)                                      # shape [batch, n_outputs=4]
            alpha1 = torch.sigmoid(alphas[:,:2])                             # shape [batch, 2]
            alpha2 = torch.nn.functional.elu(alphas[:,2:4]) + 1              # shape [batch, 2]
            beta = alphas[:,4:] * 0.01                                       # shape [batch, 2]
            new_v_w = v_w * alpha1 + uv_uw * alpha2 + beta                   # shape [batch, 2]
            return new_v_w
        elif self.model_type == "semilinear":
            v_w = inp[:,:2]                                                  # shape [batch, 2]
            uv_uw = inp[:,2:4]                                               # shape [batch, 2]
            alphas = self.model(inp**2)                                      # shape [batch, n_outputs=4]
            alpha1 = torch.sigmoid(alphas[:,:2])                             # shape [batch, 2]
            alpha2 = torch.nn.functional.elu(alphas[:,2:4]) + 1              # shape [batch, 2]
            beta = self.b_model(inp) * 0.01                                  # shape [batch, 2]
            new_v_w = v_w * alpha1 + uv_uw * alpha2 + beta                   # shape [batch, 2]
            return new_v_w
        elif self.model_type == "nonlinear":
            return self.model(inp)                                           # shape [batch, 2]

    def calc_metrics(self, predict, ground_trurh):
        """
        Calls functions to calculate prediction metrics
        Args:
            :predict: (torch.tensor of shape [batch size, time, robot_state])
                the trajectory predicted by the neural network
            :ground_trurh: (torch.tensor of shape [batch size, time, robot_state])
                the ground truth trajectory
        Return:
            :result: (dict) Dictionary with the results of calculating different metrics.
                The key is the name of the metric,
                and the value is the computed value of the metric.
        """
        result = dict()
        result['traj_ate'] = self.calc_ate(predict, ground_trurh)
        result['yaw_mae'] = self.calc_yaw_mae(predict, ground_trurh)
        return result


    def calc_ate(self, predict, ground_truth):
        """
        Calculates the average translation error for given prediction
        Args:
            :predict: (torch.tensor of shape [batch size, time, robot_state]) predicted trajectory 
            :ground_truth: (torch.tensor of shape [batch size, time, robot_state]) ground truth trajectory 
        Return:
            :err: (torch.tensor of shape [batch size, 1]) calculated error
        """
        with torch.no_grad():
            mse_x = torch.square(predict[:,:,0] - ground_truth[:,:,0]) # shape [batch size, time, 1]
            mse_y = torch.square(predict[:,:,1] - ground_truth[:,:,1]) # shape [batch size, time, 1]
            err = torch.mean(torch.sqrt(mse_x + mse_y)).cpu().detach().numpy() 
        return err

    def calc_yaw_mae(self, predict, ground_truth):
        """
        Calculates the yaw angle mean absolute error for given prediction
        Args:
            :predict: (torch.tensor of shape [batch size, time, robot_state]) predicted trajectory 
            :ground_truth: (torch.tensor of shape [batch size, time, robot_state]) ground truth trajectory 
        Return:
            :err: (torch.tensor of shape [batch size, 1]) calculated error
        """
        with torch.no_grad():
            err = torch.mean((torch.abs(predict[:,:,2] - ground_truth[:,:,2])))
        return err.cpu().detach().numpy()

    def plot_trajectories(self, predict, ground_truth_traj):
        """
        A helper function that takes the predicted and ground truth
        trajectory and plots them on the same graph.
        Optionally can save them
        Args:
            :predict: (torch.tensor of shape [batch size, time, robot_state])
                the trajectory predicted by the neural network
            :ground_trurh: (torch.tensor of shape [batch size, time, robot_state])
                the ground truth trajectory
        Return:
            :fig: (matplotlib.figure.Figure) Several plots on one figure       
        """
        ground_truth = ground_truth_traj.data_x.cpu().numpy()
        kinetic_model_traj = ground_truth_traj.data_k.cpu().numpy()
        time = ground_truth_traj.data_t.cpu().numpy()
        control = ground_truth_traj.data_u.cpu().numpy()

        fig, ax = plt.subplots(6, figsize=(7, 20)) # Width, height in inches
        ax[0].set_ylabel('m/s')
        ax[0].set_title("linear velocity and control")

        ax[0].plot(time[:,0], ground_truth[:,3], color='black', label='Robot V', ls='--')
        ax[0].plot(time[:,0], predict[:,3], color='red', label='Predict V')
        ax[0].plot(time[:,0], control[:,0], color='green', label='U_V')
        if kinetic_model_traj is not None:
            ax[0].plot(time[:,0], kinetic_model_traj[:,3], color='yellow', label='Kinetic model V')
        ax[0].legend(loc="lower right")

        # ax[1].set_aspect(1)
        ax[1].set_ylabel('m/s')
        ax[1].set_xlabel('t, sec')
        ax[1].set_title("angular velocity and control")

        ax[1].plot(time[:,0], ground_truth[:,4], color='black', label='Robot W', ls='--')
        ax[1].plot(time[:,0], predict[:,4], color='red', label='Predict W')
        ax[1].plot(time[:,0], control[:,1], color='green', label='U_W')
        if kinetic_model_traj is not None:
            ax[1].plot(time[:,0], kinetic_model_traj[:,4], color='yellow', label='Kinetic model W')
        ax[1].legend(loc="lower right")

        ax[2].set_aspect(1)
        ax[2].set_ylabel('m')
        ax[2].set_xlabel('t, sec') 
        ax[2].set_title("X coord over time") 

        ax[2].plot(time[:,0], ground_truth[:,0], color='black', label='Robot X(t)', ls='--')
        ax[2].plot(time[:,0], predict[:,0], color='red', label='Predict X(t)')
        if kinetic_model_traj is not None:
            ax[2].plot(time[:,0], kinetic_model_traj[:,0], color='yellow', label='Kinetic model X(t)')
        ax[2].legend(loc="lower right")

        # ax[3].set_aspect(1)
        ax[3].set_ylabel('m')
        ax[3].set_xlabel('t, sec')  
        ax[3].set_title("Y coord over time")

        ax[3].plot(time[:,0], ground_truth[:,1], color='black', label='Robot Y(t)', ls='--')
        ax[3].plot(time[:,0], predict[:,1], color='red', label='Predict Y(t)')
        if kinetic_model_traj is not None:
            ax[3].plot(time[:,0], kinetic_model_traj[:,1], color='yellow', label='Kinetic model Y(t)')
        ax[3].legend(loc="lower right")

        # ax[4].set_aspect(1)
        ax[4].set_ylabel('Rads')
        ax[4].set_xlabel('t, sec')  
        ax[4].set_title("Yaw angle over time")

        ax[4].plot(time[:,0], ground_truth[:,2], color='black', label='Robot Yaw(t)', ls='--')
        ax[4].plot(time[:,0], predict[:,2], color='red', label='Predict Yaw(t)')
        if kinetic_model_traj is not None:
            ax[4].plot(time[:,0], kinetic_model_traj[:,2], color='yellow', label='Kinetic model Yaw(t)')
        ax[4].legend(loc="lower right")

        # ax[5].set_aspect(1)
        ax[5].set_ylabel('Y, m')
        ax[5].set_xlabel('X, m')  
        ax[5].set_title("XY trajectory")

        ax[5].plot(ground_truth[:,0], ground_truth[:,1], color='black', label='Robot x_y', ls='--')
        ax[5].plot(predict[:,0], predict[:,1], color='red', label='Predict x_y')
        if kinetic_model_traj is not None:
            ax[5].plot(kinetic_model_traj[:,0], kinetic_model_traj[:,1], color='yellow', label='Kinetic model x_y')
        ax[5].legend(loc="lower right")
        # plt.subplots_adjust(wspace=3, hspace=3)
        return fig

    def save_predict_to_csv(self, predict, ground_truth_traj, path):
        """
        Stores neural network prediction and ground truth data in csv format.
        Args:
            :predict: (torch.tensor of shape [batch size, num_samples, robot_state])
                the trajectory predicted by the neural network
            :ground_truth_traj: (RosbotDataset) Single trajectory dataset
            :path: (str) file path
        """
        ground_truth = ground_truth_traj.data_x.cpu().detach().numpy()
        kinetic_model_traj = ground_truth_traj.data_k.cpu().detach().numpy()
        control =  ground_truth_traj.data_u.cpu().detach().numpy()
        time_seq = ground_truth_traj.data_t.cpu().detach().numpy()

        if not os.path.exists(path):
            os.mkdir(path)

        np.savetxt(
            path + "/state.csv",
            ground_truth,
            header='x y yaw v w'
        )
        np.savetxt(
            path + "/nn_model_state.csv",
            predict,
            header='x y yaw v w'
        )
        np.savetxt(
            path + "/kinetic_model_state.csv",
            kinetic_model_traj,
            header='x y yaw v w'
        )
        np.savetxt(
            path + "/control.csv",
            control,
            header='x yaw'
        )
        np.savetxt(
            path + "/time.csv",
            time_seq,
            header='t'
        )




