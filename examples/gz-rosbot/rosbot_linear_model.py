#!/usr/bin/python3
import torch
from robot_model import *

class RosbotLinearModel(RobotModel):
    def __init__(self, n_inputs=5, n_outputs=4, n_layers=0):
        """
        Args:
            n_inputs (int) - number of input parameters
            n_outputs (int) - number of output parameters
            n_layers (int) - number of layers
            hidden_size (int) - number of neurons
            activation_function (str) - Activation function elu or relu
        Attributes:
            initial_state (torch.tensor of shape [2])
              batch of [v, w]
        """
        super(RosbotModel, self).__init__()

        self.initial_state = torch.zeros([2])
        self.define_default_nn_model(n_inputs, n_outputs, n_layers)


    def update_state(self, state, control,  dt=0.03):
        """
        Args:
            state (torch.tensor of shape [batch, 5]
              batch of [x, y, yaw, v, w,]
            control: torch.tensor of shape [batch, 2]
              batch of [u_v, u_w] 
        outputs:
        - next_state torch.tensor of shape [batch, 5]
            batch of [x_next, y_next, yaw_next, v_next, w_next]
        """

        dt = dt * torch.ones(state.shape[0])[:, None]
        
        x_curr, y_curr, yaw_curr =  state[:,0:1], state[:,1:2], state[:,2:3] # shape [batch, 1]
        v_w_curr = state[:,3:]                                               # shape [batch, 2]

        inp = torch.cat([v_w_curr, control, dt], 1)                          # shape [batch, 5]
        predcted_velocities = self.layers(inp)                               # shape [batch, 2]
        v, w = predcted_velocities[:, 0:1], predcted_velocities[:, 1:]       # shape [batch, 1]
            
        yaw = yaw_curr + w * dt                                              # shape [batch, 1]
        x = x_curr + v * torch.cos(yaw)                                      # shape [batch, 1]
        y = y_curr + v * torch.sin(yaw)                                      # shape [batch, 1]

        next_state= torch.cat([
            x,
            y,
            yaw,
            predcted_velocities
        ], 1)
        return next_state                                                    # shape [batch, time, 5]

    def forward(self, inp):
        """
        Defines the computation performed at every call
        Args:
            inp (torch tensor of shape [batch, 5])
              batch of [x, y, yaw, v, w]
        """
        v_w = inp[:,:2]                                                      # shape [batch, 2]
        uv_uw = inp[:,2:4]                                                   # shape [batch, 2]
        alphas = self.layers(inp)                                            # shape [batch, n_outputs=4]
        alpha1 = torch.sigmoid(alphas[:,:2])                                 # shape [batch, 2]
        alpha2 = torch.nn.functional.elu(alphas[:,2:]) + 1                   # shape [batch, 2]
        new_v_w = v_w * alpha1 + uv_uw * alpha2                              # shape [batch, 2]
        return new_v_w