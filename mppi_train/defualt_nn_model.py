import torch
import torch.nn as nn
# -*- coding: utf-8 -*-
#licence removed for brevity

class DefaultModel(nn.Module):

    """
    """
    def __init__(        
        self,
        n_inputs=5,
        n_outputs=2,
        n_layers=2,
        hidden_size=64,
        activation_function='elu'
    ):
        """
        Args:
            n_inputs (int) - number of input parameters
            n_outputs (int) - number of output parameters
            n_layers (int) - number of layers
            hidden_size (int) - number of neurons
            activation_function (str) - Activation function elu or relu
        """
        super(DefaultModel, self).__init__()

        if n_layers == 0:
            self.layers = nn.Sequential( nn.Linear(n_inputs, n_outputs) )
        else:
            layers = [
                nn.Linear(n_inputs, hidden_size)
            ]
            # add input layer
            if activation_function == 'relu':
                layers.append(nn.ReLU())
            elif activation_function == 'elu':
                layers.append(nn.ELU())

            for _ in range(n_layers - 1):
                # add hidden layers
                layers.append(nn.Linear(hidden_size, hidden_size))
                if activation_function == 'relu':
                    layers.append(nn.ReLU())
                elif activation_function == 'elu':
                    layers.append(nn.ELU())
            
            # add output layer
            layers.append(nn.Linear(hidden_size, n_outputs))

            self.layers = nn.Sequential(*layers)

    def forward(self, inp): 
        """
        Defines the computation performed at every call.
        Args:
            inp (torch.tensor of shape [batch, n_inputs]) - input tensor
        """
        return self.layers(inp) 