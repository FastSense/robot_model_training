#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mppi_train.trainer import Trainer
from rosbot_model import RosbotModel

# model_3kre9wom

def save_control_to_file(control, name):
    """
    Args:
        control (tensor shape [1,N,2]): selected control
        name (string): output file name (or path + name)
    """
    with open(name, 'w') as f:
        f.write("t x yaw \n")
        t = 0.0
        for item in control.detach().cpu().numpy()[0]:
            f.write(str(t) + " " + str(item[0]) + " " + str(item[1]) + "\n")
            t = t + 0.03

def save_state_to_file(state, name):
    """
    Args:
        state (tensor shape [1,N,4]): selected control
        name (string): output file name (or path + name)
    """
    with open(name, 'w') as f:
        f.write("t x y yaw v \n")
        t = 0.0
        for item in state.detach().cpu().numpy()[0]:
            f.write(
                str(t) + " " +
                str(item[0]) + " " +
                str(item[1]) + " " +
                str(item[2]) + " " + 
                str(item[3]) + "\n"
            )
            t = t + 0.03


def predict_by_control(control, model, trainer):
    """
    Args:
        control: torch.tensor of shape [1, time, 2]
                 [1, time, U_v, U_w]
        model: torch.nn.Module    
    Return:
        result_xya_v: torch.tensor of shape [batch, time, 4]
                     [batch, time, x, y, yaw, lin_velocity]

    """
    # (torch.tensor of shape [batch sizes, robot_state])
    init_state = torch.zeros([1,5])
    # Integrate control
    batch_dt = control = torch.ones([1, 999, 1]) * 0.033 # shape [batch_size, time, 2]
    rollout_size = 10
    print("init_state shape = {}".format(init_state.shape))
    print("control shape = {}".format(control.shape))
    print("batch_dt shape = {}".format(batch_dt.shape))
    result_xya_vw = trainer.predict_multi_step(
        model,
        init_state, 
        control[:, 0:-1, :], 
        batch_dt, 
        rollout_size
    )
    return result_xya_vw


def loss_for_square(result_xya_v):
    """
    Args:
        result_xya_v: torch.tensor of shape [batch, time, 4]
                     [batch, time, x, y, yaw, lin_velocity]
    Return:
        loss: torch.tensor of shape [1]
    """
    x = result_xya_v[0, :, 0]
    y = result_xya_v[0, :, 1]
    v = result_xya_v[0, :, 3]
    # First goal: [1, 0]
    loss_x_1 = (x[250] - 1) ** 2
    loss_y_1 = y[250] ** 2
    # Second goal: [1, 1]
    loss_x_2 = (x[500] - 1) ** 2
    loss_y_2 = (y[500] - 1) ** 2
    # Third goal: [0, 1]
    loss_x_3 = x[750] ** 2
    loss_y_3 = (y[750] - 1) ** 2
    # Last goal
    loss_x_4 = x[999] ** 2
    loss_y_4 = y[999] ** 2
    # Summary loss
    traj_loss = (torch.sqrt(loss_x_1 + loss_y_1) + torch.sqrt(loss_x_2 + loss_y_2) +
                 torch.sqrt(loss_x_3 + loss_y_3) + torch.sqrt(loss_x_4 + loss_y_4))
    loss = traj_loss + torch.abs(v).sum() * 0.03 * 0.1 # 0.1 default
    return loss


def loss_for_obstacle(result_xya_v, obstacle_boundaries=None):
    """
    Args:
        result_xya_v: torch.tensor of shape [batch, time, 4]
        obstacle_boundaries (list of lists): obstacle boundaries
    Return:
        loss (torch.tensor shape=[1]): loss for current run
    Note:
        obstacle_boundaries example  
            [[lower left corner coords], [upper right corner coords]] = [[0,1], [2,2]]  
    """
    x = result_xya_v[0, :, 0]
    y = result_xya_v[0, :, 1]

    left_cor = obstacle_boundaries[0]
    right_cor = obstacle_boundaries[1]

    # hinge loss 
    residual = torch.cat([
        (x - left_cor[0])[None],
        (right_cor[0] - x)[None],
        (y - left_cor[0])[None],
        (right_cor[1] - y)[None]
    ], 0)

    loss = torch.min(residual, 0).values
    loss = torch.clamp(loss, 0).sum()
    return loss


def loss_for_goal(result_xya_v, goal):
    """
    Args:
        result_xya_v: torch.tensor of shape [batch, time, 4]
        goal (list of 2 elements): coord of main goal 
    Return:
        goal_loss: torch.tensor of shape [1]
    """
    x = result_xya_v[0, :, 0]
    y = result_xya_v[0, :, 1]
    loss_x_goal = (x[999] - goal[0]) ** 2
    loss_y_goal = (y[999] - goal[1]) ** 2
    goal_loss = (torch.sqrt(loss_x_goal + loss_y_goal))
    return goal_loss


def calc_complex_loss(result_xya_v, goal, control, obstacle_boundaries=None):
    """
    Args:
        result_xya_v: torch.tensor of shape [batch, time, 4]
        goal (list of 2 elements): coord of main goal 
        control: torch.tensor of shape [1, time, 2]
        obstacle_boundaries (list of lists): obstacle boundaries
    Return:
        loss (torch.tensor): complex loss for goal, obstales and velocities
    """
    v = result_xya_v[0, :, 3]
    loss = torch.tensor(0.0)
    loss += (loss_for_goal(result_xya_v, goal) * 5)
    loss += (torch.abs(v).sum() * 0.03 * 0.1)  # loss for speed
    if obstacle_boundaries is not None:
        for obstacle in obstacle_boundaries:
            loss += (loss_for_obstacle(result_xya_v, obstacle) * 20)
    return loss

def draw_plots(result_xya_v, obsctacles, i):
    """

    """
    # Plot x, y
    fig, ax = plt.subplots()
    ax.plot(
        result_xya_v[0, :, 0].cpu().detach(),
        result_xya_v[0, :, 1].cpu().detach(),
        label='Predict'
    )
    ax.annotate(
        'Control', xy=(0, 1), xytext=(12, -12),
        va='top', xycoords='axes fraction',
        textcoords='offset points',
        bbox=dict(facecolor='none', edgecolor='black')
    )
    ax.legend(loc="lower right")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect("equal")
    if obsctacles is not None:
        for obs in obsctacles:
            # Create a Rectangle patch
            left_cor = obs[0]
            right_cor = obs[1]
            w = right_cor[0] - left_cor[0]
            h = right_cor[1] - left_cor[1]
            rect = patches.Rectangle(
                left_cor, w, h,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            # Add the patch to the Axes
            ax.add_patch(rect)
    plt.savefig(f'plots/optimized_{i}.png')


def main():
    """
    Create goa, obstacles, initial control and state
    try to optimize control for obstacles avoidance
    """
    trainer = Trainer()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RosbotModel(        
        n_layers=4,
        hidden_size=128,
        activation_function='relu',
        learning_rate=0.005,
        model_type='semilinear'
    )

    state_dict = torch.load('model_3kre9wom.pt')
    model.requires_grad_(False)
    model.load_state_dict(state_dict)
    # model = model.to(device)

    # Declare goal
    goal = [2.5, 5.0]
    # Declare obsctacle
    # for each obstacle declare: [[lower left corner coords], [upper right corner coords]]
    obsctacles = [[[0,1], [2,2]], [[3,1],[4,4]]] 
    obsctacles = None
    # Create initial control
    control = torch.zeros([1, 1000, 2]) # shape [batch_size, time, 2]
    # Create initial state
    # result_xya_v = torch.zeros([1, 1000, 5])
    # result_xya_v = predict_by_control(control, model, trainer)
    # Optimize control for moving by set goal, and obstacles


    """

    TEST

    batch_x.shape = torch.Size([1, 1809, 5])
    batch_u.shape = torch.Size([1, 1809, 2])
    batch_dt.shape = torch.Size([1, 1808, 1])
    """

    init_state = torch.zeros([1, 1000, 5])
    control = torch.zeros([1, 1000, 2])
    batch_dt = torch.ones([1, 1000, 1]) * 0.033
    rollout = 1000

    print("init_state.shape = {}".format(init_state.shape))
    print("control.shape = {}".format(control.shape))
    print("batch_dt.shape = {}".format(batch_dt.shape))


    predicted_traj = trainer.predict_multi_step(
        model,
        init_state[:, 0, :],
        control[:, 0:-1, :],
        batch_dt,
        rollout
    )

    control = control.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([control], lr=0.003)
    for i in range(1001):
        # result_xya_v = predict_by_control(control, model, trainer)
        # loss = calc_complex_loss(result_xya_v, goal, control, obsctacles)
        result_xya_v = trainer.predict_multi_step(
            model,
            init_state[:, 0, :],
            control[:, 0:-1, :],
            batch_dt,
            rollout
        )
        loss = loss_for_square(result_xya_v)
        loss.backward()
        opt.step()
        opt.zero_grad()

        if i % 10 == 0:
            print(i, loss.item())
        if i % 100 == 0:
            draw_plots(result_xya_v, obsctacles, i)

    save_control_to_file(control, "cotrol_square.txt")
    # save_state_to_file(result_xya_v, "result_xya_v.txt")


if __name__ == "__main__":
    main()
