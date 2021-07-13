import argparse
import yaml
import wandb
import os


def parse_config(cfg_path):
    """
    Parses the received .yaml config file.

    :Args:
        :cfg_path: (str) The absolute path to the training config
    :Return:
        :config: (dict) training config (dictionary in which the
         keys are attributes)
    """
    config = argparse.Namespace()
    with open(cfg_path) as cfg:
        config.__dict__ = yaml.safe_load(cfg)
    return config


def init_wandb(run_name, config):
    """
    Initializes the wandb service. Save the given config for training.

    :Args:
        :run_name: (str) The name of this trainning attempt for wandb
        :config: (dict) training config (dictionary in which the
         keys are attributes)
    """
    wandb.init(
        project=config.project_name,
        entity=config.entity,
        config=config.__dict__
    )
    wandb.run.name = run_name + '-' + wandb.run.name.split('-')[-1]
