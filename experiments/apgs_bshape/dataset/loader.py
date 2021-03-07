#!/usr/bin/env python3

import os
import torch
from combinators.utils import git_root

"""
FIXME: Refactoring needed: 
Hao commented out this original function, since currently I don't 
move all datasets to the git_root directory. So we will need to 
rewrite the current in-use function load_datapaths, such that
it by default will search for the git_root/data/bshape/
"""

# def datapaths(data_dir=f'{git_root()}/data/bshape/', subfolder=''):
#     return [os.path.join(data_dir, subfolder, f) for f in os.listdir(data_dir)]


def load_datapaths(data_dir, timesteps, num_objects):
    """
    return a list of data paths and the mean_shape tensor
    """
    mean_shape = torch.load(data_dir + 'mean_shape.pt') 
    data_paths = []
    for file in os.listdir(os.path.join(data_dir, 'video')):
        if file.endswith('.pt') and 'timesteps=%d-objects=%d' % (timesteps, num_objects) in file:
            data_paths.append(os.path.join(data_dir, 'video', file))
    if len(data_paths) == 0:
        raise ValueError('Empty data path list.')
    return data_paths, mean_shape