import argparse
import os
import torch
import numpy as np
from experiments.apgs_bshape.models import init_models
from experiments.apgs_bshape.affine_transformer import Affine_Transformer
from experiments.apgs_bshape.main import train_apg
from experiments.apgs_bshape.gibbs import gibbs_sweeps
from combinators.utils import load_models
from experiments.apgs_bshape.hmc_sampler import HMC
from experiments.apgs_bshape.evaluation import density_all_instances

def rws_apg(args):
    use_markov_blanket = True

    data_dir = './dataset/'
    timesteps_train = 10
    frame_pixels = 96
    shape_pixels = 28
    num_objects_train = 3

   
    num_epochs = 1000
    lr = 2e-4
    batch_size = 1
    budget_train = 120
    num_sweeps_train = 5

    num_hidden_digit = 400
    num_hidden_coor = 400
    z_where_dim = 2
    z_what_dim = 10

    budget_test = 200
    sample_size_test = budget_test // (args.num_sweeps + 1)

    device = torch.device(args.device)
    sample_size = budget_train // (num_sweeps_train + 1)
    assert sample_size > 0, 'non-positive sample size =%d' % sample_size
    mean_shape = torch.load(data_dir + 'mean_shape.pt').to(device)    
    data_paths = []
    for file in os.listdir(data_dir+'/test_video/'):
        if file.endswith('.pt') and \
        'timesteps=%d-objects=%d' % (args.timesteps, args.num_objects) in file:
            data_paths.append(os.path.join(data_dir+'test_video/', file))
    if len(data_paths) == 0:
        raise ValueError('Empty data path list.')
    model_version = 'apg-timesteps=%d-objects=%d-sweeps=%d-samples=%d' % (timesteps_train, num_objects_train, num_sweeps_train, sample_size)
    models = init_models(frame_pixels, 
                         shape_pixels, 
                         num_hidden_digit, 
                         num_hidden_coor, 
                         z_where_dim, 
                         z_what_dim, 
                         args.num_objects, 
                         mean_shape, 
                         device,
                         use_markov_blanket=use_markov_blanket)

    load_models(models, 'running-model-1', map_location=lambda storage, loc: storage)
    for k, v in models.items():
        for p in list(v.parameters()):
            p.requires_grad = False

    frames = torch.load(data_paths[0])
    apg = gibbs_sweeps(models, args.num_sweeps, args.timesteps)

    log_p = density_all_instances(apg, 
                           models, 
                           frames, 
                           args.num_objects,
                           args.num_sweeps, 
                           sample_size_test, 
                           args.timesteps, 
                           device,
                           hmc_sampler=None,
                           batch_size=batch_size)
#     print(data_paths[0])
#     print('T=%d, K=%d, sweeps=%d, log_p=%d' % (args.timesteps, args.num_objects, args.num_sweeps, log_p))
#     print('log_p=%d, hmc_steps=%d, lf_steps=%d' % (args.hmc_steps, args.lf_steps))
if __name__ == '__main__':
    parser = argparse.ArgumentParser('rws_hmc')
    parser.add_argument('--num_sweeps', type=int)
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--num_objects', type=int)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()
    rws_apg(args)