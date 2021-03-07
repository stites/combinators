import torch
import time
import numpy as np
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import shuffle
from combinators.stochastic import Trace
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate
    
def set_seed(seed):
    import torch
    import numpy
    import random
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    
    
def density_all_instances(apg, models, frames, num_objects, num_sweeps, sample_size, timesteps, device, hmc_sampler=None, batch_size=10):
    log_ps = []
    fout = open('log_p.txt', 'a+')
    num_batches = frames.shape[0] // batch_size
    for b in tqdm(range(num_batches)):
        x = frames[b*batch_size : (b+1)*batch_size].repeat(sample_size, 1, 1, 1, 1).to(device)
        out = apg(c={'frames': x}, sample_dims=0, batch_dim=1, reparameterized=False)
        if hmc_sampler is not None and num_sweeps == 0:
            z_wheres, z_what = get_samples_for_hmc(out, timesteps)
            log_p = hmc_sampler.hmc_sampling(x, z_wheres, z_what)
        else:
            p_trace = Trace()
            for k,v in out.trace.items():
                if k != 'recon_%d_%d' % (timesteps-1, num_sweeps) \
                and k!= 'recon_opt_%d_%d' % (timesteps, num_sweeps):
                    p_trace.append(v, name=k)
            log_p = p_trace.log_joint(sample_dims=0,
                                        batch_dim=1,
                                        reparameterized=False).detach().cpu().mean()
        log_ps.append(log_p)

        time_end = time.time()
    log_p = torch.tensor(log_ps).mean()
    if hmc_sampler is not None and num_sweeps == 0:  
        print('HMC, hmc_steps=%d, LF=%d, T=%d, K=%d, log_p=%d' % (hmc_sampler.hmc_num_steps, hmc_sampler.lf_num_steps, timesteps, num_objects, log_p), file=fout, flush=True)
    else:
        print('T=%d, K=%d, sweeps=%d, log_p=%d' % (timesteps, num_objects, num_sweeps, log_p), file=fout, flush=True)
    fout.close()
    return log_p

def get_samples(out, sweeps, T):
    recon_key = 'recon_%d_%d' % (T, sweeps)
    recon_vals = out.trace[recon_key].dist.probs
    z_where_vals = []
    for t in range(T):
        z_where_vals.append(out.trace['z_where_%d_%d'%(t,sweeps)].value.unsqueeze(2))
    z_where_vals = torch.cat(z_where_vals, 2)
    return recon_vals.detach().cpu(), z_where_vals.detach().cpu()

def viz_samples(frames, rs, ws, num_sweeps, num_objects, object_size, fs=1, title_fontsize=12, lw=3, colors=['#AA3377', '#EE7733', '#009988', '#0077BB', '#BBBBBB', '#EE3377', '#DDCC77'], save=False):
    B, T, FP, _ = frames.shape
    recons_last = rs[0]
    z_wheres_last = ws[0].clone()
    z_wheres_last[:,:,:,1] =  z_wheres_last[:,:,:,1] * (-1)
    c_pixels_last = z_wheres_last
    c_pixels_last = (c_pixels_last + 1.0) * (FP - object_size) / 2. # B * T * K * D
    for b in range(B):
        num_cols = T
        num_rows =  2
        c_pixel_last, recon_last = c_pixels_last[b], recons_last[b]
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(left=0.05 , bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
        fig = plt.figure(figsize=(fs * num_cols, fs * num_rows))
        for c in range(num_cols):
            ax_infer = fig.add_subplot(gs[0, c])
            ax_infer.imshow(frames[b, c].numpy(), cmap='gray', vmin=0.0, vmax=1.0)
            ax_infer.set_xticks([])
            ax_infer.set_yticks([])
            for k in range(num_objects):
                rect_k = patches.Rectangle((c_pixel_last[c, k, :]), object_size, object_size, linewidth=lw, edgecolor=colors[k],facecolor='none')
                ax_infer.add_patch(rect_k)
            ax_recon = fig.add_subplot(gs[1, c])
            ax_recon.imshow(recon_last[c], cmap='gray', vmin=0.0, vmax=1.0)
            ax_recon.set_xticks([])
            ax_recon.set_yticks([])
    if save:
        plt.savefig('combinators_apg_samples.svg', dpi=300)