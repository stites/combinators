import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from experiments.apgs_bshape.affine_transformer import Affine_Transformer

class HMC():
    def __init__(self, decoder, frame_pixels, shape_pixels, S, B, T, K, z_where_dim, z_what_dim, hmc_num_steps, step_size_what, step_size_where, leapfrog_num_steps, device):
        self.decoder = decoder
        self.AT = Affine_Transformer(frame_pixels, shape_pixels, device)
        self.S = S
        self.B = B
        self.T = T
        self.K = K
        self.z_where_dim = z_where_dim
        self.z_what_dim = z_what_dim
        self.accept_count = 0.0
        self.smallest_accept_ratio = 0.0
        self.hmc_num_steps = hmc_num_steps
        self.lf_step_size_where = step_size_where
        self.lf_step_size_what = step_size_what
        self.lf_num_steps = leapfrog_num_steps
        self.uniformer = Uniform(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        self.gauss_dist = Normal(torch.zeros(1).to(device), torch.ones(1).to(device))

    def init_sample(self):
        """
        initialize auxiliary variables from univariate Gaussian
        return r_what, r_where
        """
        return self.gauss_dist.sample((self.S, self.B, self.T, self.K, self.z_where_dim,)).squeeze(-1), self.gauss_dist.sample((self.S, self.B, self.K, self.z_what_dim, )).squeeze(-1)

    
    def hmc_sampling(self, frames, z_where, z_what):
        self.accept_count = 0.0
        for m in range(self.hmc_num_steps):
            z_where, z_what = self.metrioplis(frames, z_where.detach(), z_what.detach())
            log_joint = self.log_joint(frames, z_where, z_what)
        self.smallest_accept_ratio = (self.accept_count / self.hmc_num_steps).min().item()
        if self.smallest_accept_ratio > 0.25: # adaptive leapfrog step size
            self.smallest_accept_ratio *= 1.005
        else:
            self.smallest_accept_ratio *= 0.995
        return log_joint.mean()

    
    def log_joint(self, frames, z_where, z_what):
        return self.decoder.hmc_log_joint(frames, z_where, z_what)

    def metrioplis(self, frames, z_where, z_what):
        r_where, r_what = self.init_sample()
        ## compute hamiltonian given original position and momentum
        H_orig = self.hamiltonian(frames, z_where, z_what, r_where, r_what)
        new_where, new_what, new_r_where, new_r_what = self.leapfrog(frames, z_where, z_what, r_where, r_what)
        ## compute hamiltonian given new proposals
        H_new = self.hamiltonian(frames, new_where, new_what, new_r_where, new_r_what)
        accept_ratio = (H_new - H_orig).exp()
        u_samples = self.uniformer.sample((self.S, self.B, )).squeeze(-1)
        accept_index = (u_samples < accept_ratio)
        # assert accept_index.shape == (self.S, self.B), "ERROR! index has unexpected shape."
        accept_index_expand1 = accept_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.K, self.z_what_dim)
        accept_index_expand2 = accept_index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.T, self.K, self.z_where_dim)
        filtered_z_where = new_where * accept_index_expand2.float() + z_where * (~accept_index_expand2).float()
        filtered_z_what = new_what * accept_index_expand1.float() + z_what * (~accept_index_expand1).float()
        self.accept_count = self.accept_count + accept_index.float()
        return filtered_z_where.detach(), filtered_z_what.detach()

    def leapfrog(self, frames, z_where, z_what, r_where, r_what):
        for step in range(self.lf_num_steps):
            z_where.requires_grad = True
            z_what.requires_grad = True
            log_p = self.log_joint(frames, z_where, z_what)
            log_p.sum().backward(retain_graph=False)

            r_where = (r_where + 0.5 * self.lf_step_size_where * z_where.grad).detach()
            r_what = (r_what + 0.5 * self.lf_step_size_what * z_what.grad).detach()
            z_where = (z_where + self.lf_step_size_where * r_where).detach()
            z_what = (z_what + self.lf_step_size_what * r_what).detach()
            z_where.requires_grad = True
            z_what.requires_grad = True
            log_p = self.log_joint(frames, z_where, z_what)
            log_p.sum().backward(retain_graph=False)
            r_where = (r_where + 0.5 * self.lf_step_size_where * z_where.grad).detach()
            r_what = (r_what + 0.5 * self.lf_step_size_what * z_what.grad).detach()
            z_where = z_where.detach()
            z_what = z_what.detach()
        return z_where, z_what, r_where, r_what

    def hamiltonian(self, frames, z_where, z_what, r_where, r_what):
        """
        compute the Hamiltonian given the position and momntum

        """
        Kp = self.kinetic_energy(r_where, r_what)
        Uq = self.log_joint(frames, z_where, z_what)
        assert Kp.shape == (self.S, self.B), "ERROR! Kp has unexpected shape."
        assert Uq.shape ==  (self.S, self.B), 'ERROR! Uq has unexpected shape.'
        return Kp + Uq

    def kinetic_energy(self, r_where, r_what):
        """
        r_tau, r_mu : S * B * K * D
        return - 1/2 * ||(r_tau, r_mu)||^2
        """
        return - ((r_where ** 2).sum(-1).sum(-1).sum(-1) + (r_what ** 2).sum(-1).sum(-1)) * 0.5