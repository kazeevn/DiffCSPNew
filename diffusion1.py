import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from tqdm import tqdm

from data_utils import lattice_params_to_matrix_torch
from utils import d_log_p_wrapped_normal, BetaScheduler, SigmaScheduler
from crystal_family import CrystalFamily
from cspnet import CSPNet


MAX_ATOMIC_NUM = 100


class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings).to(device)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()

        self.decoder = CSPNet()
        self.beta_scheduler = BetaScheduler(1000, 'cosine')
        self.sigma_scheduler = SigmaScheduler(1000, 0.005, 0.5)
        self.time_dim = 256
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.device = device
        self.crystal_family = CrystalFamily()

    def forward(self, batch):
        batch_size = batch.batch_size
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        lattices = self.crystal_family.de_so3(lattices)
        frac_coords = batch.frac_coords

        rand_x = torch.randn_like(frac_coords)

        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]

        rand_x_anchor = rand_x[batch.anchor_index]
        rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        ori_crys_fam = self.crystal_family.m2v(lattices)
        ori_crys_fam = self.crystal_family.proj_k_to_spacegroup(ori_crys_fam, batch.spacegroup)
        rand_crys_fam = torch.randn_like(ori_crys_fam)
        rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)
        input_crys_fam = c0[:, None] * ori_crys_fam + c1[:, None] * rand_crys_fam
        input_crys_fam = self.crystal_family.proj_k_to_spacegroup(input_crys_fam, batch.spacegroup)

        pred_crys_fam, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_crys_fam,
                                             batch.num_atoms, batch.batch)
        pred_crys_fam = self.crystal_family.proj_k_to_spacegroup(pred_crys_fam, batch.spacegroup)

        pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)

        tar_x_anchor = d_log_p_wrapped_normal(sigmas_per_atom * rand_x_anchor, sigmas_per_atom) / torch.sqrt(
            sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_crys_fam, rand_crys_fam)

        loss_coord = F.mse_loss(pred_x_proj, tar_x_anchor)

        loss = (
                1. * loss_lattice +
                1. * loss_coord)

        return {
            'loss': loss,
            'loss_lattice': loss_lattice,
            'loss_coord': loss_coord
        }

    @torch.no_grad()
    def sample(self, batch, step_lr=1e-5):
        batch_size = batch.batch_size

        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        crys_fam_T = torch.randn([batch_size, 6]).to(self.device)
        crys_fam_T = self.crystal_family.proj_k_to_spacegroup(crys_fam_T, batch.spacegroup)

        time_start = self.beta_scheduler.timesteps - 1

        l_T = self.crystal_family.v2m(crys_fam_T)

        x_T_all = torch.cat([x_T[batch.anchor_index], torch.ones(batch.ops.size(0), 1).to(x_T.device)],
                            dim=-1).unsqueeze(-1)  # N * 4 * 1

        x_T = (batch.ops @ x_T_all).squeeze(-1)[:, :3] % 1.  # N * 3

        traj = {time_start: {
            'num_atoms': batch.num_atoms,
            'atom_types': batch.atom_types,
            'frac_coords': x_T % 1.,
            'lattices': l_T,
            'crys_fam': crys_fam_T
        }}

        for t in tqdm(range(time_start, 0, -1)):
            times = torch.full((batch_size,), t, device=self.device)

            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            crys_fam_t = traj[t]['crys_fam']

            # Corrector

            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr / (sigma_norm * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2 * step_size)

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            pred_crys_fam, pred_x = self.decoder(time_emb, batch.atom_types, x_t, crys_fam_t, batch.num_atoms,
                                                 batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce='mean')[batch.anchor_index]

            pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

            crys_fam_t_minus_05 = crys_fam_t

            frac_coords_all = torch.cat(
                [x_t_minus_05[batch.anchor_index], torch.ones(batch.ops.size(0), 1).to(x_t_minus_05.device)],
                dim=-1).unsqueeze(-1)  # N * 4 * 1

            x_t_minus_05 = (batch.ops @ frac_coords_all).squeeze(-1)[:, :3] % 1.  # N * 3

            # Predictor

            rand_crys_fam = torch.randn_like(crys_fam_T)
            rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)
            ori_crys_fam = crys_fam_t
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            pred_crys_fam, pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, crys_fam_t, batch.num_atoms,
                                                 batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            crys_fam_t_minus_1 = c0 * (ori_crys_fam - c1 * pred_crys_fam) + sigmas * rand_crys_fam
            crys_fam_t_minus_1 = self.crystal_family.proj_k_to_spacegroup(crys_fam_t_minus_1, batch.spacegroup)

            pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce='mean')[batch.anchor_index]
            pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

            l_t_minus_1 = self.crystal_family.v2m(crys_fam_t_minus_1)

            frac_coords_all = torch.cat(
                [x_t_minus_1[batch.anchor_index], torch.ones(batch.ops.size(0), 1).to(x_t_minus_1.device)],
                dim=-1).unsqueeze(-1)  # N * 4 * 1

            x_t_minus_1 = (batch.ops @ frac_coords_all).squeeze(-1)[:, :3] % 1.  # N * 3

            traj[t - 1] = {
                'num_atoms': batch.num_atoms,
                'atom_types': batch.atom_types,
                'frac_coords': x_t_minus_1 % 1.,
                'lattices': l_t_minus_1,
                'crys_fam': crys_fam_t_minus_1
            }

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': batch.atom_types,
            'all_frac_coords': torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices': torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss = output_dict['loss']

        if loss.isnan():
            return None

        return loss
