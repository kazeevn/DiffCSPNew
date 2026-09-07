"""Space-group constrained diffusion model for crystal structure prediction."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from diffcsp.core.crystal_family import CrystalFamily
from diffcsp.core.scatter import scatter
from diffcsp.core.schedulers import (
    BetaScheduler,
    SigmaScheduler,
    SinusoidalTimeEmbeddings,
    d_log_p_wrapped_normal,
)
from diffcsp.data.transforms import lattice_params_to_matrix_torch
from diffcsp.models.cspnet import CSPNet


class CSPDiffusion(nn.Module):
    """Symmetry-preserving diffusion model for crystal coordinates and lattices."""

    def __init__(
        self,
        device: str | torch.device = "cpu",
        decoder: nn.Module | None = None,
        time_dim: int = 256,
        timesteps: int = 1000,
        beta_scheduler_mode: str = "cosine",
        sigma_begin: float = 0.005,
        sigma_end: float = 0.5,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.time_dim = time_dim
        self.decoder = decoder if decoder is not None else CSPNet()
        self.beta_scheduler = BetaScheduler(timesteps, beta_scheduler_mode)
        self.sigma_scheduler = SigmaScheduler(timesteps, sigma_begin, sigma_end)
        self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
        self.crystal_family = CrystalFamily()

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Calculates diffusion training loss with Wyckoff symmetry projection.

        Args:
            batch: PyTorch Geometric Data/Batch instance.

        Returns:
            Dictionary containing 'loss', 'loss_lattice', and 'loss_coord'.
        """
        batch_size = batch.batch_size if hasattr(batch, "batch_size") else batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1.0 - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        lattices = self.crystal_family.de_so3(lattices)
        frac_coords = batch.frac_coords

        rand_x = torch.randn_like(frac_coords)
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]

        # Symmetrize coordinate noise across Wyckoff positions
        rand_x_anchor = rand_x[batch.anchor_index]
        rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.0

        ori_crys_fam = self.crystal_family.m2v(lattices)
        ori_crys_fam = self.crystal_family.proj_k_to_spacegroup(ori_crys_fam, batch.spacegroup)
        rand_crys_fam = torch.randn_like(ori_crys_fam)
        rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)
        input_crys_fam = c0[:, None] * ori_crys_fam + c1[:, None] * rand_crys_fam
        input_crys_fam = self.crystal_family.proj_k_to_spacegroup(input_crys_fam, batch.spacegroup)

        pred_crys_fam, pred_x = self.decoder(
            time_emb,
            batch.atom_types,
            input_frac_coords,
            input_crys_fam,
            batch.num_atoms,
            batch.batch,
        )
        pred_crys_fam = self.crystal_family.proj_k_to_spacegroup(pred_crys_fam, batch.spacegroup)
        pred_x_proj = torch.einsum("bij, bj -> bi", batch.ops_inv, pred_x)

        tar_x_anchor = d_log_p_wrapped_normal(sigmas_per_atom * rand_x_anchor, sigmas_per_atom) / torch.sqrt(
            sigmas_norm_per_atom
        )

        loss_lattice = F.mse_loss(pred_crys_fam, rand_crys_fam)
        loss_coord = F.mse_loss(pred_x_proj, tar_x_anchor)
        total_loss = loss_lattice + loss_coord

        return {
            "loss": total_loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
        }

    @torch.no_grad()
    def sample(
        self, batch: Any, step_lr: float = 1e-5, disable_progress: bool = False
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Predictor-Corrector sampling algorithm for crystal structure generation."""
        batch_size = batch.batch_size if hasattr(batch, "batch_size") else batch.num_graphs
        x_t = torch.rand([batch.num_nodes, 3], device=self.device)
        crys_fam_t = torch.randn([batch_size, 6], device=self.device)
        crys_fam_t = self.crystal_family.proj_k_to_spacegroup(crys_fam_t, batch.spacegroup)

        time_start = self.beta_scheduler.timesteps - 1
        l_t = self.crystal_family.v2m(crys_fam_t)

        x_t_all = torch.cat(
            [x_t[batch.anchor_index], torch.ones((batch.ops.size(0), 1), device=self.device)], dim=-1
        ).unsqueeze(-1)
        x_t = (batch.ops @ x_t_all).squeeze(-1)[:, :3] % 1.0

        traj = {
            time_start: {
                "num_atoms": batch.num_atoms,
                "atom_types": batch.atom_types,
                "frac_coords": x_t % 1.0,
                "lattices": l_t,
                "crys_fam": crys_fam_t,
            }
        }

        pbar = range(time_start, 0, -1)
        if not disable_progress:
            pbar = tqdm(pbar, desc="Diffusion Sampling", leave=False)

        for t in pbar:
            times = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm_val = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1.0 - alphas) / torch.sqrt(1.0 - alphas_cumprod)

            cur_x = traj[t]["frac_coords"]
            cur_crys_fam = traj[t]["crys_fam"]

            # --- Corrector Step ---
            rand_x = torch.randn_like(cur_x) if t > 1 else torch.zeros_like(cur_x)
            step_size = step_lr / (sigma_norm_val * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2.0 * step_size)

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            _, pred_x = self.decoder(
                time_emb, batch.atom_types, cur_x, cur_crys_fam, batch.num_atoms, batch.batch
            )
            pred_x = pred_x * torch.sqrt(sigma_norm_val)

            pred_x_proj = torch.einsum("bij, bj -> bi", batch.ops_inv, pred_x)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce="mean")[batch.anchor_index]
            pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

            x_half = cur_x - step_size * pred_x + std_x * rand_x
            frac_coords_all = torch.cat(
                [x_half[batch.anchor_index], torch.ones((batch.ops.size(0), 1), device=self.device)], dim=-1
            ).unsqueeze(-1)
            x_half = (batch.ops @ frac_coords_all).squeeze(-1)[:, :3] % 1.0

            # --- Predictor Step ---
            rand_crys_fam = torch.randn_like(cur_crys_fam)
            rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)
            rand_x = torch.randn_like(cur_x) if t > 1 else torch.zeros_like(cur_x)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size_pred = sigma_x**2 - adjacent_sigma_x**2
            std_x_pred = torch.sqrt(
                torch.clamp(
                    (adjacent_sigma_x**2 * (sigma_x**2 - adjacent_sigma_x**2)) / (sigma_x**2), min=0.0
                )
            )

            rand_x_anchor = rand_x[batch.anchor_index]
            rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            pred_crys_fam, pred_x = self.decoder(
                time_emb, batch.atom_types, x_half, cur_crys_fam, batch.num_atoms, batch.batch
            )
            pred_x = pred_x * torch.sqrt(sigma_norm_val)

            crys_fam_next = c0 * (cur_crys_fam - c1 * pred_crys_fam) + sigmas * rand_crys_fam
            crys_fam_next = self.crystal_family.proj_k_to_spacegroup(crys_fam_next, batch.spacegroup)

            pred_x_proj = torch.einsum("bij, bj -> bi", batch.ops_inv, pred_x)
            pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce="mean")[batch.anchor_index]
            pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

            x_next = x_half - step_size_pred * pred_x + std_x_pred * rand_x
            l_next = self.crystal_family.v2m(crys_fam_next)

            frac_coords_all = torch.cat(
                [x_next[batch.anchor_index], torch.ones((batch.ops.size(0), 1), device=self.device)], dim=-1
            ).unsqueeze(-1)
            x_next = (batch.ops @ frac_coords_all).squeeze(-1)[:, :3] % 1.0

            traj[t - 1] = {
                "num_atoms": batch.num_atoms,
                "atom_types": batch.atom_types,
                "frac_coords": x_next % 1.0,
                "lattices": l_next,
                "crys_fam": crys_fam_next,
            }

        traj_stack = {
            "num_atoms": batch.num_atoms,
            "atom_types": batch.atom_types,
            "all_frac_coords": torch.stack([traj[i]["frac_coords"] for i in range(time_start, -1, -1)]),
            "all_lattices": torch.stack([traj[i]["lattices"] for i in range(time_start, -1, -1)]),
        }
        return traj[0], traj_stack

    def training_step(self, batch: Any, batch_idx: int = 0) -> torch.Tensor | None:
        """Convenience training step wrapper."""
        outputs = self(batch)
        loss = outputs["loss"]
        if torch.isnan(loss):
            return None
        return loss
