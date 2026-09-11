"""Space-group constrained diffusion model operating on the Asymmetric Unit."""

import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from diffcsp.core.crystal_family import CrystalFamily
from diffcsp.core.schedulers import (
    BetaScheduler,
    SigmaScheduler,
    SinusoidalTimeEmbeddings,
    d_log_p_wrapped_normal,
)
from diffcsp.data.transforms import lattice_params_to_matrix_torch
from diffcsp.models.wyckoff_cspnet import WyckoffCSPNet
from diffcsp.models.layers import generate_asymmetric_edges


class WyckoffDiffusion(nn.Module):
    """Symmetry-preserving diffusion model denoising Wyckoff site anchors directly.

    Rather than instantiating all N atoms in the conventional cell with O(N^2) edges,
    WyckoffDiffusion denoises the K unique Wyckoff sites (the asymmetric unit)
    using Wyckoff-to-Wyckoff multi-edges.

    Reference:
        Innovation 1, docs/architectural-innovations.md
    """

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
        self.decoder = decoder if decoder is not None else WyckoffCSPNet()
        self.beta_scheduler = BetaScheduler(timesteps, beta_scheduler_mode)
        self.sigma_scheduler = SigmaScheduler(timesteps, sigma_begin, sigma_end)
        self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
        self.crystal_family = CrystalFamily()

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Calculates asymmetric unit diffusion loss.

        Args:
            batch: PyTorch Geometric Data/Batch instance.

        Returns:
            Dictionary with 'loss', 'loss_lattice', and 'loss_coord'.
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

        # Extract asymmetric unit (unique Wyckoff site anchors)
        unique_anchors, inverse_site_map = torch.unique(batch.anchor_index, return_inverse=True, sorted=True)
        site2graph = batch.batch[unique_anchors]
        num_sites = torch.bincount(site2graph, minlength=batch_size)

        site_coords = batch.frac_coords[unique_anchors]
        site_atom_types = batch.atom_types[unique_anchors]
        site_ops = batch.ops[unique_anchors]
        site_P = site_ops[:, :3, :3]
        site_x0 = site_ops[:, :3, 3]
        site_dofs = torch.round(torch.diagonal(site_P, dim1=-2, dim2=-1).sum(-1))

        sigmas_per_site = sigmas.repeat_interleave(num_sites)[:, None]
        sigmas_norm_per_site = sigmas_norm.repeat_interleave(num_sites)[:, None]

        # Innovation 2: Tangent-Space Subspace Coordinate Parameterization
        # Project Gaussian noise strictly onto tangent space T_k; 0-DoF sites receive zero noise
        site_eps = torch.randn_like(site_coords)
        rand_site_x = (site_P @ site_eps.unsqueeze(-1)).squeeze(-1)
        input_site_coords = (site_coords + sigmas_per_site * rand_site_x) % 1.0

        # Expand noisy site coordinates to full unit cell for multi-edge computation
        input_full_coords = (
            batch.ops[:, :3, :3] @ input_site_coords[inverse_site_map].unsqueeze(-1)
        ).squeeze(-1) + batch.ops[:, :3, 3]
        input_full_coords = input_full_coords % 1.0

        # Lattice diffusion
        ori_crys_fam = self.crystal_family.m2v(lattices)
        ori_crys_fam = self.crystal_family.proj_k_to_spacegroup(ori_crys_fam, batch.spacegroup)
        rand_crys_fam = torch.randn_like(ori_crys_fam)
        rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)
        input_crys_fam = c0[:, None] * ori_crys_fam + c1[:, None] * rand_crys_fam
        input_crys_fam = self.crystal_family.proj_k_to_spacegroup(input_crys_fam, batch.spacegroup)

        pred_crys_fam, pred_site_x = self.decoder(
            time_emb,
            site_atom_types,
            input_site_coords,
            input_full_coords,
            input_crys_fam,
            num_sites,
            batch.num_atoms,
            site2graph,
            inverse_site_map,
            site_projectors=site_P,
            batch_ops=batch.ops,
        )
        pred_crys_fam = self.crystal_family.proj_k_to_spacegroup(pred_crys_fam, batch.spacegroup)

        # Target score in tangent space
        raw_tar_site_x = d_log_p_wrapped_normal(
            sigmas_per_site * rand_site_x, sigmas_per_site
        ) / torch.sqrt(sigmas_norm_per_site)
        tar_site_x = (site_P @ raw_tar_site_x.unsqueeze(-1)).squeeze(-1)

        loss_lattice = F.mse_loss(pred_crys_fam, rand_crys_fam)

        # Innovation 2: Compute coordinate loss strictly on free degrees of freedom (DoF > 0)
        # Multiplicity weighting so Wyckoff loss matches full-cell MSE atom scaling
        free_dof_mask = (site_dofs > 0)
        free_atom_mask = free_dof_mask[inverse_site_map]
        if free_atom_mask.any():
            loss_coord = F.mse_loss(
                pred_site_x[inverse_site_map][free_atom_mask],
                tar_site_x[inverse_site_map][free_atom_mask],
            )
        else:
            loss_coord = torch.tensor(0.0, device=self.device)
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
        """Predictor-Corrector sampling operating on the Asymmetric Unit."""
        batch_size = batch.batch_size if hasattr(batch, "batch_size") else batch.num_graphs

        unique_anchors, inverse_site_map = torch.unique(batch.anchor_index, return_inverse=True, sorted=True)
        site2graph = batch.batch[unique_anchors]
        num_sites = torch.bincount(site2graph, minlength=batch_size)
        site_atom_types = batch.atom_types[unique_anchors]
        site_ops = batch.ops[unique_anchors]
        site_P = site_ops[:, :3, :3]
        site_x0 = site_ops[:, :3, 3]

        # Precompute edge topology once for all 1,000 steps
        target_sites, source_atoms = generate_asymmetric_edges(num_sites, batch.num_atoms, self.device)
        source_sites = inverse_site_map[source_atoms]
        edge2graph = site2graph[target_sites]

        # Innovation 2: Initialize coordinates strictly in the affine tangent subspace
        u = torch.rand([len(unique_anchors), 3], device=self.device)
        site_x_t = ((site_P @ u.unsqueeze(-1)).squeeze(-1) + site_x0) % 1.0

        crys_fam_t = torch.randn([batch_size, 6], device=self.device)
        crys_fam_t = self.crystal_family.proj_k_to_spacegroup(crys_fam_t, batch.spacegroup)

        time_start = self.beta_scheduler.timesteps - 1
        pbar = range(time_start, 0, -1)
        if not disable_progress:
            pbar = tqdm(pbar, desc="Wyckoff Diffusion Sampling", leave=False)

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

            full_x = (
                batch.ops[:, :3, :3] @ site_x_t[inverse_site_map].unsqueeze(-1)
            ).squeeze(-1) + batch.ops[:, :3, 3]
            full_x = full_x % 1.0

            # --- Corrector Step ---
            # Tangent-space Brownian noise
            rand_x_raw = torch.randn_like(site_x_t) if t > 1 else torch.zeros_like(site_x_t)
            rand_x = (site_P @ rand_x_raw.unsqueeze(-1)).squeeze(-1)

            step_size = step_lr / (sigma_norm_val * (self.sigma_scheduler.sigma_begin) ** 2)
            std_x = torch.sqrt(2.0 * step_size)

            _, pred_x = self.decoder(
                time_emb,
                site_atom_types,
                site_x_t,
                full_x,
                crys_fam_t,
                num_sites,
                batch.num_atoms,
                site2graph,
                inverse_site_map,
                target_sites=target_sites,
                source_atoms=source_atoms,
                source_sites=source_sites,
                edge2graph=edge2graph,
                site_projectors=site_P,
                batch_ops=batch.ops,
            )
            pred_x = pred_x * torch.sqrt(sigma_norm_val)

            site_x_half = site_x_t - step_size * pred_x + std_x * rand_x
            site_x_half = ((site_P @ site_x_half.unsqueeze(-1)).squeeze(-1) + site_x0) % 1.0

            # --- Predictor Step ---
            rand_crys_fam = torch.randn_like(crys_fam_t)
            rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)

            rand_x_raw = torch.randn_like(site_x_t) if t > 1 else torch.zeros_like(site_x_t)
            rand_x = (site_P @ rand_x_raw.unsqueeze(-1)).squeeze(-1)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size_pred = sigma_x**2 - adjacent_sigma_x**2
            std_x_pred = torch.sqrt(
                torch.clamp(
                    (adjacent_sigma_x**2 * (sigma_x**2 - adjacent_sigma_x**2)) / (sigma_x**2), min=0.0
                )
            )

            full_x_half = (
                batch.ops[:, :3, :3] @ site_x_half[inverse_site_map].unsqueeze(-1)
            ).squeeze(-1) + batch.ops[:, :3, 3]
            full_x_half = full_x_half % 1.0

            pred_crys_fam, pred_x = self.decoder(
                time_emb,
                site_atom_types,
                site_x_half,
                full_x_half,
                crys_fam_t,
                num_sites,
                batch.num_atoms,
                site2graph,
                inverse_site_map,
                target_sites=target_sites,
                source_atoms=source_atoms,
                source_sites=source_sites,
                edge2graph=edge2graph,
                site_projectors=site_P,
                batch_ops=batch.ops,
            )
            pred_x = pred_x * torch.sqrt(sigma_norm_val)

            crys_fam_next = c0 * (crys_fam_t - c1 * pred_crys_fam) + sigmas * rand_crys_fam
            crys_fam_next = self.crystal_family.proj_k_to_spacegroup(crys_fam_next, batch.spacegroup)
            crys_fam_next = torch.clamp(crys_fam_next, -8.0, 8.0)

            site_x_next = site_x_half - step_size_pred * pred_x + std_x_pred * rand_x
            site_x_next = ((site_P @ site_x_next.unsqueeze(-1)).squeeze(-1) + site_x0) % 1.0

            site_x_t = site_x_next
            crys_fam_t = crys_fam_next

        # Final reconstruction of all N atoms in the conventional cell
        l_final = self.crystal_family.v2m(crys_fam_t)
        full_x_final = (
            batch.ops[:, :3, :3] @ site_x_t[inverse_site_map].unsqueeze(-1)
        ).squeeze(-1) + batch.ops[:, :3, 3]
        full_x_final = full_x_final % 1.0

        out = {
            "num_atoms": batch.num_atoms,
            "atom_types": batch.atom_types,
            "frac_coords": full_x_final,
            "lattices": l_final,
            "crys_fam": crys_fam_t,
        }
        return out, {}

    def load_vanilla_checkpoint(self, ckpt_path: str) -> None:
        """Loads weights from a vanilla CSPDiffusion / CSPNet checkpoint."""
        raw = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if isinstance(raw, dict) and "model_state_dict" in raw:
            raw = raw["model_state_dict"]
        sd = {}
        for k, v in raw.items():
            k = re.sub(r"^_orig_mod\.", "", k)
            k = re.sub(r"^decoder\.csp_layer_(\d+)\.", lambda m: f"decoder.csp_layers.{m.group(1)}.", k)
            sd[k] = v
        ne = "decoder.node_embedding.weight"
        if ne in sd and sd[ne].shape[0] == 100:
            w = torch.zeros(101, sd[ne].shape[1], dtype=sd[ne].dtype, device=self.device)
            w[1:101] = sd[ne]
            sd[ne] = w
        res = self.load_state_dict(sd, strict=False)
        missing = [
            k for k in res.missing_keys
            if not any(s in k for s in ("scheduler", "time_embedding", "crystal_family", "dis_emb"))
        ]
        if missing:
            raise RuntimeError(f"Missing required learned weights: {missing}")

    def training_step(self, batch: Any, batch_idx: int = 0) -> torch.Tensor | None:
        """Convenience training step wrapper."""
        outputs = self(batch)
        loss = outputs.get("loss")
        if loss is None or torch.isnan(loss):
            return None
        return loss
