"""GeoV2Diffusion: Enhanced symmetry-projected full-cell diffusion for crystal structures.

DiffCSP-GeoV2 diffusion integrates:
1. EMA (Exponential Moving Average) support with decay 0.9999 during training.
2. Annealed predictor-corrector sampler (base step_lr=5e-6 decaying smoothly as t -> 0).
3. Free-DoF loss masking and Lie-algebra Wyckoff tangent space noise injection.
4. Exact locking of 0-DoF coordinates to prevent numerical drift.
"""

from contextlib import contextmanager
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
from diffcsp.models.diffusion import CSPDiffusion
from diffcsp.models.geo_v2_cspnet import GeoV2CSPNet


class EMAModel:
    """Exponential Moving Average (EMA) manager for PyTorch model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.collected_params: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        """Updates shadow parameters with current model parameters."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data.to(self.shadow[name].device), alpha=1.0 - self.decay
                    )

    def apply_shadow(self, model: nn.Module) -> None:
        """Replaces model parameters with shadow parameters, caching originals."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.collected_params[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restores model parameters from cached originals."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.collected_params:
                param.data.copy_(self.collected_params[name])
        self.collected_params.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Returns shadow parameters dictionary on CPU."""
        return {k: v.clone().cpu() for k, v in self.shadow.items()}

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], device: torch.device | None = None
    ) -> None:
        """Loads shadow parameters from dictionary."""
        self.shadow = {}
        for k, v in state_dict.items():
            t = v.clone()
            if device is not None:
                t = t.to(device)
            self.shadow[k] = t


class GeoV2Diffusion(CSPDiffusion):
    """Full-cell diffusion model with EMA, annealed Langevin corrector, and tangent projections."""

    def __init__(
        self,
        device: str | torch.device = "cpu",
        decoder: nn.Module | None = None,
        time_dim: int = 256,
        timesteps: int = 1000,
        beta_scheduler_mode: str = "cosine",
        sigma_begin: float = 0.005,
        sigma_end: float = 0.5,
        ema_decay: float = 0.9999,
    ) -> None:
        decoder = decoder if decoder is not None else GeoV2CSPNet()
        super().__init__(
            device=device,
            decoder=decoder,
            time_dim=time_dim,
            timesteps=timesteps,
            beta_scheduler_mode=beta_scheduler_mode,
            sigma_begin=sigma_begin,
            sigma_end=sigma_end,
        )
        self.ema_decay = ema_decay
        self.ema: EMAModel | None = None

    def init_ema(self, decay: float | None = None) -> None:
        """Initializes EMA shadow parameters."""
        d = decay if decay is not None else self.ema_decay
        self.ema = EMAModel(self.decoder, decay=d)

    def update_ema(self) -> None:
        """Updates EMA shadow parameters with current decoder weights."""
        if self.ema is not None:
            self.ema.update(self.decoder)

    def apply_ema(self) -> None:
        """Swaps decoder weights to EMA shadow weights."""
        if self.ema is not None:
            self.ema.apply_shadow(self.decoder)

    def restore_ema(self) -> None:
        """Restores decoder weights from before apply_ema."""
        if self.ema is not None:
            self.ema.restore(self.decoder)

    @contextmanager
    def ema_scope(self):
        """Context manager to execute under EMA parameters."""
        if self.ema is not None:
            self.apply_ema()
            try:
                yield
            finally:
                self.restore_ema()
        else:
            yield

    def ema_state_dict(self) -> dict[str, torch.Tensor] | None:
        """Returns EMA state dict if initialized."""
        return self.ema.state_dict() if self.ema is not None else None

    def load_ema_state_dict(
        self, state_dict: dict[str, torch.Tensor], device: torch.device | None = None
    ) -> None:
        """Loads EMA state dict into shadow parameters."""
        if self.ema is None:
            self.init_ema()
        dev = device if device is not None else torch.device(self.device)
        self.ema.load_state_dict(state_dict, device=dev)

    def derive_wyckoff_symmetries(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Derives anchor projectors, atom projectors, degrees of freedom, and multiplicities.

        Returns:
            Pa: (N, 3, 3) anchor tangent space projector for each atom.
            Pj: (N, 3, 3) atom tangent space projector for each atom.
            dofs: (N,) integer degrees of freedom for each atom (0..3).
            multiplicities: (N,) multiplicity of Wyckoff orbit for each atom.
            spacegroup: (B,) space group index for each crystal.
        """
        Pa = batch.ops[batch.anchor_index, :3, :3]  # (N, 3, 3)
        R = batch.ops[:, :3, :3]  # (N, 3, 3)
        R_inv = batch.ops_inv  # (N, 3, 3)

        # Atom projector: P_j = R_j @ P_a @ R_j^{-1}
        Pj = R @ Pa @ R_inv

        # Degrees of freedom: trace(P_a) in {0, 1, 2, 3}
        if hasattr(batch, "dofs") and batch.dofs is not None:
            dofs = batch.dofs
        else:
            dofs = torch.round(torch.diagonal(Pa, dim1=-2, dim2=-1).sum(-1)).long().clamp(0, 3)

        # Multiplicity via bincount on anchor_index
        if hasattr(batch, "multiplicities") and batch.multiplicities is not None:
            multiplicities = batch.multiplicities
        elif hasattr(batch, "multiplicity") and batch.multiplicity is not None:
            multiplicities = batch.multiplicity
        else:
            mult_per_anchor = torch.bincount(batch.anchor_index)
            multiplicities = mult_per_anchor[batch.anchor_index].clamp(0, 192)

        spacegroup = batch.spacegroup
        return Pa, Pj, dofs, multiplicities, spacegroup

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        """Calculates diffusion training loss with free-DoF masking and tangent space noise injection.

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

        Pa, Pj, dofs, multiplicities, spacegroups = self.derive_wyckoff_symmetries(batch)
        free_mask = (dofs > 0)

        # Tangent-space noise injection:
        # eps_anchor ~ N(0, I_3), projected by Pa; locked to 0 if anchor DoF == 0.
        # Atom j noise is R_j @ eps_anchor.
        rand_x_raw = torch.randn_like(frac_coords)
        rand_x_anchor = rand_x_raw[batch.anchor_index]
        rand_x_anchor = (Pa @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x_anchor = torch.where(
            free_mask.unsqueeze(-1), rand_x_anchor, torch.zeros_like(rand_x_anchor)
        )

        R = batch.ops[:, :3, :3]
        rand_x = (R @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]

        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.0
        # Exactly lock 0-DoF special positions to prevent numerical drift
        if (~free_mask).any():
            input_frac_coords[~free_mask] = frac_coords[~free_mask]

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
            spacegroups=spacegroups,
            multiplicities=multiplicities,
            dofs=dofs,
            site_projectors=Pj,
        )
        pred_crys_fam = self.crystal_family.proj_k_to_spacegroup(pred_crys_fam, batch.spacegroup)

        # Coordinate target in anchor tangent space
        pred_x_proj = torch.einsum("bij, bj -> bi", batch.ops_inv, pred_x)
        raw_tar_anchor = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x_anchor, sigmas_per_atom
        ) / torch.sqrt(sigmas_norm_per_atom)
        tar_x_anchor = (Pa @ raw_tar_anchor.unsqueeze(-1)).squeeze(-1)

        loss_lattice = F.mse_loss(pred_crys_fam, rand_crys_fam)
        # Free-DoF loss masking
        if free_mask.any():
            loss_coord = F.mse_loss(pred_x_proj[free_mask], tar_x_anchor[free_mask])
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
        self,
        batch: Any,
        step_lr: float = 5e-6,
        disable_progress: bool = False,
        orbit_average: bool = True,
        anneal_corrector: bool = True,
        noise_cutoff_t: int = 0,
        use_ema: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Annealed Predictor-Corrector sampling preserving exact 0-DoF sites and tangent projections.

        Args:
            batch: PyTorch Geometric Data/Batch instance.
            step_lr: Base Langevin corrector step scale (default: 5e-6).
            disable_progress: Suppresses per-step progress bar.
            orbit_average: If True, averages score over all atoms of each Wyckoff orbit.
            anneal_corrector: If True, decays corrector step size smoothly as t -> 0.
            noise_cutoff_t: Timestep below which Langevin and predictor noise is disabled.
            use_ema: If True and EMA shadow model is available, uses EMA weights for sampling.
        """
        if use_ema and self.ema is not None:
            with self.ema_scope():
                return self._sample_impl(
                    batch=batch,
                    step_lr=step_lr,
                    disable_progress=disable_progress,
                    orbit_average=orbit_average,
                    anneal_corrector=anneal_corrector,
                    noise_cutoff_t=noise_cutoff_t,
                )
        return self._sample_impl(
            batch=batch,
            step_lr=step_lr,
            disable_progress=disable_progress,
            orbit_average=orbit_average,
            anneal_corrector=anneal_corrector,
            noise_cutoff_t=noise_cutoff_t,
        )

    @torch.no_grad()
    def _sample_impl(
        self,
        batch: Any,
        step_lr: float = 5e-6,
        disable_progress: bool = False,
        orbit_average: bool = True,
        anneal_corrector: bool = True,
        noise_cutoff_t: int = 0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch_size = batch.batch_size if hasattr(batch, "batch_size") else batch.num_graphs
        Pa, Pj, dofs, multiplicities, spacegroups = self.derive_wyckoff_symmetries(batch)
        free_mask = (dofs > 0)
        is_zero_dof = ~free_mask

        if hasattr(batch, "frac_coords"):
            init_coords = batch.frac_coords % 1.0
            init_coords = torch.where(
                init_coords >= 1.0 - 1e-6, torch.zeros_like(init_coords), init_coords
            )
            init_coords = torch.where(
                init_coords.abs() < 1e-6, torch.zeros_like(init_coords), init_coords
            )
        else:
            init_coords = None
        anchor_x0 = batch.ops[batch.anchor_index, :3, 3]
        R = batch.ops[:, :3, :3]

        # Initialize coordinates: random for free DoFs, exact for 0-DoF
        x_rand = torch.rand([batch.num_nodes, 3], device=self.device)
        x_anchor = x_rand[batch.anchor_index]
        x_anchor = ((Pa @ x_anchor.unsqueeze(-1)).squeeze(-1) + anchor_x0) % 1.0
        x_t = ((R @ x_anchor.unsqueeze(-1)).squeeze(-1) + batch.ops[:, :3, 3]) % 1.0
        x_t = torch.where(x_t >= 1.0 - 1e-6, torch.zeros_like(x_t), x_t)

        if init_coords is not None and is_zero_dof.any():
            x_t[is_zero_dof] = init_coords[is_zero_dof]

        crys_fam_t = torch.randn([batch_size, 6], device=self.device)
        crys_fam_t = self.crystal_family.proj_k_to_spacegroup(crys_fam_t, batch.spacegroup)

        time_start = self.beta_scheduler.timesteps - 1
        l_t = self.crystal_family.v2m(crys_fam_t)

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
            pbar = tqdm(pbar, desc="GeoV2Diffusion Sampling", leave=False)

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

            # --- Corrector Step (Annealed Langevin Corrector) ---
            rand_x_raw = (
                torch.randn_like(cur_x)
                if (t > 1 and t > noise_cutoff_t)
                else torch.zeros_like(cur_x)
            )

            # Smoothly anneal step_lr as t -> 0
            if anneal_corrector:
                effective_step_lr = step_lr * (float(t) / float(time_start))
            else:
                effective_step_lr = step_lr

            step_size = effective_step_lr / (
                sigma_norm_val * (self.sigma_scheduler.sigma_begin) ** 2
            )
            std_x = torch.sqrt(2.0 * step_size) if t > noise_cutoff_t else 0.0

            rand_x_anchor = (Pa @ rand_x_raw[batch.anchor_index].unsqueeze(-1)).squeeze(-1)
            rand_x_anchor = torch.where(
                free_mask.unsqueeze(-1), rand_x_anchor, torch.zeros_like(rand_x_anchor)
            )
            rand_x = (R @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            _, pred_x = self.decoder(
                time_emb,
                batch.atom_types,
                cur_x,
                cur_crys_fam,
                batch.num_atoms,
                batch.batch,
                spacegroups=spacegroups,
                multiplicities=multiplicities,
                dofs=dofs,
                site_projectors=Pj,
            )
            pred_x = pred_x * torch.sqrt(sigma_norm_val)

            pred_x_proj = torch.einsum("bij, bj -> bi", batch.ops_inv, pred_x)
            if orbit_average:
                pred_x_anchor = scatter(
                    pred_x_proj, batch.anchor_index, dim=0, reduce="mean"
                )[batch.anchor_index]
            else:
                pred_x_anchor = pred_x_proj[batch.anchor_index]

            pred_x_anchor = (Pa @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)
            pred_x_anchor = torch.where(
                free_mask.unsqueeze(-1), pred_x_anchor, torch.zeros_like(pred_x_anchor)
            )
            pred_x = (R @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

            x_half = cur_x - step_size * pred_x + std_x * rand_x
            x_anchor = x_half[batch.anchor_index]
            x_anchor = ((Pa @ x_anchor.unsqueeze(-1)).squeeze(-1) + anchor_x0) % 1.0
            x_half = ((R @ x_anchor.unsqueeze(-1)).squeeze(-1) + batch.ops[:, :3, 3]) % 1.0
            x_half = torch.where(x_half >= 1.0 - 1e-6, torch.zeros_like(x_half), x_half)
            if init_coords is not None and is_zero_dof.any():
                x_half[is_zero_dof] = init_coords[is_zero_dof]

            # --- Predictor Step ---
            rand_crys_fam = (
                torch.randn_like(cur_crys_fam) if t > noise_cutoff_t else torch.zeros_like(cur_crys_fam)
            )
            rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(
                rand_crys_fam, batch.spacegroup
            )
            rand_x_raw = (
                torch.randn_like(cur_x)
                if (t > 1 and t > noise_cutoff_t)
                else torch.zeros_like(cur_x)
            )

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size_pred = sigma_x**2 - adjacent_sigma_x**2
            std_x_pred = (
                torch.sqrt(
                    torch.clamp(
                        (adjacent_sigma_x**2 * (sigma_x**2 - adjacent_sigma_x**2)) / (sigma_x**2),
                        min=0.0,
                    )
                )
                if t > noise_cutoff_t
                else 0.0
            )

            rand_x_anchor = (Pa @ rand_x_raw[batch.anchor_index].unsqueeze(-1)).squeeze(-1)
            rand_x_anchor = torch.where(
                free_mask.unsqueeze(-1), rand_x_anchor, torch.zeros_like(rand_x_anchor)
            )
            rand_x = (R @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

            pred_crys_fam, pred_x = self.decoder(
                time_emb,
                batch.atom_types,
                x_half,
                cur_crys_fam,
                batch.num_atoms,
                batch.batch,
                spacegroups=spacegroups,
                multiplicities=multiplicities,
                dofs=dofs,
                site_projectors=Pj,
            )
            pred_crys_fam = self.crystal_family.proj_k_to_spacegroup(
                pred_crys_fam, batch.spacegroup
            )
            pred_x = pred_x * torch.sqrt(sigma_norm_val)

            crys_fam_next = c0 * (cur_crys_fam - c1 * pred_crys_fam) + sigmas * rand_crys_fam
            crys_fam_next = torch.clamp(crys_fam_next, min=-15.0, max=15.0)
            crys_fam_next = self.crystal_family.proj_k_to_spacegroup(
                crys_fam_next, batch.spacegroup
            )

            pred_x_proj = torch.einsum("bij, bj -> bi", batch.ops_inv, pred_x)
            if orbit_average:
                pred_x_anchor = scatter(
                    pred_x_proj, batch.anchor_index, dim=0, reduce="mean"
                )[batch.anchor_index]
            else:
                pred_x_anchor = pred_x_proj[batch.anchor_index]

            pred_x_anchor = (Pa @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)
            pred_x_anchor = torch.where(
                free_mask.unsqueeze(-1), pred_x_anchor, torch.zeros_like(pred_x_anchor)
            )
            pred_x = (R @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

            x_next = x_half - step_size_pred * pred_x + std_x_pred * rand_x
            x_anchor = x_next[batch.anchor_index]
            x_anchor = ((Pa @ x_anchor.unsqueeze(-1)).squeeze(-1) + anchor_x0) % 1.0
            x_next = ((R @ x_anchor.unsqueeze(-1)).squeeze(-1) + batch.ops[:, :3, 3]) % 1.0
            x_next = torch.where(x_next >= 1.0 - 1e-6, torch.zeros_like(x_next), x_next)
            if init_coords is not None and is_zero_dof.any():
                x_next[is_zero_dof] = init_coords[is_zero_dof]

            l_next = self.crystal_family.v2m(crys_fam_next)

            traj[t - 1] = {
                "num_atoms": batch.num_atoms,
                "atom_types": batch.atom_types,
                "frac_coords": x_next,
                "lattices": l_next,
                "crys_fam": crys_fam_next,
            }

        traj_stack = {
            "num_atoms": batch.num_atoms,
            "atom_types": batch.atom_types,
            "all_frac_coords": torch.stack(
                [traj[i]["frac_coords"] for i in range(time_start, -1, -1)]
            ),
            "all_lattices": torch.stack(
                [traj[i]["lattices"] for i in range(time_start, -1, -1)]
            ),
        }
        return traj[0], traj_stack
