"""GeoOrbDiffusion: Deep MLIP-Conditioned Crystal Diffusion with Two-Phase Hybrid Sampling.

DiffCSP-GeoOrb diffusion integrates:
1. Frozen ORB MLIP Backbone (orb-v3) with ~25.6M frozen parameters and <= 12.28M trainable parameters.
2. EMA (Exponential Moving Average) support with decay 0.9999 during training.
3. Subspace tangent noise injection and free-DoF loss masking.
4. Exact locking of 0-DoF Wyckoff special positions to prevent numerical drift.
5. Two-Phase Hybrid Sampling with `--orb_handoff_t <float>`: hands off intermediate
   structures at t <= orb_handoff_t to symmetry-constrained ORB FIRE relaxation
   (FixSymmetry + FrechetCellFilter).
"""

from contextlib import contextmanager
import logging
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
from diffcsp.models.geo_orb_cspnet import GeoOrbCSPNet
from diffcsp.models.geo_v2_diffusion import EMAModel

logger = logging.getLogger(__name__)


class GeoOrbDiffusion(CSPDiffusion):
    """Deep MLIP-Conditioned Diffusion model with EMA and Two-Phase Hybrid Sampling."""

    def __init__(
        self,
        device: str | torch.device = "cpu",
        decoder: nn.Module | None = None,
        orb_model_name: str = "orb-v3",
        use_mock_orb: bool = False,
        hidden_dim: int = 512,
        num_layers: int = 6,
        edge_dim: int | None = None,
        time_dim: int = 256,
        timesteps: int = 1000,
        beta_scheduler_mode: str = "cosine",
        sigma_begin: float = 0.005,
        sigma_end: float = 0.5,
        ema_decay: float = 0.9999,
        orb_handoff_t: float | None = None,
        relax_fmax: float = 0.05,
        relax_steps: int = 100,
    ) -> None:
        device_obj = torch.device(device)
        if decoder is None:
            decoder = GeoOrbCSPNet(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                edge_dim=edge_dim,
                latent_dim=time_dim,
                orb_model_name=orb_model_name,
                use_mock_orb=use_mock_orb,
                device=device_obj,
            )
        super().__init__(
            device=device_obj,
            decoder=decoder,
            time_dim=time_dim,
            timesteps=timesteps,
            beta_scheduler_mode=beta_scheduler_mode,
            sigma_begin=sigma_begin,
            sigma_end=sigma_end,
        )
        self.orb_backbone = decoder.orb_backbone
        self.ema_decay = ema_decay
        self.ema: EMAModel | None = None
        self.orb_handoff_t = orb_handoff_t
        self.relax_fmax = relax_fmax
        self.relax_steps = relax_steps

    def init_ema(self, decay: float | None = None) -> None:
        """Initializes EMA shadow parameters for trainable weights."""
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

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Returns only the trainable model parameters (excluding frozen ORB backbone)."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> dict[str, Any]:
        """Returns parameter counts distinguishing trainable adapter vs frozen backbone."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total,
            "trainable_pct": (trainable / total * 100.0) if total > 0 else 0.0,
        }

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

        if hasattr(batch, "dofs") and batch.dofs is not None:
            dofs = batch.dofs
        else:
            dofs = torch.round(torch.diagonal(Pa, dim1=-2, dim2=-1).sum(-1)).long().clamp(0, 3)

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
        free_mask = dofs > 0

        # Tangent-space noise injection:
        # eps_anchor ~ N(0, I_3), projected by Pa; locked to 0 if anchor DoF == 0.
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

    def relax_structures_with_orb(
        self,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        atom_types: torch.Tensor,
        num_atoms: torch.Tensor,
        fmax: float = 0.05,
        steps: int = 100,
        fix_symmetry: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Relaxes crystal structures with symmetry-constrained ORB FIRE (FixSymmetry + FrechetCellFilter).

        Args:
            frac_coords: (N, 3) fractional atomic coordinates.
            lattices: (B, 3, 3) Cartesian lattice matrices.
            atom_types: (N,) atomic numbers.
            num_atoms: (B,) atom counts per structure.
            fmax: Maximum force tolerance for FIRE optimizer.
            steps: Maximum FIRE optimization steps.
            fix_symmetry: If True, constrains cell and site degrees of freedom to space group.

        Returns:
            Tuple of:
                relaxed_frac_coords: (N, 3) relaxed fractional coordinates.
                relaxed_lattices: (B, 3, 3) relaxed lattice matrices.
        """
        import numpy as np
        from ase import Atoms
        from ase.filters import FrechetCellFilter
        from ase.optimize import FIRE

        B = lattices.shape[0]
        device = frac_coords.device
        dtype = frac_coords.dtype

        counts = num_atoms.detach().cpu().numpy()
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        atom_types_np = atom_types.detach().cpu().numpy()
        frac_coords_np = frac_coords.detach().cpu().numpy()
        lattices_np = lattices.detach().cpu().numpy()

        relaxed_coords_list = []
        relaxed_lattices_list = []

        # Build calculator
        calc = None
        if hasattr(self.orb_backbone, "orb_model") and not self.orb_backbone.use_mock:
            try:
                from orb_models.forcefield.inference.calculator import ORBCalculator

                calc = ORBCalculator(
                    self.orb_backbone.orb_model,
                    self.orb_backbone.atoms_adapter,
                    device=self.device,
                )
            except Exception as exc:
                logger.warning("Could not initialize ORBCalculator: %s", exc)
                calc = None

        for b in range(B):
            cell_b = lattices_np[b]
            types_b = atom_types_np[offsets[b] : offsets[b + 1]]
            coords_b = frac_coords_np[offsets[b] : offsets[b + 1]] % 1.0

            try:
                atoms = Atoms(numbers=types_b, scaled_positions=coords_b, cell=cell_b, pbc=True)
                if calc is not None:
                    atoms.calc = calc
                else:
                    from ase.calculators.calculator import Calculator, all_changes

                    class _MockAseCalculator(Calculator):
                        implemented_properties = ["energy", "forces", "stress"]

                        def calculate(
                            self, atoms=None, properties=["energy"], system_changes=all_changes
                        ):
                            super().calculate(atoms, properties, system_changes)
                            self.results = {
                                "energy": 0.0,
                                "forces": np.zeros((len(self.atoms), 3)),
                                "stress": np.zeros(6),
                            }

                    atoms.calc = _MockAseCalculator()

                if fix_symmetry:
                    try:
                        from ase.constraints import FixSymmetry

                        atoms.set_constraint(FixSymmetry(atoms))
                    except Exception:
                        pass

                opt = FIRE(FrechetCellFilter(atoms), logfile=None)
                opt.run(fmax=fmax, steps=steps)

                r_cell = atoms.get_cell()[:]
                r_coords = atoms.get_scaled_positions() % 1.0
                relaxed_lattices_list.append(torch.as_tensor(r_cell, dtype=dtype, device=device))
                relaxed_coords_list.append(torch.as_tensor(r_coords, dtype=dtype, device=device))
            except Exception as err:
                logger.warning("Relaxation failed for crystal %d: %s. Using intermediate.", b, err)
                relaxed_lattices_list.append(lattices[b])
                relaxed_coords_list.append(frac_coords[offsets[b] : offsets[b + 1]])

        relaxed_frac = torch.cat(relaxed_coords_list, dim=0)
        relaxed_lat = torch.stack(relaxed_lattices_list, dim=0)
        return relaxed_frac, relaxed_lat

    @torch.no_grad()
    def sample(
        self,
        batch: Any,
        step_lr: float = 5e-6,
        disable_progress: bool = False,
        orbit_average: bool = True,
        anneal_corrector: bool = True,
        use_ema: bool = False,
        orb_handoff_t: float | None = None,
        relax_fmax: float | None = None,
        relax_steps: int | None = None,
        fix_symmetry: bool = True,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Two-phase hybrid sampling with predictor-corrector diffusion and symmetry-constrained ORB handoff.

        Args:
            batch: PyTorch Geometric Batch instance.
            step_lr: Base Langevin corrector step scale.
            disable_progress: Suppresses per-step progress bar.
            orbit_average: If True, averages coordinate scores over Wyckoff orbits.
            anneal_corrector: If True, anneals corrector step size smoothly as t -> 0.
            use_ema: If True, executes under EMA shadow weights.
            orb_handoff_t: Optional normalized handoff threshold in (0, 1]. At t <= orb_handoff_t,
                           hands off intermediate structure to symmetry-constrained ORB FIRE relaxation.
            relax_fmax: Target maximum force tolerance for FIRE relaxation.
            relax_steps: Maximum FIRE relaxation steps.
            fix_symmetry: If True, applies FixSymmetry constraint during ORB relaxation.
        """
        handoff_t = orb_handoff_t if orb_handoff_t is not None else self.orb_handoff_t
        fmax = relax_fmax if relax_fmax is not None else self.relax_fmax
        steps = relax_steps if relax_steps is not None else self.relax_steps

        if use_ema and self.ema is not None:
            with self.ema_scope():
                return self._sample_impl(
                    batch=batch,
                    step_lr=step_lr,
                    disable_progress=disable_progress,
                    orbit_average=orbit_average,
                    anneal_corrector=anneal_corrector,
                    orb_handoff_t=handoff_t,
                    relax_fmax=fmax,
                    relax_steps=steps,
                    fix_symmetry=fix_symmetry,
                )
        return self._sample_impl(
            batch=batch,
            step_lr=step_lr,
            disable_progress=disable_progress,
            orbit_average=orbit_average,
            anneal_corrector=anneal_corrector,
            orb_handoff_t=handoff_t,
            relax_fmax=fmax,
            relax_steps=steps,
            fix_symmetry=fix_symmetry,
        )

    @torch.no_grad()
    def _sample_impl(
        self,
        batch: Any,
        step_lr: float = 5e-6,
        disable_progress: bool = False,
        orbit_average: bool = True,
        anneal_corrector: bool = True,
        orb_handoff_t: float | None = None,
        relax_fmax: float = 0.05,
        relax_steps: int = 100,
        fix_symmetry: bool = True,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch_size = batch.batch_size if hasattr(batch, "batch_size") else batch.num_graphs
        Pa, Pj, dofs, multiplicities, spacegroups = self.derive_wyckoff_symmetries(batch)
        free_mask = dofs > 0
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
            pbar = tqdm(pbar, desc="GeoOrbDiffusion Sampling", leave=False)

        handed_off = False
        final_t = 0

        for t in pbar:
            # Check for ORB handoff threshold
            t_norm = float(t) / float(time_start)
            if orb_handoff_t is not None and t_norm <= orb_handoff_t:
                cur_x = traj[t]["frac_coords"]
                cur_lat = traj[t]["lattices"]
                rel_x, rel_lat = self.relax_structures_with_orb(
                    frac_coords=cur_x,
                    lattices=cur_lat,
                    atom_types=batch.atom_types,
                    num_atoms=batch.num_atoms,
                    fmax=relax_fmax,
                    steps=relax_steps,
                    fix_symmetry=fix_symmetry,
                )
                if init_coords is not None and is_zero_dof.any():
                    rel_x[is_zero_dof] = init_coords[is_zero_dof]
                rel_crys_fam = self.crystal_family.m2v(rel_lat)
                rel_crys_fam = self.crystal_family.proj_k_to_spacegroup(
                    rel_crys_fam, batch.spacegroup
                )
                rel_lat = self.crystal_family.v2m(rel_crys_fam)
                traj[0] = {
                    "num_atoms": batch.num_atoms,
                    "atom_types": batch.atom_types,
                    "frac_coords": rel_x % 1.0,
                    "lattices": rel_lat,
                    "crys_fam": rel_crys_fam,
                }
                handed_off = True
                final_t = 0
                break

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
            rand_x_raw = torch.randn_like(cur_x) if t > 1 else torch.zeros_like(cur_x)

            if anneal_corrector:
                effective_step_lr = step_lr * (float(t) / float(time_start))
            else:
                effective_step_lr = step_lr

            step_size = effective_step_lr / (
                sigma_norm_val * (self.sigma_scheduler.sigma_begin) ** 2
            )
            std_x = torch.sqrt(2.0 * step_size)

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
            rand_crys_fam = torch.randn_like(cur_crys_fam)
            rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(
                rand_crys_fam, batch.spacegroup
            )
            rand_x_raw = torch.randn_like(cur_x) if t > 1 else torch.zeros_like(cur_x)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size_pred = sigma_x**2 - adjacent_sigma_x**2
            std_x_pred = torch.sqrt(
                torch.clamp(
                    (adjacent_sigma_x**2 * (sigma_x**2 - adjacent_sigma_x**2)) / (sigma_x**2),
                    min=0.0,
                )
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

        traj_steps = sorted(traj.keys(), reverse=True)
        traj_stack = {
            "num_atoms": batch.num_atoms,
            "atom_types": batch.atom_types,
            "all_frac_coords": torch.stack([traj[i]["frac_coords"] for i in traj_steps]),
            "all_lattices": torch.stack([traj[i]["lattices"] for i in traj_steps]),
        }
        return traj[0], traj_stack
