"""Unified training CLI for DiffCSP++ (standard GNN and ORB MLIP adapter)."""

import argparse
import logging
import os
import random
import math
import signal
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader, PrefetchLoader
from tqdm import tqdm, trange

from diffcsp.data.dataset import CrystDataset
from diffcsp.data.packed import EdgeBudgetBatchSampler, PackedCrystDataset
from diffcsp.models.cspnet import CSPNet
from diffcsp.models.diffusion import CSPDiffusion
from diffcsp.models.diffusion_orb import CSPDiffusionORB
from diffcsp.models.geo_cspnet import GeoCSPNet
from diffcsp.models.geo_diffusion import GeoDiffusion
from diffcsp.models.geo_v2_cspnet import GeoV2CSPNet
from diffcsp.models.geo_v2_diffusion import GeoV2Diffusion
from diffcsp.models.geo_orb_cspnet import GeoOrbCSPNet
from diffcsp.models.geo_orb_diffusion import GeoOrbDiffusion
from diffcsp.models.wyckoff_cspnet import WyckoffCSPNet
from diffcsp.models.wyckoff_diffusion import WyckoffDiffusion
from diffcsp.models.wyckoff_painn import WyckoffPaiNN

logger = logging.getLogger(__name__)


def _resolve_wandb_checkpoint(uri: str) -> Path:
    """Downloads a checkpoint from W&B given a ``wandb://`` URI and returns the local path.

    Supported URI formats::

        wandb://entity/project/artifact_name:alias
        wandb://project/artifact_name:alias
        wandb://artifact_name:alias

    The alias defaults to ``latest`` if omitted.
    """
    import wandb

    ref = uri[len("wandb://"):]
    parts = ref.split("/")
    if len(parts) == 3:
        entity, project, artifact_ref = parts
    elif len(parts) == 2:
        entity = None
        project, artifact_ref = parts
    elif len(parts) == 1:
        entity = None
        project = None
        artifact_ref = parts[0]
    else:
        raise ValueError(
            f"Invalid wandb URI: '{uri}'. "
            "Expected wandb://[entity/][project/]artifact_name[:alias]"
        )

    if ":" not in artifact_ref:
        artifact_ref += ":latest"

    full_name_parts = [p for p in [entity, project, artifact_ref] if p]
    full_name = "/".join(full_name_parts)

    api = wandb.Api()
    print(f"Downloading W&B artifact '{full_name}' ...")
    artifact = api.artifact(full_name, type="model")
    artifact_dir = Path(artifact.download())

    pt_files = list(artifact_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in downloaded artifact at {artifact_dir}")
    resolved = pt_files[0]
    print(f"Resolved checkpoint to {resolved}")
    return resolved


ORB_BACKBONE_PREFIX = "orb_backbone."


def _adapter_state_dict(decoder: torch.nn.Module) -> dict[str, Any]:
    """Returns the decoder's trainable weights, excluding the frozen MLIP backbone.

    The backbone is pretrained and frozen, so storing it only bloats the
    checkpoint and ties it to one ORB variant. Leaving it out keeps a checkpoint
    loadable after switching backbone (e.g. orb-v2 to orb-v3).
    """
    return {k: v for k, v in decoder.state_dict().items() if not k.startswith(ORB_BACKBONE_PREFIX)}


def _load_adapter_state_dict(decoder: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    """Loads adapter weights into the decoder, tolerating an embedded backbone.

    Checkpoints written before the backbone was excluded carry its weights under
    ``orb_backbone.*``; those keys are dropped so that a checkpoint trained
    against one ORB variant still restores the adapter when another is loaded.
    """
    filtered = {k: v for k, v in state_dict.items() if not k.startswith(ORB_BACKBONE_PREFIX)}
    dropped = len(state_dict) - len(filtered)

    result = decoder.load_state_dict(filtered, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(ORB_BACKBONE_PREFIX)]

    print(f"Restored {len(filtered)} adapter tensors" + (f" (dropped {dropped} frozen backbone tensors)" if dropped else ""))
    if missing:
        print(f"[WARN] {len(missing)} adapter tensors absent from checkpoint and left at init: {missing[:8]}")
    if result.unexpected_keys:
        print(f"[WARN] {len(result.unexpected_keys)} unexpected tensors ignored: {result.unexpected_keys[:8]}")


def _worker_init_fn(worker_id: int) -> None:
    """Sets worker process CPU threads to 1 to avoid thread oversubscription."""
    torch.set_num_threads(1)


def set_random_seed(seed: int = 17) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(
    train_csv: str = "data/mp-20/train.csv",
    test_csv: str | None = "data/mp-20/test.csv",
    model_type: str = "orb",
    orb_model: str = "orb-v3",
    mock_orb: bool = False,
    use_orb_node_features: bool = True,
    enforce_zero_force: bool = False,
    force_residual: bool = False,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 100,
    eval_freq: int = 10,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ckpt_path: str = "diffcsp_ckpt.pt",
    resume: str | None = None,
    save_freq: int = 1,
    use_wandb: bool = True,
    wandb_project: str = "diffcsp",
    wandb_entity: str | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    cache_dir: str | None = None,
    max_e_hull: float | None = None,
    max_atoms: int | None = None,
    eval_sample: bool = False,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    async_dataloader: bool = True,
    time_limit_hours: float | None = None,
    data_dir: str | None = None,
    max_edges_per_batch: int | None = None,
    cond_props: list[str] | None = None,
    cond_drop_prob: float = 0.0,
    warmup_steps: int | None = None,
    extra_val_max_e_hull: float | None = None,
    wandb_name: str | None = None,
    wandb_tags: list[str] | None = None,
) -> None:
    """Executes model training with periodic evaluation.

    With ``data_dir`` (a packed dataset from ``scripts/pack_dataset.py``) the train and
    validation splits are ``<data_dir>/train`` and ``<data_dir>/val``, batches are formed
    by ``max_edges_per_batch`` (sum of N^2 per rank) instead of ``batch_size``, and, with
    ``warmup_steps``, the learning rate follows a per-step linear warmup and cosine decay.
    """
    cond_props = list(cond_props or [])
    if cond_props and model_type != "geov2":
        raise ValueError("Property conditioning is implemented for --model geov2 only")
    if (cond_props or max_edges_per_batch) and not data_dir:
        raise ValueError("--cond_props and --max_edges_per_batch need a packed --data_dir")
    t_start = time.monotonic()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = world_size > 1

    if is_ddp:
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
        # Rank 0 validates, checkpoints and uploads while the others wait at a barrier;
        # the default 10 min collective timeout is too tight for that on a big split.
        dist.init_process_group(backend="nccl", device_id=dev, timeout=timedelta(hours=1))
    else:
        dev = torch.device(device)

    is_main = (not is_ddp) or (rank == 0)

    set_random_seed(17 + rank)

    if is_main:
        print("=" * 65)
        print(f"DiffCSP++ Training | Model: {model_type.upper()} | World Size: {world_size} | Device: {dev}")
        print("=" * 65)

    # Each backbone has its own natural size: the ORB adapter is a small head on a
    # frozen 25.6M potential, CSPNet is the whole denoiser. A single CLI default
    # would silently shrink one of them, so the flags override per-model defaults.
    ARCH_DEFAULTS = {
        "orb": (128, 2),
        "cspnet": (512, 6),
        "wyckoff": (512, 6),
        "asymm": (512, 6),
        "painn": (512, 6),
        "wyckoff_painn": (512, 6),
        "geo": (512, 6),
        "geov2": (512, 6),
        "geo_orb": (512, 6),
    }
    arch_h, arch_l = ARCH_DEFAULTS[model_type]
    if hidden_dim is not None:
        arch_h = hidden_dim
    if num_layers is not None:
        arch_l = num_layers

    if model_type == "orb":
        model = CSPDiffusionORB(
            device=dev,
            orb_model_name=orb_model,
            use_mock_orb=mock_orb,
            use_orb_node_features=use_orb_node_features,
            enforce_zero_force_condition=enforce_zero_force,
            use_force_residual=force_residual,
            hidden_dim=arch_h,
            num_layers=arch_l,
        ).to(dev)
        params_to_train = model.get_trainable_parameters()
        counts = model.count_parameters()
        if is_main:
            print(f"CSPNetORB adapter: hidden_dim={arch_h} num_layers={arch_l}")
            print(f"Trainable params: {counts['trainable']:,} ({counts['trainable_pct']:.2f}%)")
            print(f"Frozen params:    {counts['frozen']:,}")
    elif model_type in ("painn", "wyckoff_painn"):
        model = WyckoffDiffusion(
            device=dev, decoder=WyckoffPaiNN(hidden_dim=arch_h, num_layers=arch_l)
        ).to(dev)
        params_to_train = list(model.parameters())
        if is_main:
            print(f"WyckoffPaiNN: hidden_dim={arch_h} num_layers={arch_l}")
            print(f"Total params: {sum(p.numel() for p in params_to_train):,}")
    elif model_type in ("wyckoff", "asymm"):
        model = WyckoffDiffusion(
            device=dev, decoder=WyckoffCSPNet(hidden_dim=arch_h, num_layers=arch_l)
        ).to(dev)
        params_to_train = list(model.parameters())
        if is_main:
            print(f"WyckoffCSPNet: hidden_dim={arch_h} num_layers={arch_l}")
            print(f"Total params: {sum(p.numel() for p in params_to_train):,}")
    elif model_type == "geo":
        model = GeoDiffusion(
            device=dev, decoder=GeoCSPNet(hidden_dim=arch_h, num_layers=arch_l)
        ).to(dev)
        params_to_train = list(model.parameters())
        if is_main:
            print(f"GeoCSPNet: hidden_dim={arch_h} num_layers={arch_l}")
            print(f"Total params: {sum(p.numel() for p in params_to_train):,}")
    elif model_type == "geov2":
        model = GeoV2Diffusion(
            device=dev,
            decoder=GeoV2CSPNet(hidden_dim=arch_h, num_layers=arch_l, cond_props=cond_props),
            cond_drop_prob=cond_drop_prob,
        ).to(dev)
        model.init_ema(decay=0.9999)
        params_to_train = list(model.parameters())
        if is_main:
            print(f"GeoV2CSPNet: hidden_dim={arch_h} num_layers={arch_l}")
            if cond_props:
                print(f"Conditioned on {cond_props}, condition dropout {cond_drop_prob}")
            print(f"Total params: {sum(p.numel() for p in params_to_train):,}")
    elif model_type == "geo_orb":
        model = GeoOrbDiffusion(
            device=dev,
            decoder=GeoOrbCSPNet(hidden_dim=arch_h, num_layers=arch_l, device=dev),
        ).to(dev)
        model.init_ema(decay=0.9999)
        params_to_train = model.get_trainable_parameters()
        if is_main:
            counts = model.count_parameters()
            print(f"GeoOrbDiffusion: hidden_dim={arch_h} num_layers={arch_l}")
            print(f"Trainable params: {counts['trainable']:,} | Frozen ORB: {counts['frozen']:,}")
    else:
        model = CSPDiffusion(
            device=dev, decoder=CSPNet(hidden_dim=arch_h, num_layers=arch_l)
        ).to(dev)
        params_to_train = list(model.parameters())
        if is_main:
            print(f"CSPNet: hidden_dim={arch_h} num_layers={arch_l}")
            print(f"Total params: {sum(p.numel() for p in params_to_train):,}")

    if is_ddp:
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        ddp_model = model

    # A packed dataset is memory-mapped, so opening it is cheap and every rank can do
    # it before the scheduler, which needs the number of steps per epoch.
    packed_train = packed_val = packed_extra_val = train_batch_sampler = None
    if data_dir:
        packed_train = PackedCrystDataset(
            Path(data_dir) / "train", cond_props=cond_props, max_atoms=max_atoms,
            max_e_hull=max_e_hull, max_samples=max_train_samples,
        )
        train_batch_sampler = EdgeBudgetBatchSampler(
            packed_train.sizes(), max_edges=max_edges_per_batch or 64_000, shuffle=True,
            seed=17, rank=rank, world_size=world_size,
        )
        if (Path(data_dir) / "val" / "meta.json").exists():
            packed_val = PackedCrystDataset(
                Path(data_dir) / "val", cond_props=cond_props, max_atoms=max_atoms,
                max_e_hull=max_e_hull, max_samples=max_test_samples,
            )
            if extra_val_max_e_hull is not None:
                packed_extra_val = PackedCrystDataset(
                    Path(data_dir) / "val", cond_props=cond_props, max_atoms=max_atoms,
                    max_e_hull=extra_val_max_e_hull, max_samples=max_test_samples,
                )
        if is_main:
            for name, ds in (("train", packed_train), ("val", packed_val), ("extra val", packed_extra_val)):
                if ds is not None:
                    print(f"{name}: {len(ds):,} structures from {ds.split_dir} ({'; '.join(ds.filter_log) or 'unfiltered'})")
            print(f"{len(train_batch_sampler):,} steps per epoch per rank at <= {train_batch_sampler.max_edges:,} edges per batch")

    per_step_schedule = warmup_steps is not None
    if per_step_schedule:
        if train_batch_sampler is None:
            raise ValueError("--warmup_steps needs a packed --data_dir")
        steps_per_epoch = len(train_batch_sampler)
        total_steps = max(1, epochs * steps_per_epoch)
        min_lr_ratio = 1e-6 / lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return 0.01 + 0.99 * step / max(1, warmup_steps)
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        optimizer = torch.optim.AdamW(params_to_train, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif model_type in ("geov2", "geo_orb"):
        optimizer = torch.optim.AdamW(params_to_train, lr=lr, weight_decay=1e-4)
        warmup_epochs = min(10, max(1, epochs // 10))
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
        )
    else:
        optimizer = torch.optim.Adam(params_to_train, lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=15, min_lr=1e-4)

    start_epoch = 0
    resume_path = None
    if resume is not None:
        if resume == "auto":
            resume_path = Path(ckpt_path)
        elif resume.startswith("wandb://"):
            resume_path = _resolve_wandb_checkpoint(resume)
        else:
            resume_path = Path(resume)

    resume_wandb_id = None
    resume_best_val_loss = float("inf")
    if resume_path and resume_path.exists():
        if is_main:
            print(f"Loading checkpoint from '{resume_path}' to resume...")
        ckpt_data = torch.load(resume_path, map_location=dev, weights_only=False)
        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
            if model_type == "orb":
                _load_adapter_state_dict(model.decoder, ckpt_data["model_state_dict"])
            else:
                model.load_state_dict(ckpt_data["model_state_dict"])
            if "optimizer_state_dict" in ckpt_data:
                optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt_data:
                scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
            if "ema_state_dict" in ckpt_data and hasattr(model, "load_ema_state_dict"):
                model.load_ema_state_dict(ckpt_data["ema_state_dict"], device=dev)
            start_epoch = ckpt_data.get("epoch", 0)
            resume_wandb_id = ckpt_data.get("wandb_run_id")
            resume_best_val_loss = ckpt_data.get("best_val_loss", float("inf"))
            if is_main:
                print(
                    f"Successfully resumed from epoch {start_epoch} (Last train loss: {ckpt_data.get('train_loss', 'N/A')})"
                )
        elif isinstance(ckpt_data, dict):
            if model_type == "orb":
                _load_adapter_state_dict(model.decoder, ckpt_data)
            else:
                model.load_state_dict(ckpt_data)
            if is_main:
                print(f"Loaded model weights from legacy checkpoint '{resume_path}'")

    # Restored on resume: otherwise the first evaluation after every resume counts as
    # an improvement and overwrites the best checkpoint with a possibly worse one.
    best_val_loss = resume_best_val_loss

    def save_checkpoint(epoch_idx: int, train_loss_val: float, val_loss_val: float | None = None) -> None:
        nonlocal best_val_loss
        if not is_main:
            return
        # Track the best validation loss independently of W&B: the caller uses it
        # to decide whether an evaluation improved, so tying it to the logger
        # would leave it at infinity for every run without --wandb.
        is_best = (
            val_loss_val is not None
            and not np.isnan(val_loss_val)
            and val_loss_val < best_val_loss
        )
        if is_best:
            best_val_loss = val_loss_val
        ckpt = {
            "epoch": epoch_idx + 1,
            "model_type": model_type,
            "model_config": {
                "hidden_dim": arch_h,
                "num_layers": arch_l,
                "cond_props": cond_props,
                "cond_drop_prob": cond_drop_prob,
            },
            "model_state_dict": _adapter_state_dict(model.decoder) if model_type == "orb" else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss_val,
            "val_loss": val_loss_val,
            "best_val_loss": best_val_loss,
            "wandb_run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
        }
        if hasattr(model, "ema_state_dict"):
            ema_sd = model.ema_state_dict()
            if ema_sd is not None:
                ckpt["ema_state_dict"] = ema_sd
        p = Path(ckpt_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = p.with_suffix(".tmp")
        torch.save(ckpt, tmp_p)
        tmp_p.replace(p)
        print(f"Saved checkpoint (epoch {epoch_idx + 1}) to {ckpt_path}")

        if is_best:
            best_p = p.with_name(f"{p.stem}_best{p.suffix}")
            torch.save(ckpt, best_p.with_suffix(".tmp"))
            best_p.with_suffix(".tmp").replace(best_p)
            print(f"  new best val loss {val_loss_val:.4f} -> also saved {best_p.name}")

        # Upload checkpoint as a W&B Artifact for portability
        if wandb_run is not None:
            import wandb

            artifact = wandb.Artifact(
                name=f"model-{wandb_run.id}",
                type="model",
                metadata={
                    "epoch": epoch_idx + 1,
                    "model_type": model_type,
                    "train_loss": train_loss_val,
                    "val_loss": val_loss_val,
                },
            )
            artifact.add_file(str(p))
            aliases = [f"epoch-{epoch_idx + 1}", "latest"]
            if is_best:
                aliases.append("best")
            wandb_run.log_artifact(artifact, aliases=aliases)
            print(f"Uploaded checkpoint artifact to W&B (aliases: {aliases})")

    stop_requested = False
    longest_epoch_s = 0.0

    def _sig_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        if is_main:
            print("\n[INFO] Interrupt signal received! Saving checkpoint and gracefully exiting...")

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    wandb_run = None
    if is_main and use_wandb:
        import wandb

        wandb_kwargs = {
            "project": wandb_project,
            "entity": wandb_entity,
            "config": {
                "model_type": model_type,
                "orb_model": orb_model,
                "use_orb_node_features": use_orb_node_features,
                "enforce_zero_force": enforce_zero_force,
                "force_residual": force_residual,
                "batch_size": batch_size,
                "world_size": world_size,
                "effective_batch_size": batch_size * world_size,
                "lr": lr,
                "epochs": epochs,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "max_train_samples": max_train_samples,
                "max_test_samples": max_test_samples,
                "max_e_hull": max_e_hull,
                "max_atoms": max_atoms,
                "hidden_dim_effective": arch_h,
                "num_layers_effective": arch_l,
                "train_csv": str(train_csv),
                "test_csv": str(test_csv),
                "data_dir": data_dir,
                "data_meta": packed_train.meta if packed_train is not None else None,
                "n_train": len(packed_train) if packed_train is not None else None,
                "n_val": len(packed_val) if packed_val is not None else None,
                "max_edges_per_batch": max_edges_per_batch,
                "steps_per_epoch": len(train_batch_sampler) if train_batch_sampler is not None else None,
                "cond_props": cond_props,
                "cond_drop_prob": cond_drop_prob,
                "warmup_steps": warmup_steps,
                "extra_val_max_e_hull": extra_val_max_e_hull,
            },
        }
        if wandb_name:
            wandb_kwargs["name"] = wandb_name
        if wandb_tags:
            wandb_kwargs["tags"] = wandb_tags
        if resume_wandb_id:
            wandb_kwargs["id"] = resume_wandb_id
            wandb_kwargs["resume"] = "allow"
        wandb_run = wandb.init(**wandb_kwargs)
        # W&B cannot read the commit itself where git is absent (e.g. inside a
        # container), so a launcher passes it in. A chained run records the commit
        # of its latest link; earlier links' commits are in their job logs.
        if os.environ.get("DIFFCSP_GIT_COMMIT"):
            wandb_run.config.update(
                {"git_commit": os.environ["DIFFCSP_GIT_COMMIT"], "git_branch": os.environ.get("DIFFCSP_GIT_BRANCH")},
                allow_val_change=True,
            )

    train_path = Path(train_csv)
    if packed_train is None and not train_path.exists():
        if is_main:
            print(f"Dataset '{train_csv}' not found. Exiting training.")
        return

    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": (dev.type == "cuda"),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["worker_init_fn"] = _worker_init_fn

    extra_val_loader = None
    if packed_train is not None:
        train_sampler = train_batch_sampler
        train_loader = DataLoader(
            packed_train, batch_sampler=train_batch_sampler,
            generator=torch.Generator().manual_seed(rank), **loader_kwargs,
        )
    # In DDP, avoid race conditions during initial dataset cache extraction
    elif is_ddp:
        if is_main:
            train_set = CrystDataset(train_path, mode="train_sym", max_samples=max_train_samples, cache_dir=cache_dir, max_energy_above_hull=max_e_hull, max_atoms=max_atoms)
        dist.barrier()
        if not is_main:
            train_set = CrystDataset(train_path, mode="train_sym", max_samples=max_train_samples, cache_dir=cache_dir, max_energy_above_hull=max_e_hull, max_atoms=max_atoms)
        dist.barrier()
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, generator=torch.Generator().manual_seed(rank), **loader_kwargs)
    else:
        train_set = CrystDataset(train_path, mode="train_sym", max_samples=max_train_samples, cache_dir=cache_dir, max_energy_above_hull=max_e_hull, max_atoms=max_atoms)
        # The shuffle has its own generator, reseeded every epoch, and the loader (which
        # draws a worker base seed whenever it spawns workers -- once per process with
        # persistent workers) another: neither then touches the global RNG, so the
        # order and the diffusion noise of an epoch do not depend on where a run resumed.
        shuffle_generator = torch.Generator()
        train_sampler = RandomSampler(train_set, generator=shuffle_generator)
        train_loader = DataLoader(
            train_set, sampler=train_sampler, batch_size=batch_size,
            generator=torch.Generator().manual_seed(rank), **loader_kwargs,
        )

    if async_dataloader and dev.type == "cuda":
        train_loader = PrefetchLoader(train_loader, device=dev)

    test_loader = None
    if packed_train is not None:
        val_kwargs = dict(loader_kwargs, num_workers=min(num_workers, 4))
        if val_kwargs["num_workers"] == 0:
            val_kwargs = {"pin_memory": loader_kwargs["pin_memory"]}
        for ds, which in ((packed_val, "val"), (packed_extra_val, "extra")):
            if ds is None:
                continue
            loader = DataLoader(
                ds,
                batch_sampler=EdgeBudgetBatchSampler(ds.sizes(), max_edges=max_edges_per_batch or 64_000, shuffle=False),
                **val_kwargs,
            )
            if async_dataloader and dev.type == "cuda":
                loader = PrefetchLoader(loader, device=dev)
            if which == "val":
                test_loader = loader
            else:
                extra_val_loader = loader
    elif test_csv and Path(test_csv).exists():
        if is_ddp:
            if is_main:
                test_set = CrystDataset(Path(test_csv), mode="test_sym", max_samples=max_test_samples, cache_dir=cache_dir, max_energy_above_hull=max_e_hull, max_atoms=max_atoms)
            dist.barrier()
            if not is_main:
                test_set = CrystDataset(Path(test_csv), mode="test_sym", max_samples=max_test_samples, cache_dir=cache_dir, max_energy_above_hull=max_e_hull, max_atoms=max_atoms)
            dist.barrier()
        else:
            test_set = CrystDataset(Path(test_csv), mode="test_sym", max_samples=max_test_samples, cache_dir=cache_dir, max_energy_above_hull=max_e_hull, max_atoms=max_atoms)
        test_loader = DataLoader(
            test_set, shuffle=False, batch_size=batch_size,
            generator=torch.Generator().manual_seed(rank), **loader_kwargs,
        )
        if async_dataloader and dev.type == "cuda":
            test_loader = PrefetchLoader(test_loader, device=dev)

    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    epoch_iter = trange(start_epoch, epochs, desc="Epochs") if is_main else range(start_epoch, epochs)
    for epoch in epoch_iter:
        epoch_start = time.monotonic()
        # Reseed per epoch so that the shuffle and the diffusion noise of epoch k do
        # not depend on where the run was last resumed: a run chained across
        # walltime-limited jobs then draws the same randomness as an uninterrupted one.
        set_random_seed(17 + rank + 1_000_003 * epoch)
        if isinstance(train_sampler, (DistributedSampler, EdgeBudgetBatchSampler)):
            train_sampler.set_epoch(epoch)
        else:
            shuffle_generator.manual_seed(17 + 1_000_003 * epoch)
        ddp_model.train()
        train_losses = []
        n_train_structures = 0
        n_skipped_steps = 0

        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False) if is_main else train_loader
        for batch in batch_iter:
            batch = batch.to(dev, non_blocking=True)
            if model_type in ("geo", "geov2", "geo_orb"):
                if not hasattr(batch, "multiplicities"):
                    mult_per_anchor = torch.bincount(batch.anchor_index)
                    batch.multiplicities = mult_per_anchor[batch.anchor_index].clamp(0, 192)
                if not hasattr(batch, "dofs"):
                    batch.dofs = torch.round(
                        torch.diagonal(batch.ops[batch.anchor_index, :3, :3], dim1=-2, dim2=-1).sum(-1)
                    ).long().clamp(0, 3)
            out = ddp_model(batch)
            loss = out.get("loss")
            if loss is None:
                continue
            if is_ddp:
                # Every rank must reach the gradient all-reduce, so a non-finite loss
                # is not skipped before backward (that would hang the others): DDP
                # averages the gradients, all ranks then see the same non-finite
                # values, and all skip the same step.
                loss.backward()
                grads = [p.grad for p in params_to_train if p.grad is not None]
                grads_ok = bool(torch.isfinite(torch.stack(torch._foreach_norm(grads))).all())
                if not grads_ok:
                    n_skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    if per_step_schedule:
                        scheduler.step()
                    continue
            elif torch.isnan(loss):
                continue
            else:
                loss.backward()

            train_losses.append(loss.item())
            n_train_structures += batch.num_graphs
            torch.nn.utils.clip_grad_value_(params_to_train, 0.4)
            optimizer.step()
            if per_step_schedule:
                scheduler.step()
            if model_type in ("geov2", "geo_orb"):
                if is_ddp:
                    ddp_model.module.update_ema()
                else:
                    model.update_ema()
            optimizer.zero_grad(set_to_none=True)

        local_avg_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        if is_ddp:
            loss_tensor = torch.tensor([local_avg_loss if not np.isnan(local_avg_loss) else 0.0], device=dev)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = float((loss_tensor / world_size).item())
        else:
            avg_loss = local_avg_loss

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_loss)
        elif not per_step_schedule:
            scheduler.step()
        log_data = {"epoch": epoch, "train_loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]}
        epoch_train_s = time.monotonic() - epoch_start
        if is_ddp:
            counts = torch.tensor([n_train_structures, n_skipped_steps], device=dev)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            n_train_structures, n_skipped_steps = (int(c) for c in counts.tolist())
        log_data.update(
            epoch_train_minutes=epoch_train_s / 60,
            train_structures_per_s=n_train_structures / max(epoch_train_s, 1e-9),
            skipped_steps=n_skipped_steps,
            peak_gpu_mem_gib=torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0,
        )
        if is_main:
            print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Periodic Evaluation
        if is_main and test_loader is not None and ((epoch + 1) % eval_freq == 0 or (epoch + 1) == epochs):
            model.eval()
            ema_ctx = model.ema_scope() if (hasattr(model, "ema_scope") and model.ema is not None) else nullcontext()
            with ema_ctx, torch.no_grad():

                def _val_loss(loader: Any, uncond: bool = False) -> float:
                    # Same diffusion times and noise at every evaluation, so the curve
                    # moves with the model rather than with the draw.
                    set_random_seed(123_457)
                    losses = []
                    for vb in loader:
                        vb = vb.to(dev, non_blocking=True)
                        if uncond:
                            vb.props = None
                        vloss = model.training_step(vb, 0)
                        if vloss is not None:
                            losses.append(vloss.item())
                    return float(np.mean(losses)) if losses else float("nan")

                if packed_val is not None:
                    log_data["val_loss"] = avg_val_loss = _val_loss(test_loader)
                    if cond_props and cond_drop_prob > 0:
                        log_data["val_loss_uncond"] = _val_loss(test_loader, uncond=True)
                    if extra_val_loader is not None:
                        key = f"val_loss_ehull{extra_val_max_e_hull:g}"
                        log_data[key] = _val_loss(extra_val_loader)
                        if cond_props and cond_drop_prob > 0:
                            log_data[key + "_uncond"] = _val_loss(extra_val_loader, uncond=True)
                    print(f"Epoch {epoch:03d} | " + " | ".join(f"{k}: {v:.4f}" for k, v in log_data.items() if k.startswith("val_loss")))
                val_losses = []
                for batch in (tqdm(test_loader, desc="Validating", leave=False) if packed_val is None else []):
                    batch = batch.to(dev, non_blocking=True)
                    if model_type in ("geo", "geov2", "geo_orb"):
                        if not hasattr(batch, "multiplicities"):
                            mult_per_anchor = torch.bincount(batch.anchor_index)
                            batch.multiplicities = mult_per_anchor[batch.anchor_index].clamp(0, 192)
                        if not hasattr(batch, "dofs"):
                            batch.dofs = torch.round(
                                torch.diagonal(batch.ops[batch.anchor_index, :3, :3], dim1=-2, dim2=-1).sum(-1)
                            ).long().clamp(0, 3)
                    loss = model.training_step(batch, 0)
                    if loss is not None:
                        val_losses.append(loss.item())

                if packed_val is None:
                    avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
                    log_data["val_loss"] = avg_val_loss
                    print(f"Epoch {epoch:03d} | Val Loss: {avg_val_loss:.4f}")

                if eval_sample:
                    frac_coords_list, num_atoms_list, atom_types_list, lattices_list, input_data_list = (
                        [],
                        [],
                        [],
                        [],
                        [],
                    )
                    for batch in tqdm(test_loader, desc="Sampling Validation", leave=False):
                        batch = batch.to(dev, non_blocking=True)
                        outputs, _ = model.sample(batch, disable_progress=True)
                        frac_coords_list.append(outputs["frac_coords"].detach().cpu())
                        num_atoms_list.append(outputs["num_atoms"].detach().cpu())
                        atom_types_list.append(outputs["atom_types"].detach().cpu())
                        lattices_list.append(outputs["lattices"].detach().cpu())
                        input_data_list.extend(batch.to_data_list())

                    frac_coords = torch.cat(frac_coords_list, dim=0)
                    num_atoms = torch.cat(num_atoms_list, dim=0)
                    atom_types = torch.cat(atom_types_list, dim=0)
                    lattices = torch.cat(lattices_list, dim=0)

                    preds_list = []
                    start_idx = 0
                    for n_atoms, lat in zip(num_atoms, lattices):
                        c_frac = frac_coords.narrow(0, start_idx, n_atoms)
                        c_types = atom_types.narrow(0, start_idx, n_atoms)
                        preds_list.append(
                            Structure(lattice=lat, species=c_types, coords=c_frac, coords_are_cartesian=False)
                        )
                        start_idx += n_atoms

                    input_list = [
                        Structure(
                            lattice=Lattice.from_parameters(*(s.lengths.tolist()[0] + s.angles.tolist()[0])),
                            species=s.atom_types.detach().cpu(),
                            coords=s.frac_coords.detach().cpu(),
                            coords_are_cartesian=False,
                        )
                        for s in input_data_list
                    ]

                    matches = [matcher.get_rms_dist(s1, s2) is not None for s1, s2 in zip(input_list, preds_list)]
                    match_rate = float(np.mean(matches)) if matches else 0.0
                    log_data["match_rate"] = match_rate
                    print(f"Validation Match Rate: {match_rate * 100:.2f}%")

        # Save on the periodic schedule, and additionally on any evaluation that
        # improves validation loss. Without the second condition the "best"
        # alias could only ever land on epochs where the save and eval schedules
        # coincide -- every lcm(save_freq, eval_freq) epochs -- so the true
        # minimum was usually evaluated and then discarded.
        current_val_loss = log_data.get("val_loss")
        due_for_save = (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs
        val_improved = (
            current_val_loss is not None
            and not np.isnan(current_val_loss)
            and current_val_loss < best_val_loss
        )
        if due_for_save or val_improved:
            save_checkpoint(epoch, avg_loss, current_val_loss)

        # Stop while a further epoch as long as the longest so far still fits, so a
        # walltime-limited job checkpoints on its own terms instead of relying on a
        # signal reaching this process (container wrappers do not all forward one).
        if time_limit_hours is not None:
            longest_epoch_s = max(longest_epoch_s, time.monotonic() - epoch_start)
            if time.monotonic() - t_start + longest_epoch_s > 3600 * time_limit_hours and epoch + 1 < epochs:
                stop_requested = True
                if is_main:
                    print(f"[INFO] Time limit {time_limit_hours} h: no room for another {longest_epoch_s / 60:.1f} min epoch.")

        # Check for interrupt / stop request across ranks
        if is_ddp:
            stop_t = torch.tensor([1 if stop_requested else 0], device=dev)
            dist.all_reduce(stop_t, op=dist.ReduceOp.MAX)
            if stop_t.item() > 0:
                stop_requested = True

        if stop_requested:
            save_checkpoint(epoch, avg_loss, log_data.get("val_loss"))
            if is_main and wandb_run is not None:
                wandb_run.log(log_data)
            if is_main:
                print(f"\n[INFO] Training stopped gracefully at epoch {epoch}. Resume anytime with --resume {ckpt_path}.")
            if is_ddp:
                dist.barrier()
            break

        if is_ddp:
            dist.barrier()

        if is_main and wandb_run is not None:
            wandb_run.log(log_data)

    if is_main and wandb_run is not None:
        wandb_run.finish()

    if is_ddp:
        dist.destroy_process_group()


def main() -> None:
    """CLI entrypoint for training."""
    parser = argparse.ArgumentParser(description="DiffCSP++ Training CLI")
    parser.add_argument("--train_csv", type=str, default="data/mp-20/train.csv", help="Path to training CSV")
    parser.add_argument("--test_csv", type=str, default="data/mp-20/test.csv", help="Path to test CSV")
    parser.add_argument(
        "--model",
        type=str,
        choices=["orb", "cspnet", "wyckoff", "asymm", "painn", "wyckoff_painn", "geo", "geov2", "geo_orb"],
        default="orb",
        help="Model backbone",
    )
    parser.add_argument(
        "--orb_model",
        type=str,
        default="orb-v3",
        help="Pretrained ORB backbone: an alias (orb-v3, orb-v3-direct-omat, orb-v2, ...) "
        "or an orb_models.pretrained loader name (e.g. orb_v3_direct_20_mpa)",
    )
    parser.add_argument("--mock_orb", action="store_true", help="Use lightweight mock ORB backbone")
    parser.add_argument(
        "--no_orb_node_features",
        dest="use_orb_node_features",
        action="store_false",
        help="Condition only on ORB forces and stress, without its learned atomic representations",
    )
    parser.set_defaults(use_orb_node_features=True)
    parser.add_argument(
        "--enforce_zero_force",
        action="store_true",
        help="Constrain the coordinate score to gamma*f_frac + v_perp (gamma>0); "
        "guarantees score=0 only when the ORB force is 0, but permits one direction along the force",
    )
    parser.add_argument(
        "--force_residual",
        action="store_true",
        help="Add a time-gated ORB force residual to the coordinate score instead of the hard constraint",
    )

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--eval_freq", type=int, default=10, help="Evaluation frequency in epochs")
    parser.add_argument(
        "--hidden_dim", type=int, default=None,
        help="Denoiser hidden dimension (default: 128 for --model orb, 512 for --model cspnet/geo)",
    )
    parser.add_argument(
        "--num_layers", type=int, default=None,
        help="Number of denoiser layers (default: 2 for --model orb, 6 for --model cspnet/geo)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_path", type=str, default="diffcsp_ckpt.pt", help="Checkpoint save path")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        nargs="?",
        const="auto",
        help="Resume checkpoint path (or auto for --ckpt_path)",
    )
    parser.add_argument("--save_freq", type=int, default=1, help="Checkpoint saving frequency in epochs")
    parser.add_argument(
        "--wandb",
        dest="wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Weights & Biases logging (default: True; use --no-wandb to disable)",
    )
    parser.add_argument("--wandb_project", type=str, default="diffcsp", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max test samples")
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Directory for the preprocessed dataset cache (default: alongside the CSV)",
    )
    parser.add_argument(
        "--max_e_hull", type=float, default=None,
        help="Keep only structures with energy_above_hull <= this value, in eV",
    )
    parser.add_argument(
        "--max_atoms", type=int, default=None,
        help="Drop structures whose conventional cell exceeds this many atoms "
        "(CSPNet's intra-cell graph is fully connected, so cost grows as N^2)",
    )
    parser.add_argument("--eval_sample", action="store_true", help="Run full generative sampling validation")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader worker processes per rank")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor")
    parser.add_argument(
        "--no_async_dataloader",
        dest="async_dataloader",
        action="store_false",
        help="Disable asynchronous PrefetchLoader",
    )
    parser.set_defaults(async_dataloader=True)
    parser.add_argument(
        "--time_limit_hours", type=float, default=None,
        help="Checkpoint and exit cleanly once another epoch would overrun this many hours of wall time",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Packed dataset root (scripts/pack_dataset.py) with train/ and val/ splits; replaces --train_csv/--test_csv",
    )
    parser.add_argument(
        "--max_edges_per_batch", type=int, default=None,
        help="With --data_dir: batch by a budget of sum(N^2) full-cell edges per rank instead of --batch_size",
    )
    parser.add_argument(
        "--cond_props", nargs="*", default=[],
        help="Scalar properties to condition on (geov2 only), e.g. energy_above_hull",
    )
    parser.add_argument(
        "--cond_drop_prob", type=float, default=0.0,
        help="Probability of dropping a structure's condition in training (> 0 enables classifier-free guidance)",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=None,
        help="Per-step schedule: linear warmup over this many steps, then cosine to 1e-6 over --epochs",
    )
    parser.add_argument(
        "--extra_val_max_e_hull", type=float, default=None,
        help="Also report the validation loss on val structures with E_hull <= this value",
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_tags", nargs="*", default=None, help="W&B run tags")
    args = parser.parse_args()

    train(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        model_type=args.model,
        orb_model=args.orb_model,
        mock_orb=args.mock_orb,
        use_orb_node_features=args.use_orb_node_features,
        enforce_zero_force=args.enforce_zero_force,
        force_residual=args.force_residual,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        eval_freq=args.eval_freq,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
        ckpt_path=args.ckpt_path,
        resume=args.resume,
        save_freq=args.save_freq,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        cache_dir=args.cache_dir,
        max_e_hull=args.max_e_hull,
        max_atoms=args.max_atoms,
        eval_sample=args.eval_sample,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        async_dataloader=args.async_dataloader,
        time_limit_hours=args.time_limit_hours,
        data_dir=args.data_dir,
        max_edges_per_batch=args.max_edges_per_batch,
        cond_props=args.cond_props,
        cond_drop_prob=args.cond_drop_prob,
        warmup_steps=args.warmup_steps,
        extra_val_max_e_hull=args.extra_val_max_e_hull,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
    )


if __name__ == "__main__":
    main()
