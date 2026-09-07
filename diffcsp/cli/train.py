"""Unified training CLI for DiffCSP++ (standard GNN and ORB MLIP adapter)."""

import argparse
import logging
import os
import random
import signal
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange

from diffcsp.data.dataset import CrystDataset
from diffcsp.models.diffusion import CSPDiffusion
from diffcsp.models.diffusion_orb import CSPDiffusionORB

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 17) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(
    train_csv: str = "train.csv",
    test_csv: str | None = "test.csv",
    model_type: str = "orb",
    orb_model: str = "orb-v2",
    mock_orb: bool = False,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 100,
    eval_freq: int = 10,
    hidden_dim: int = 128,
    num_layers: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ckpt_path: str = "diffcsp_ckpt.pt",
    resume: str | None = None,
    save_freq: int = 1,
    use_wandb: bool = False,
    wandb_project: str = "diffcsp",
    wandb_entity: str | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    eval_sample: bool = False,
) -> None:
    """Executes model training with periodic evaluation."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = world_size > 1

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
    else:
        dev = torch.device(device)

    is_main = (not is_ddp) or (rank == 0)

    set_random_seed(17 + rank)

    if is_main:
        print("=" * 65)
        print(f"DiffCSP++ Training | Model: {model_type.upper()} | World Size: {world_size} | Device: {dev}")
        print("=" * 65)

    if model_type == "orb":
        model = CSPDiffusionORB(
            device=dev,
            orb_model_name=orb_model,
            use_mock_orb=mock_orb,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(dev)
        params_to_train = model.get_trainable_parameters()
        counts = model.count_parameters()
        if is_main:
            print(f"Trainable params: {counts['trainable']:,} ({counts['trainable_pct']:.2f}%)")
            print(f"Frozen params:    {counts['frozen']:,}")
    else:
        model = CSPDiffusion(device=dev).to(dev)
        params_to_train = list(model.parameters())
        if is_main:
            print(f"Total params: {sum(p.numel() for p in params_to_train):,}")

    if is_ddp:
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        ddp_model = model

    optimizer = torch.optim.Adam(params_to_train, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=15, min_lr=1e-4)

    start_epoch = 0
    resume_path = None
    if resume is not None:
        resume_path = Path(ckpt_path) if resume == "auto" else Path(resume)

    resume_wandb_id = None
    if resume_path and resume_path.exists():
        if is_main:
            print(f"Loading checkpoint from '{resume_path}' to resume...")
        ckpt_data = torch.load(resume_path, map_location=dev, weights_only=False)
        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
            if model_type == "orb":
                model.decoder.load_state_dict(ckpt_data["model_state_dict"])
            else:
                model.load_state_dict(ckpt_data["model_state_dict"])
            if "optimizer_state_dict" in ckpt_data:
                optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt_data:
                scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
            start_epoch = ckpt_data.get("epoch", 0)
            resume_wandb_id = ckpt_data.get("wandb_run_id")
            if is_main:
                print(
                    f"Successfully resumed from epoch {start_epoch} (Last train loss: {ckpt_data.get('train_loss', 'N/A')})"
                )
        elif isinstance(ckpt_data, dict):
            if model_type == "orb":
                model.decoder.load_state_dict(ckpt_data)
            else:
                model.load_state_dict(ckpt_data)
            if is_main:
                print(f"Loaded model weights from legacy checkpoint '{resume_path}'")

    def save_checkpoint(epoch_idx: int, train_loss_val: float, val_loss_val: float | None = None) -> None:
        if not is_main:
            return
        ckpt = {
            "epoch": epoch_idx + 1,
            "model_type": model_type,
            "model_state_dict": model.decoder.state_dict() if model_type == "orb" else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss_val,
            "val_loss": val_loss_val,
            "wandb_run_id": getattr(wandb_run, "id", None) if wandb_run is not None else None,
        }
        p = Path(ckpt_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = p.with_suffix(".tmp")
        torch.save(ckpt, tmp_p)
        tmp_p.replace(p)
        print(f"Saved checkpoint (epoch {epoch_idx + 1}) to {ckpt_path}")

    stop_requested = False

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
                "batch_size": batch_size,
                "world_size": world_size,
                "effective_batch_size": batch_size * world_size,
                "lr": lr,
                "epochs": epochs,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "max_train_samples": max_train_samples,
                "max_test_samples": max_test_samples,
            },
        }
        if resume_wandb_id:
            wandb_kwargs["id"] = resume_wandb_id
            wandb_kwargs["resume"] = "allow"
        wandb_run = wandb.init(**wandb_kwargs)

    train_path = Path(train_csv)
    if not train_path.exists():
        if is_main:
            print(f"Dataset '{train_csv}' not found. Exiting training.")
        return

    # In DDP, avoid race conditions during initial dataset cache extraction
    if is_ddp:
        if is_main:
            train_set = CrystDataset(train_path, mode="train_sym", max_samples=max_train_samples)
        dist.barrier()
        if not is_main:
            train_set = CrystDataset(train_path, mode="train_sym", max_samples=max_train_samples)
        dist.barrier()
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    else:
        train_set = CrystDataset(train_path, mode="train_sym", max_samples=max_train_samples)
        train_sampler = None
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

    test_loader = None
    if test_csv and Path(test_csv).exists():
        if is_ddp:
            if is_main:
                test_set = CrystDataset(Path(test_csv), mode="test_sym", max_samples=max_test_samples)
            dist.barrier()
            if not is_main:
                test_set = CrystDataset(Path(test_csv), mode="test_sym", max_samples=max_test_samples)
            dist.barrier()
        else:
            test_set = CrystDataset(Path(test_csv), mode="test_sym", max_samples=max_test_samples)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    epoch_iter = trange(start_epoch, epochs, desc="Epochs") if is_main else range(start_epoch, epochs)
    for epoch in epoch_iter:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        ddp_model.train()
        train_losses = []

        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False) if is_main else train_loader
        for batch in batch_iter:
            batch = batch.to(dev)
            out = ddp_model(batch)
            loss = out.get("loss")
            if loss is None or torch.isnan(loss):
                continue

            train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_value_(params_to_train, 0.4)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        local_avg_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        if is_ddp:
            loss_tensor = torch.tensor([local_avg_loss if not np.isnan(local_avg_loss) else 0.0], device=dev)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = float((loss_tensor / world_size).item())
        else:
            avg_loss = local_avg_loss

        scheduler.step(avg_loss)
        log_data = {"epoch": epoch, "train_loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]}
        if is_main:
            print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Periodic Evaluation
        if is_main and test_loader is not None and ((epoch + 1) % eval_freq == 0 or (epoch + 1) == epochs):
            model.eval()
            with torch.no_grad():
                val_losses = []
                for batch in tqdm(test_loader, desc="Validating", leave=False):
                    batch = batch.to(dev)
                    loss = model.training_step(batch, 0)
                    if loss is not None:
                        val_losses.append(loss.item())

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
                        batch = batch.to(dev)
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

        # Save checkpoint periodically
        if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
            save_checkpoint(epoch, avg_loss, log_data.get("val_loss"))

        # Check for interrupt / stop request across ranks
        if is_ddp:
            stop_t = torch.tensor([1 if stop_requested else 0], device=dev)
            dist.all_reduce(stop_t, op=dist.ReduceOp.MAX)
            if stop_t.item() > 0:
                stop_requested = True

        if stop_requested:
            save_checkpoint(epoch, avg_loss, log_data.get("val_loss"))
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
    parser.add_argument("--train_csv", type=str, default="train.csv", help="Path to training CSV")
    parser.add_argument("--test_csv", type=str, default="test.csv", help="Path to test CSV")
    parser.add_argument("--model", type=str, choices=["orb", "cspnet"], default="orb", help="Model backbone")
    parser.add_argument("--orb_model", type=str, default="orb-v2", help="Pretrained ORB model variant")
    parser.add_argument("--mock_orb", action="store_true", help="Use lightweight mock ORB backbone")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--eval_freq", type=int, default=10, help="Evaluation frequency in epochs")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Adapter hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of adapter layers")
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
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="diffcsp", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max test samples")
    parser.add_argument("--eval_sample", action="store_true", help="Run full generative sampling validation")
    args = parser.parse_args()

    train(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        model_type=args.model,
        orb_model=args.orb_model,
        mock_orb=args.mock_orb,
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
        eval_sample=args.eval_sample,
    )


if __name__ == "__main__":
    main()
