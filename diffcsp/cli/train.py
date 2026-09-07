"""Unified training CLI for DiffCSP++ (standard GNN and ORB MLIP adapter)."""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
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
    use_wandb: bool = False,
    wandb_project: str = "diffcsp",
) -> None:
    """Executes model training with periodic evaluation."""
    set_random_seed(17)
    dev = torch.device(device)

    print("=" * 65)
    print(f"DiffCSP++ Training | Model: {model_type.upper()} | Device: {dev}")
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
        print(f"Trainable params: {counts['trainable']:,} ({counts['trainable_pct']:.2f}%)")
        print(f"Frozen params:    {counts['frozen']:,}")
    else:
        model = CSPDiffusion(device=dev).to(dev)
        params_to_train = list(model.parameters())
        print(f"Total params: {sum(p.numel() for p in params_to_train):,}")

    optimizer = torch.optim.Adam(params_to_train, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=15, min_lr=1e-4)

    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            config={
                "model_type": model_type,
                "batch_size": batch_size,
                "lr": lr,
                "epochs": epochs,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
            },
        )

    train_path = Path(train_csv)
    if not train_path.exists():
        print(f"Dataset '{train_csv}' not found. Exiting training.")
        return

    train_set = CrystDataset(train_path, mode="train_sym")
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

    test_loader = None
    if test_csv and Path(test_csv).exists():
        test_set = CrystDataset(Path(test_csv), mode="test_sym")
        test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    for epoch in trange(epochs, desc="Epochs"):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(dev)
            loss = model.training_step(batch, 0)
            if loss is None:
                continue

            train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_value_(params_to_train, 0.4)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        scheduler.step(avg_loss)
        log_data = {"epoch": epoch, "train_loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]}
        print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Periodic Evaluation
        if test_loader is not None and (epoch + 1) % eval_freq == 0:
            model.eval()
            with torch.no_grad():
                frac_coords_list, num_atoms_list, atom_types_list, lattices_list, input_data_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for batch in tqdm(test_loader, desc="Validating", leave=False):
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

            # Checkpoint saving
            save_dict = model.decoder.state_dict() if model_type == "orb" else model.state_dict()
            torch.save(save_dict, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        if wandb_run is not None:
            wandb_run.log(log_data)


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
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="diffcsp", help="W&B project name")
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
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
