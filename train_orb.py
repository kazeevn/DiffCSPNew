import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import trange, tqdm

from dataset import CrystDataset
from diffusion_orb import CSPDiffusionORB


def set_random_seed(seed: int = 17):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train DiffCSP++ with Frozen ORB MLIP + Lightweight Adapter")
    parser.add_argument("--train_csv", type=str, default="train.csv", help="Path to training CSV")
    parser.add_argument("--test_csv", type=str, default="test.csv", help="Path to test CSV")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for adapter")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Adapter hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of adapter CSPLayers")
    parser.add_argument("--orb_model", type=str, default="orb-v2", help="ORB model variant")
    parser.add_argument("--mock_orb", action="store_true", help="Use lightweight mock ORB backbone for testing")
    parser.add_argument("--no_force_residual", action="store_true", help="Disable direct force residual prior")
    parser.add_argument("--no_zero_force_condition", action="store_true", help="Disable by-design S=0 ==> F=0 condition")
    parser.add_argument("--zero_stress_condition", action="store_true", help="Enable by-design S_lattice=0 ==> Stress=0 condition")
    parser.add_argument("--gamma_min", type=float, default=1e-3, help="Minimum along-force scaling factor")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_path", type=str, default="orb_diffcsp_ckpt.pt", help="Checkpoint save path")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()

    set_random_seed(17)
    device = torch.device(args.device)

    print("=" * 60)
    print("DiffCSP++ with Frozen ORB MLIP Backbone & Lightweight Adapter")
    print(f"Device: {device} | ORB Model: {args.orb_model} | Mock: {args.mock_orb}")
    print(f"Adapter Hidden Dim: {args.hidden_dim} | Layers: {args.num_layers}")
    print(f"Zero-force condition: {not args.no_zero_force_condition} | gamma_min: {args.gamma_min}")
    print("=" * 60)

    # Initialize model
    model = CSPDiffusionORB(
        device=device,
        orb_model_name=args.orb_model,
        use_mock_orb=args.mock_orb,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_force_residual=not args.no_force_residual,
        enforce_zero_force_condition=not args.no_zero_force_condition,
        enforce_zero_stress_condition=args.zero_stress_condition,
        gamma_min=args.gamma_min
    ).to(device)

    # Print parameter efficiency
    param_info = model.count_parameters()
    print(f"Trainable parameters: {param_info['trainable']:,} ({param_info['trainable_pct']:.2f}%)")
    print(f"Frozen parameters:    {param_info['frozen']:,}")
    print(f"Total parameters:     {param_info['total']:,}")
    print("=" * 60)

    # Only pass trainable adapter parameters to the optimizer
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.6, patience=15, min_lr=1e-4
    )

    if args.wandb:
        import wandb
        wandb.init(project="diffcsp_orb_adapter", config=vars(args))

    # Load datasets if present
    if Path(args.train_csv).exists():
        trainset = CrystDataset(args.train_csv, 'train_sym')
        train_loader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size)
    else:
        print(f"Notice: {args.train_csv} not found. Running with synthetic test loader...")
        train_loader = None

    if train_loader is not None:
        for epoch in trange(args.epochs, desc="Epochs"):
            model.train(True)
            train_losses = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                batch = batch.to(device)
                loss = model.training_step(batch, 0)
                if loss is None:
                    continue

                train_losses.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_value_(model.get_trainable_parameters(), 0.4)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            avg_loss = float(np.mean(train_losses))
            scheduler.step(avg_loss)
            print(f"Epoch {epoch}: Average Train Loss = {avg_loss:.4f}")

            if (epoch + 1) % 10 == 0:
                # Save trainable adapter weights
                torch.save(model.decoder.state_dict(), args.ckpt_path)
                print(f"Saved checkpoint to {args.ckpt_path}")

            if args.wandb:
                wandb.log({'train_loss': avg_loss, 'lr': optimizer.param_groups[0]['lr']})
    else:
        print("Ready for training on cluster or machine with dataset files.")


if __name__ == '__main__':
    main()
