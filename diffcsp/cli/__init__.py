"""Command-line interfaces for training and structure inference."""

from diffcsp.cli.inference import generate_structures
from diffcsp.cli.inference import main as inference_main
from diffcsp.cli.train import main as train_main
from diffcsp.cli.train import train

__all__ = ["train", "train_main", "generate_structures", "inference_main"]
