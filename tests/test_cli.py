"""Tests for CLI arguments and entrypoint parsing."""

import subprocess
import sys


def test_train_cli_help():
    result = subprocess.run([sys.executable, "train.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "DiffCSP++ Training CLI" in result.stdout

    res_module = subprocess.run([sys.executable, "-m", "diffcsp.cli.train", "--help"], capture_output=True, text=True)
    assert res_module.returncode == 0
    assert "DiffCSP++ Training CLI" in res_module.stdout


def test_inference_cli_help():
    result = subprocess.run([sys.executable, "inference.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "DiffCSP++ Inference CLI" in result.stdout

    res_module = subprocess.run([sys.executable, "-m", "diffcsp.cli.inference", "--help"], capture_output=True, text=True)
    assert res_module.returncode == 0
    assert "DiffCSP++ Inference CLI" in res_module.stdout
