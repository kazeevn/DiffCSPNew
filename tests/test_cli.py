"""Tests for CLI arguments and entrypoint parsing."""

import subprocess
import sys


def test_train_cli_help():
    res_module = subprocess.run([sys.executable, "-m", "diffcsp.cli.train", "--help"], capture_output=True, text=True)
    assert res_module.returncode == 0
    assert "DiffCSP++ Training CLI" in res_module.stdout

    res_script = subprocess.run(["diffcsp-train", "--help"], capture_output=True, text=True)
    assert res_script.returncode == 0
    assert "DiffCSP++ Training CLI" in res_script.stdout


def test_inference_cli_help():
    res_module = subprocess.run([sys.executable, "-m", "diffcsp.cli.inference", "--help"], capture_output=True, text=True)
    assert res_module.returncode == 0
    assert "DiffCSP++ Inference CLI" in res_module.stdout

    res_script = subprocess.run(["diffcsp-inference", "--help"], capture_output=True, text=True)
    assert res_script.returncode == 0
    assert "DiffCSP++ Inference CLI" in res_script.stdout
