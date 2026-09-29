"""Packed datasets, edge-budget batching, and property conditioning with classifier-free guidance."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

from diffcsp.data.dataset import graph_arrays_to_pyg_data
from diffcsp.data.graph import process_one
from diffcsp.data.packed import (
    EdgeBudgetBatchSampler,
    PackedCrystDataset,
    concatenate_packed,
    pack_records,
    write_packed,
)
from diffcsp.models.geo_v2_cspnet import GeoV2CSPNet, PropertyEmbedding
from diffcsp.models.geo_v2_diffusion import GeoV2Diffusion

MP20_TEST = Path("data/mp-20/test.csv")
PROPS = ["energy_above_hull", "formation_energy_per_atom"]


@pytest.fixture(scope="module")
def mp20_rows():
    if not MP20_TEST.exists():
        pytest.skip(f"{MP20_TEST} not found")
    df = pd.read_csv(MP20_TEST, nrows=12)
    return df.rename(columns={"e_above_hull": "energy_above_hull", "material_id": "immutable_id"})


@pytest.fixture(scope="module")
def records(mp20_rows):
    out = []
    for _, row in mp20_rows.iterrows():
        rec = process_one({"cif": row["cif"], "immutable_id": row["immutable_id"]}, graph_method="none")
        assert rec is not None
        out.append({"id": row["immutable_id"], "graph_arrays": rec["graph_arrays"], "props": {p: row[p] for p in PROPS}})
    return out


@pytest.fixture(scope="module")
def packed_root(records, tmp_path_factory):
    root = tmp_path_factory.mktemp("packed")
    # Two chunks, as scripts/pack_dataset.py writes them, to exercise the offset rebase.
    arrays = concatenate_packed([pack_records(records[:5], PROPS), pack_records(records[5:], PROPS)])
    write_packed(root / "train", arrays, {"source": "test"})
    write_packed(root / "val", pack_records(records[:6], PROPS), {"source": "test"})
    return root


def test_packed_roundtrip_matches_cryst_dataset(records, packed_root):
    ds = PackedCrystDataset(packed_root / "train", cond_props=PROPS)
    assert len(ds) == len(records)
    for i, rec in enumerate(records):
        ref = graph_arrays_to_pyg_data(rec)
        got = ds[i]
        for key in ("frac_coords", "atom_types", "lengths", "angles", "ops", "ops_inv", "anchor_index", "spacegroup"):
            torch.testing.assert_close(got[key], ref[key], msg=key)
        assert got.num_atoms == ref.num_atoms
        torch.testing.assert_close(got.props, torch.tensor([[rec["props"][p] for p in PROPS]], dtype=torch.float32))


def test_packed_filters(records, packed_root):
    e_hull = np.array([r["props"]["energy_above_hull"] for r in records])
    n_atoms = np.array([r["graph_arrays"][6] for r in records])
    ds = PackedCrystDataset(packed_root / "train", max_e_hull=0.05, max_atoms=int(np.median(n_atoms)))
    expected = np.flatnonzero((e_hull <= 0.05) & (n_atoms <= int(np.median(n_atoms))))
    np.testing.assert_array_equal(ds.indices, expected)
    sub = PackedCrystDataset(packed_root / "train", max_samples=4, subset_seed=3)
    assert len(sub) == 4
    np.testing.assert_array_equal(sub.indices, PackedCrystDataset(packed_root / "train", max_samples=4, subset_seed=3).indices)


def test_edge_budget_sampler_partitions_ranks():
    sizes = np.random.default_rng(0).integers(1, 40, size=1000)
    budget = 5000
    per_rank = []
    for rank in range(4):
        s = EdgeBudgetBatchSampler(sizes, max_edges=budget, seed=5, rank=rank, world_size=4)
        s.set_epoch(3)
        per_rank.append(list(s))
    assert len({len(b) for b in per_rank}) == 1, "every rank must run the same number of steps"
    seen = [i for batches in per_rank for b in batches for i in b]
    assert len(seen) == len(set(seen)), "ranks must not share structures"
    assert all(int((sizes[b] ** 2).sum()) <= budget for batches in per_rank for b in batches)
    # Deterministic per (seed, epoch), different across epochs.
    s = EdgeBudgetBatchSampler(sizes, max_edges=budget, seed=5, rank=1, world_size=4)
    s.set_epoch(3)
    assert list(s) == per_rank[1]
    s.set_epoch(4)
    assert list(s) != per_rank[1]
    with pytest.raises(ValueError):
        EdgeBudgetBatchSampler(np.array([100]), max_edges=5000)


def test_property_embedding_starts_as_null():
    emb = PropertyEmbedding(PROPS, out_dim=16)
    values = torch.tensor([[0.0, -1.0], [0.3, 2.0]])
    on = torch.ones(2, dtype=torch.bool)
    assert torch.all(emb(values, on) == 0) and torch.all(emb(values, ~on) == 0)
    torch.nn.init.normal_(emb.mlps[0][-1].weight)
    torch.nn.init.normal_(emb.null)
    out_on, out_off = emb(values, on), emb(values, ~on)
    assert not torch.allclose(out_on[0], out_on[1])  # depends on the value
    torch.testing.assert_close(out_off[0], out_off[1])  # null ignores it
    torch.testing.assert_close(out_off[0], emb.null.sum(0))


def _tiny_model(cond_props, drop=0.0):
    decoder = GeoV2CSPNet(hidden_dim=512, num_layers=1, edge_dim=32, cond_props=cond_props)
    model = GeoV2Diffusion(device="cpu", decoder=decoder, timesteps=6, cond_drop_prob=drop)
    model.init_ema()
    return model


def _batch(packed_root, cond_props):
    ds = PackedCrystDataset(packed_root / "train", cond_props=cond_props)
    return Batch.from_data_list([ds[i] for i in range(4)])


def test_conditional_forward_and_gradients(packed_root):
    torch.manual_seed(0)
    model = _tiny_model(["energy_above_hull"], drop=0.5)
    batch = _batch(packed_root, ["energy_above_hull"])
    model.train()
    out = model(batch)
    out["loss"].backward()
    emb = model.decoder.property_embedding
    assert emb.mlps[0][-1].weight.grad is not None and emb.mlps[0][-1].weight.grad.abs().sum() > 0
    # At init the conditional model is exactly the unconditional one.
    model.eval()
    torch.manual_seed(1)
    cond = model(batch)["loss"]
    batch.props = None
    torch.manual_seed(1)
    uncond = model(batch)["loss"]
    torch.testing.assert_close(cond, uncond)


def test_guided_sampling(packed_root):
    torch.manual_seed(0)
    model = _tiny_model(["formation_energy_per_atom"], drop=0.2)
    for mlp in model.decoder.property_embedding.mlps:
        torch.nn.init.normal_(mlp[-1].weight, std=0.5)
    model.eval()
    batch = _batch(packed_root, ["formation_energy_per_atom"])
    results = {}
    for w in (0.0, 1.0, 2.0):
        torch.manual_seed(7)
        final, _ = model.sample(batch, disable_progress=True, guidance_scale=w)
        results[w] = final["frac_coords"]
        assert torch.isfinite(final["lattices"]).all()
    batch_uncond = batch.clone()
    batch_uncond.props = None
    torch.manual_seed(7)
    uncond, _ = model.sample(batch_uncond, disable_progress=True)
    torch.testing.assert_close(results[0.0], uncond["frac_coords"])  # w = 0 is unconditional
    assert not torch.allclose(results[1.0], results[2.0])


def test_train_cli_on_packed_data_with_cfg(packed_root, tmp_path):
    ckpt = tmp_path / "m.pt"
    cmd = [
        sys.executable, "-m", "diffcsp.cli.train", "--model", "geov2", "--hidden_dim", "512", "--num_layers", "1",
        "--data_dir", str(packed_root), "--max_edges_per_batch", "3000", "--epochs", "2", "--eval_freq", "1",
        "--cond_props", "energy_above_hull", "--cond_drop_prob", "0.2", "--warmup_steps", "2",
        "--extra_val_max_e_hull", "0.1", "--num_workers", "0", "--device", "cpu", "--no-wandb",
        "--ckpt_path", str(ckpt), "--save_freq", "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "val_loss_uncond" in res.stdout and "val_loss_ehull0.1" in res.stdout
    data = torch.load(ckpt, weights_only=False)
    assert data["epoch"] == 2
    assert data["model_config"]["cond_props"] == ["energy_above_hull"]
    assert any(k.startswith("property_embedding.") for k in data["ema_state_dict"])
    # Resuming a finished run is a no-op that keeps the schedule state.
    res = subprocess.run(cmd + ["--resume", "auto"], capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
