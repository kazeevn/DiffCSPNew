# Training stability: three proposed fixes

Status: **proposed, not applied.** All three touch `diffcsp/cli/train.py`, which is
shared by the CSPNet and ORB paths, so applying them changes how every future run
optimizes. They are written up here rather than committed so the change can be made
deliberately, and so the checkpoints trained under the current settings stay
interpretable.

## What prompted this

Vanilla CSPNet (512/6, 12,277,248 params) on LeMat-Bulk fmax1, filtered to
`E_hull <= 0.1 eV` and `<= 128` atoms — 1,577,517 structures, batch 256, lr 1e-3,
20 epochs, 6,162 steps per epoch.
Run: `wandb.ai/symmetry-advantage/diffcsp/runs/mqk461a6`.

| epoch | train | val | | epoch | train | val |
|---|---|---|---|---|---|---|
| 0 | 0.4411 | | | 10 | 0.6850 | |
| 1 | 0.3631 | 0.3346 | | 11 | 0.7326 | 0.3161 |
| 2 | 0.3440 | | | 12 | 0.4769 | |
| 3 | 0.3314 | 0.3129 | | 13 | 0.4336 | 0.3085 |
| 4 | **0.3204** | | | 14 | 0.3233 | |
| 5 | 0.4230 | 0.3637 | | 15 | 0.3692 | 0.2992 |
| 6 | 0.8755 | | | 16 | 0.3126 | |
| 7 | **1.5624** | 0.3441 | | 17 | 0.3184 | 0.2944 |
| 8 | 0.3693 | | | 18 | 0.8123 | |
| 9 | 1.1485 | 0.3551 | | 19 | 0.3347 | **0.2902** |

Train loss reached 0.3204 by epoch 4, then spiked to 1.5624 and kept oscillating to
the end; the final epoch (0.3347) is worse than epoch 4. No NaNs, no OOM, and the
learning rate never left 1.00e-03.

Validation nonetheless improved steadily through the second half and was **still
falling at the last measurement** (0.2902, epoch 20). So the instability cost
training efficiency rather than preventing learning, and the epoch budget — not
convergence — ended the run.

That divergence between a chaotic train curve and a smooth val curve is the useful
clue: the spikes are a handful of pathological *batches* per epoch, not a
systematically wrong learning rate. Each fix below follows from that.

None of this appeared on MP-20, where 1000 short epochs of 212 steps gave the
scheduler room to act and the smaller cells produced smaller gradients.

---

## Fix 1 — clip gradients by norm, not by value

`diffcsp/cli/train.py:432`

```python
torch.nn.utils.clip_grad_value_(params_to_train, 0.4)          # current
torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=0.5)  # proposed
```

`clip_grad_value_` bounds each element independently. For a 12.3M-parameter model
every element can sit at the 0.4 limit and the total gradient norm still be
enormous, so one bad batch moves the weights arbitrarily far. Norm clipping bounds
the step itself, which is what the spikes call for.

This is the highest-value of the three: it targets the actual failure mode, and it
is the one change that would have prevented the epoch-7 blowup.

Worth knowing which batches are pathological. Cost per batch scales with the number
of edges, `sum(n_i^2)` over the cell sizes in the batch, and that distribution is
heavy-tailed on this dataset: cells above 32 atoms are 14.8% of structures but
73.7% of all edges. A batch drawn from the upper tail produces a far larger
gradient than a typical one.

## Fix 2 — step the scheduler on validation loss

`diffcsp/cli/train.py:444`

```python
scheduler.step(avg_loss)                                  # current: train loss
scheduler.step(current_val_loss if current_val_loss is not None else avg_loss)
```

`avg_loss` is the train loss — precisely the spiking signal. Feeding it to
`ReduceLROnPlateau` means a single noisy epoch can reset patience and mask a real
plateau. Validation is measured on a fixed set every `eval_freq` epochs and is far
smoother here; fall back to train loss only on epochs where no evaluation ran.

Note `current_val_loss` already exists at that point in the loop, computed for the
checkpoint-saving logic.

## Fix 3 — make the schedule reachable within the epoch budget

`diffcsp/cli/train.py:222`

```python
ReduceLROnPlateau(optimizer, factor=0.6, patience=15, min_lr=1e-4)   # current
```

With `patience=15` on a 20-epoch run the scheduler cannot fire even once, and it
did not: the LR was 1.00e-03 for all 20 epochs. The setting is inherited from MP-20,
where epochs are 212 steps; here an epoch is 6,162 steps, so patience measured in
epochs means something ~30x larger.

Either scale patience to the epoch budget (`patience=3` at 20-30 epochs), or drop
the initial LR to 3e-4, or add a few hundred warmup steps and keep 1e-3. Warmup is
the most targeted if Fix 1 is also applied, since the early epochs were stable and
the blowup began at epoch 5.

---

## Verifying a rerun

Compare against the run above:

- **train loss monotone after epoch ~3**, with no epoch exceeding roughly 1.3x its
  predecessor. That is the direct test of Fix 1.
- **LR below 1e-3 by the end**, showing the scheduler actually engaged (Fixes 2, 3).
- **val below 0.2902**, the current best, at equal or fewer epochs.

Since validation was still descending at epoch 20, raise the budget to ~30 epochs
in the same rerun; at ~52 min/epoch that is roughly 26 hours. The preprocessed cache
is warm, so a rerun reaches the training loop in about two minutes.

## What is not proposed

The instability is **not** an argument for a smaller batch or a smaller model. Batch
256 was chosen from measured peak memory (19.11 GiB of 44.4) and ran 20 epochs
without a single OOM. Shrinking it would reduce the per-batch edge count and so damp
the spikes, but it treats the symptom and costs throughput; norm clipping addresses
the cause directly.
