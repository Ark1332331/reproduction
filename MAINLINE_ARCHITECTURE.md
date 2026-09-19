# Mainline reproduction architecture

This is the maintained implementation path. The repository intentionally does
not retain every historical experiment script; old numerical evidence remains
in local `data/` and `results/` files when available.

## Execution flow

```text
paper_config.py
       │
       ├── IsaacLab collection (isaaclab environment)
       │     paper_terrains.py
       │     collect_isaaclab_anymal_vectorized.py
       │     collect_r7_vectorized_dataset.py / collect_r7_quota.py
       │     freeze_r7_split_manifest.py
       │
       └── model path (nsr-me-cu130-t291 environment)
             r1_data_representation.py
             r5_sparse_input.py
             r5_sparse_model.py
             r5_sparse_loss.py
             r7_autoregressive_rollout.py
             r7_data_augmentation.py
                    │
                    ├── run_r7_paper_train.py
                    └── run_r7_paper_eval.py
```

## Source file responsibilities

| Layer | Files | Responsibility |
|---|---|---|
| Contract/config | `paper_config.py` | Paper-stated geometry, rollout, pruning and optimizer constants; explicit implementation assumptions |
| Terrain/data source | `paper_terrains.py` | Five structured terrain generators and their recorded primitive geometry |
| IsaacLab capture | `collect_isaaclab_anymal_vectorized.py`, `collect_isaaclab_anymal_trajectory.py` | Four-camera ANYmal simulation capture; vectorized path is primary, serial path is fallback |
| Capture orchestration | `collect_r7_dataset.py`, `collect_r7_vectorized_dataset.py`, `collect_r7_quota.py` | Seed allocation, retries, timeout/process cleanup, manifests and quota collection |
| Split control | `freeze_r7_split_manifest.py` | Freeze explicit train/validation trajectory membership and reject scene-seed overlap |
| Capture identity | `r7_vectorized_capture_contract.py`, `r7_capture_provenance.py` | Stable filenames, environment identity, code/checkpoint/environment metadata |
| Representation | `r1_data_representation.py` | Robot-centric 3D voxelization and centroid offsets |
| Sparse tensor bridge | `r5_sparse_input.py` | Conversion from NumPy voxel batches to MinkowskiEngine tensors |
| Network | `r5_sparse_model.py` | Four-level 4D sparse U-Net-like completion model, generative decoder, pruning and k=0 output |
| Loss | `r5_sparse_loss.py` | Final occupancy/offset loss, multiscale likelihood targets and BCE |
| Training augmentation | `r7_measurement_augmentation.py`, `r7_data_augmentation.py` | Measurement corruption and trajectory mirroring; targets remain clean |
| Autoregressive method | `r7_autoregressive_rollout.py` | Detached previous prediction feedback, 12-step rollout, training and evaluation loops |
| Metrics | `r5_sparse_evaluation.py` | Occupancy P/R/F1, macro/micro aggregation and height error with coverage |
| Entrypoints | `run_r7_paper_train.py`, `run_r7_paper_eval.py` | Canonical checkpointed training and validation |

## Environment split

- `isaaclab`: IsaacLab/Isaac Sim, ANYmal policy, cameras and terrain capture.
- `nsr-me-cu130-t291`: PyTorch + MinkowskiEngine, model training/evaluation and
  sparse-network tests.

The two environments are intentionally separate because the IsaacLab runtime
and the MinkowskiEngine build use incompatible Python/package stacks in this
workspace.

## What is not part of the mainline

Toy R2/R3/R3b pipelines, the R6 controller prototype, urban-depth experiments,
one-off pruning/feature/overfit probes, old non-checkpointed wrappers and GPU
wait scripts were removed from the source tree. Their generated evidence is
not silently reinterpreted as a paper result.
