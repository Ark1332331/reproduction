# NSR reproduction workspace

This directory contains a staged, testable reproduction of *Neural Scene
Representation for Locomotion on Structured Terrain*. The paper PDF is the
source of truth for method claims. The code does not contain the authors'
implementation; values not stated in the paper are labelled as assumptions.

## Start here

1. Read [`PAPER_ALIGNMENT.md`](PAPER_ALIGNMENT.md) for the paper-to-code
   contract and known deviations.
2. Read [`AGENTS.md`](AGENTS.md) for the chronological experiment record and
   the current reproduction status.
3. Use `run_r7_paper_train.py` and `run_r7_paper_eval.py` as the canonical
   training/evaluation entrypoints.
4. Read [`MAINLINE_ARCHITECTURE.md`](MAINLINE_ARCHITECTURE.md) for the exact
   source-file map and execution order.

The canonical method defaults are centralized in [`paper_config.py`](paper_config.py):
64³ voxels over 3.2 m³, 0.05 m cells, 12-step temporal rollout, alpha 0.5,
and Adam learning rate 0.01 exponentially decayed to 0.0001. Canonical
training also defaults to strict-paper candidate generation (`target_guard` is
off); enable `--target-guard` only for a recorded engineering comparison.

## Logical layout

| Area | Role |
|---|---|
| `paper_config.py` | Constants explicitly stated by the paper plus clearly named implementation defaults |
| `r1_*` | Pose alignment and voxel/centroid representation |
| `r5_*` | Minkowski sparse input, four-level network, losses, augmentation, metrics |
| `r7_*` | Detached autoregressive rollout and IsaacLab trajectory contract |
| `collect_*.py`, `paper_terrains.py` | Simulation data collection and terrain generation |
| `r7_capture_provenance.py` | Code, checkpoint, package and paper-contract provenance for new captures |
| `run_r7_paper_train.py`, `run_r7_paper_eval.py` | Canonical training and evaluation |
| `collect_r7_*.py` | Recoverable collection orchestration |
| `test_*.py` | Unit and mechanism-contract tests |
| `data/` | Captured trajectories, manifests, logs, and diagnostic JSON/PT files |
| `results/` | Checkpoints and selected result artifacts |

Generated data and checkpoints are evidence, not source code. They remain on
the local machine but are ignored by Git so the source repository stays small.

Newly captured NPZ files carry `capture_schema_version` and a nested
`provenance` record. Do not mix them with historical NPZ files without checking
the terrain profile, motion randomization, and provenance fields.

## Environment

The default system Python in this workspace does not provide `numpy`, `torch`,
or `MinkowskiEngine`. Run tests and training in the project environment that
provides those packages (the existing scripts refer to the IsaacLab and
MinkowskiEngine environments in their module docstrings).

Example test command from this directory:

```bash
python -m unittest discover -s . -p 'test_*.py'
```

This command must be run in the configured environment; a missing dependency is
an environment failure, not a model result.
