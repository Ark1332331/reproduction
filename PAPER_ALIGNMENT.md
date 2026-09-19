# Paper-to-code alignment

This is the canonical comparison for the current workspace. “Paper” means the
NSR PDF included beside this file (accepted 2022 version). “Assumption” means
the paper does not publish enough detail to recover the value; it must not be
reported as an author hyper-parameter.

## Method contract

| Paper statement | Code path | Status |
|---|---|---|
| Current and previous output are transformed into the current frame and concatenated temporally | `r7_autoregressive_rollout.py`, `r1_data_representation.py` | Aligned; previous input is the detached model estimate, labelled `k=1` |
| 64×64×64 grid representing 3.2×3.2×3.2 m | `paper_config.py`, `r1_data_representation.py` | Aligned default |
| Voxel feature is centroid offset from the voxel corner | `r1_data_representation.py` | Aligned |
| Four spatial downsampling convolutions, temporal dimension preserved | `r5_sparse_model.py` | Structural match; exact kernels/channels are not published |
| U-Net skip connections and sparse generative decoder upsampling | `r5_sparse_model.py` | Structural match; exact block layout is an assumption |
| Likelihood-based pruning at decoder stages with α=0.5 | `r5_sparse_model.py` | Mechanism and canonical threshold present; target-guard behavior is an implementation addition |
| Final output is a 3D sub-voxel point estimate | `r5_sparse_model.py`, `r7_autoregressive_rollout.py` | Structural match; final candidates are restricted to `k=0` as an explicit contract inference |
| Occupancy BCE plus mean Euclidean sub-voxel position loss | `r5_sparse_loss.py` | Implemented; exact relative weighting is not stated |
| 12-step rollout and no gradient propagation through time | `r7_autoregressive_rollout.py` | Aligned |
| Adam, 0.01 initial LR, exponential decay to 0.0001 | `paper_config.py`, `run_r7_paper_train.py` | Aligned default |
| Position, tilt, height-patch, pruning, outlier, pose noise and x/y mirroring | `r7_data_augmentation.py`, `r7_measurement_augmentation.py` | Implemented for canonical R7 training; exact patch sampling is an assumption |
| Four front/back/left/right depth cameras tilted down 30° | `collect_isaaclab_anymal_vectorized.py` | Direction/count/tilt intent aligned; mount, resolution and point cap are assumptions |

## Data generation

| Paper statement | Current implementation | Consequence |
|---|---|---|
| IsaacGym randomized stairs, boxes, walls, roadblocks/structured obstacles and corridors | IsaacLab ANYmal-C collector with five named terrain types | Simulator and locomotion stack differ; this is not an exact data-source reproduction. New vectorized captures default to randomized command and initial yaw; use `--no-random-motion` only for a diagnostic/control run |
| Stairs width [0.2, 0.5] m and rise [0.08, 0.25] m | `paper_terrains.py` now uses these ranges | Existing NPZ files collected before this change may use the older safe policy range and must be treated as historical data until regenerated |
| Boxes width/length [0.2, 2.0] m and height [0.08, 0.25] m | `paper_terrains.py` | Closest documented match; number/layout of boxes is an assumption |
| Walls sampled to produce corridors of width [2, 6] m | `paper_terrains.py` | Corridor width range is now unclipped; wall dimensions/height/layout are assumptions |
| Pole dimensions and full scene sampling procedure | `paper_terrains.py`, collector | Not specified in the paper; current pole geometry is an explicit assumption |
| More than 200,000 time-step observations, average 43% visible | Captured IsaacLab data and manifests under `data/` | Current data volume/distribution must be read from the manifests; it is not evidence of 200k paper-scale data. Newly collected files include code/checkpoint/environment provenance; old NPZ files do not |

## Evaluation contract

The paper reports mean precision, recall, F1 on a 64³ robot-centric voxel grid
and mean absolute height difference. The evaluator now exposes both
per-frame macro means (the primary paper-style report) and global count-based
micro aggregates. Height error remains a *paper-inspired diagnostic*: it uses
the highest z value per XY cell on shared XY cells, and reports matched-cell
coverage explicitly because the paper does not define the exact height
matching/reduction rule. Do not compare it to Table I without stating this
limitation.

The current-measurement merge baseline is an engineering baseline for the
captured data. The paper's reported baseline numbers come from robot
experiments and are not reproduced merely by running this evaluator.

## Canonical versus diagnostic settings

The following are intentionally exposed as diagnostics and must be recorded in
the checkpoint/result JSON when used:

- `--disable-data-augmentation`;
- `--pruning-alpha` values other than 0.5;
- `--feedback-alpha` values other than the pruning alpha;
- external `--likelihood-logit-offset` calibration;
- positive-class weights, warmup, altered channels or generative kernels;
- target guard (`--target-guard`) and no-history ablations. The canonical
  training default is now strict-paper behavior with target guard disabled;
  target guard is an explicitly labelled engineering safeguard.

The existing `data/` and `results/` artifacts contain many such ablations. They
are useful evidence for debugging but are not interchangeable with the
canonical paper-aligned result.

## Reproduction status

The current project is **partially complete**: representation, sparse-network
contracts, collection contracts and several diagnostics exist; the workspace
does not yet contain a stable paper-scale training/evaluation result. See
`AGENTS.md` for the dated evidence, failed gates, checkpoints and next action.
