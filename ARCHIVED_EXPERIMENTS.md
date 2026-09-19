# Removed experiment code

The source tree now contains only the maintained R1→R5→R7 reproduction path.
The following categories were removed during the cleanup because they are not
required to run the current reproduction:

- R2/R3/R3b toy completion and grid baselines;
- R5 urban-depth and standalone R5 training experiments;
- R6 controller/height-scan and policy-play prototypes;
- historical R7 non-checkpointed wrappers;
- one-off R7 pruning, feature, target, seed, temporal and overfit probes;
- old GPU wait/scale launch scripts and legacy depth collectors;
- generated Python bytecode and local runtime logs.

Numeric evidence produced by those experiments remains local under `data/` and
`results/` when present, but those paths are ignored by Git and are not part of
the maintained source API.
