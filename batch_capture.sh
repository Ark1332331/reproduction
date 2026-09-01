#!/bin/bash
# Batch capture of paper-distribution ANYmal trajectories.
# Usage: bash batch_capture.sh <split> where split = train|validation (seeds below)
cd /home/ark/projects/IsaacLab
source /home/ark/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
export TERM=xterm CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

SCRIPT=/media/ark/Data/devpy/projects/allinone/reproduction/collect_isaaclab_anymal_trajectory.py
OUT=/media/ark/Data/devpy/projects/allinone/reproduction/data

if [ "$1" = "train" ]; then
  SEEDS="0 1 2 3"
elif [ "$1" = "validation" ]; then
  SEEDS="10 11 12 13"
else
  echo "usage: batch_capture.sh train|validation"; exit 1
fi

for terrain in stairs boxes walls poles corridors; do
  for seed in $SEEDS; do
    outfile=${OUT}/isaac_anymal_${terrain}_s${seed}.npz
    if [ -f "$outfile" ]; then
      echo "SKIP $outfile (exists)"; continue
    fi
    # wait until the GPU is free (a previous Isaac instance may still release memory)
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' MiB')" -gt 1500 ]; do
      echo "waiting for GPU to free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
      sleep 30
    done
    for attempt in 1 2; do
      echo "=== CAPTURE $terrain seed=$seed -> $outfile attempt=$attempt ($(date +%H:%M:%S))"
      timeout 600 ./isaaclab.sh -p "$SCRIPT" --headless --enable_cameras --terrain "$terrain" --seed "$seed" \
        --output "$outfile" --trajectory-steps 12 --frames-per-step 30 --settle-env-steps 30 2>&1 \
        | grep -E "trajectory\]|saved|WARN|Error|Traceback"
      echo "--- exit=$? file=$([ -f "$outfile" ] && echo OK || echo MISSING) ($(date +%H:%M:%S))"
      if [ -f "$outfile" ]; then
        break
      fi
      sleep 10
    done
    sleep 5
  done
done
echo "BATCH DONE $1"
