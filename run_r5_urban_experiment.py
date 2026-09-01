"""Reproducible R5 urban experiment: train/validation split, baseline, alpha curve.

By default this uses the project's analytical urban-like generator. Supplying
capture paths instead uses the `.npz` contract produced by Isaac Lab. Either
mode is diagnostic evidence, not paper-level reproduction by itself.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from r5_terrain_dataset import (
    load_isaaclab_depth_sample,
    make_randomized_urban_samples,
    prepare_sparse_dataset,
)
from r5_training import train_four_level_sparse_model
from r5_sparse_evaluation import (
    current_frame_merge_baseline,
    current_frame_merge_baseline_with_offsets,
    current_frame_prediction_coordinates,
    current_frame_prediction_with_offsets,
    height_metrics,
    occupancy_metrics,
)
from r5_sparse_model import EmptyPruningError


# alpha=0 is a diagnostic "do not prune by likelihood" setting. The remaining
# values are the actual threshold sweep; no paper claim is based on alpha=0.
ALPHAS = (0.0, .05, .1, .2, .3, .4, .5)


def evaluate_split(model, dataset, voxel_size: float, label: str) -> dict:
    """Measure one fixed data split against the same explicit merge baseline."""
    baseline_coordinates = current_frame_merge_baseline(dataset.input_batch)
    baseline_with_offsets = current_frame_merge_baseline_with_offsets(dataset.input_batch)
    baseline_occupancy = occupancy_metrics(baseline_coordinates, dataset.target_batch)
    baseline_height = height_metrics(*baseline_with_offsets, dataset.target_batch, voxel_size=voxel_size)
    print(f'{label} baseline occupancy:', baseline_occupancy)
    print(f'{label} baseline height:', baseline_height)
    report = {'baseline': {'occupancy': asdict(baseline_occupancy), 'height': asdict(baseline_height)}, 'alphas': {}}
    for alpha in ALPHAS:
        try:
            prediction = model(dataset.input_tensor, alpha=alpha)
            metrics = occupancy_metrics(
                current_frame_prediction_coordinates(
                    prediction.candidates.C, prediction.occupancy_logits.F, alpha,
                ),
                dataset.target_batch,
            )
            predicted_with_offsets = current_frame_prediction_with_offsets(
                prediction.candidates.C,
                prediction.occupancy_logits.F,
                prediction.position_offsets.F,
                alpha,
            )
            height = height_metrics(*predicted_with_offsets, dataset.target_batch, voxel_size=voxel_size)
            report['alphas'][str(alpha)] = {'occupancy': asdict(metrics), 'height': asdict(height)}
            print(f'{label} alpha={alpha} occupancy:', metrics)
            print(f'{label} alpha={alpha} height:', height)
        except EmptyPruningError:
            report['alphas'][str(alpha)] = {'empty_prediction': True}
            print(f'{label} alpha={alpha}: empty prediction')
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--train-count', type=int, default=128)
    parser.add_argument('--val-count', type=int, default=32)
    parser.add_argument('--train-captures', nargs='+', help='Optional Isaac Lab .npz captures for training.')
    parser.add_argument('--val-captures', nargs='+', help='Optional Isaac Lab .npz captures for validation.')
    parser.add_argument('--voxel-size', type=float, default=.5)
    parser.add_argument('--grid-size', type=int, default=8)
    parser.add_argument('--results', type=Path, help='Optional JSON file for config and measured metrics.')
    parser.add_argument('--checkpoint', type=Path, help='Optional PyTorch state_dict destination.')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if bool(args.train_captures) != bool(args.val_captures):
        raise ValueError('--train-captures and --val-captures must be supplied together')
    if args.voxel_size <= 0 or args.grid_size <= 0:
        raise ValueError('--voxel-size and --grid-size must be positive')
    if args.train_captures:
        train_samples = [load_isaaclab_depth_sample(path) for path in args.train_captures]
        val_samples = [load_isaaclab_depth_sample(path) for path in args.val_captures]
        source = 'isaaclab_depth_capture'
    else:
        train_samples = make_randomized_urban_samples(args.train_count, 1)
        val_samples = make_randomized_urban_samples(args.val_count, 2)
        source = 'analytical_urban_generator'
    train = prepare_sparse_dataset(
        train_samples,
        voxel_size=args.voxel_size,
        grid_size=args.grid_size,
        augmentation_seed=11,
        device=args.device,
    )
    val = prepare_sparse_dataset(
        val_samples,
        voxel_size=args.voxel_size,
        grid_size=args.grid_size,
        augmentation_seed=12,
        device=args.device,
    )
    run = train_four_level_sparse_model(train, steps=args.steps, seed=0)
    print(f'train_loss: {run.initial_loss:.6f} -> {run.final_loss:.6f}')
    results = {
        'config': vars(args) | {
            'results': str(args.results) if args.results else None,
            'checkpoint': str(args.checkpoint) if args.checkpoint else None,
            'source': source,
        },
        'training': {
            'initial_loss': run.initial_loss,
            'final_loss': run.final_loss,
            'learning_rate_history': list(run.learning_rate_history),
        },
        'train_evaluation': {},
        'validation': {},
    }
    results['train_evaluation'] = evaluate_split(run.model, train, args.voxel_size, 'train')
    results['validation'] = evaluate_split(run.model, val, args.voxel_size, 'validation')
    # Keep these aliases while earlier result readers migrate to split-aware output.
    results['baseline'] = results['validation']['baseline']
    results['alphas'] = results['validation']['alphas']
    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f'saved results: {args.results}')
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run.model.state_dict(), args.checkpoint)
        print(f'saved checkpoint: {args.checkpoint}')

if __name__ == '__main__':
    main()
