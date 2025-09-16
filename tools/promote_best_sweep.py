#!/usr/bin/env python3
"""
Promote best sweep config to a longer retrain run.

Finds the best run in a W&B sweep by a given metric, then relaunches the
training script with the same hyperparameters but overridden settings like
more epochs and richer visualisations. Works for both VAE and AE residual flows.

Examples:
  # Promote best residual AE run (objective_score) to 120 epochs with PCA
  .venv/Scripts/python.exe tools/promote_best_sweep.py \
      --entity <your_entity> --project nn-batch-correction \
      --sweep-id <sweep_hash> --metric objective_score --goal maximize \
      --program NN_batch_correct.py --epochs 120 --log_latent_every 0 \
      --out_corrected artifacts/outputs/promoted_corrected.csv --out_latent artifacts/outputs/promoted_latent.csv \
      --viz_pca_before artifacts/outputs/promoted_pca_before.png --viz_pca_after artifacts/outputs/promoted_pca_after.png --viz_pca_panel artifacts/outputs/promoted_pca_panel.png

  # Promote best VAE run (cond_minus_batch) to 150 epochs with viz
  .venv/Scripts/python.exe tools/promote_best_sweep.py \
      --entity <your_entity> --project nn-batch-correction \
      --sweep-id <sweep_hash> --metric cond_minus_batch --goal maximize \
      --program VAEModel/vaemodeltest.py --epochs 150 --generate_viz \
      --out_corrected VAEModel/promoted_corrected.csv --out_latent VAEModel/promoted_latent.csv \
      --viz_pca_before VAEModel/promoted_pca_before.png --viz_pca_after VAEModel/promoted_pca_after.png --viz_pca_panel VAEModel/promoted_pca_panel.png
"""

import argparse
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path


def get_best_run(entity: str, project: str, sweep_id: str, metric: str, goal: str):
    import wandb
    api = wandb.Api()
    sid = sweep_id if '/' in sweep_id else f"{entity}/{project}/{sweep_id}"
    sweep = api.sweep(sid)
    runs = sweep.runs
    if not runs:
        raise SystemExit(f"No runs found for sweep: {sid}")
    def get_val(r):
        # Prefer summary over history for final metric
        v = r.summary.get(metric)
        if v is None:
            # last resort: run.summary is missing, try config (unlikely)
            v = float('-inf') if goal == 'maximize' else float('inf')
        return v
    reverse = (goal == 'maximize')
    best = sorted(runs, key=get_val, reverse=reverse)[0]
    return best


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return None


def build_cmd(py: str, program: str, cfg: dict, overrides: dict, exclude_keys: set | None = None):
    # Boolean flags to emit as presence (store_true)
    bool_flags = {
        'amp', 'pin_memory', 'cudnn_benchmark', 'fused_optim', 'compile',
        'generate_viz', 'genes_in_rows', 'use_wandb', 'use_residual',
        'use_nb_loss'
    }
    # Keys known to be strings but should pass as plain strings
    passthrough_keys = {
        'counts','metadata','sample_col','batch_col','label_col','out_corrected','out_latent',
        'viz_pca_before','viz_pca_after','viz_pca_panel','viz_boxplot','cache_dir','precision',
        'enc_hidden','dec_hidden','adv_hidden','sup_hidden','scheduler'
    }
    # Core numeric keys
    numeric_keys = {
        'epochs','patience','hvg','latent_dim','adv_weight','sup_weight','kl_weight','dropout','lr',
        'batch_size','num_workers','prefetch_factor','warmup_ratio','grad_accum','log_latent_every','seed',
        'weight_decay'
    }

    args = [py, program]
    # Ensure W&B logging enabled by default
    cfg = dict(cfg)
    cfg.setdefault('use_wandb', True)
    # Apply overrides
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    # Emit flags in stable order
    exclude_keys = exclude_keys or set()
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if v is None:
            continue
        if k in exclude_keys:
            continue
        if k in bool_flags:
            bv = _to_bool(v)
            if bv is True:
                args.append(f"--{k}")
            # if False or unknown, omit flag entirely
            continue
        # Everything else is key value
        args.append(f"--{k}")
        args.append(str(v))

    return args


def main():
    ap = argparse.ArgumentParser(description="Promote best W&B sweep run to longer retrain")
    ap.add_argument('--entity', required=True)
    ap.add_argument('--project', required=True)
    ap.add_argument('--sweep-id', required=True, help='Short or full sweep ID')
    ap.add_argument('--metric', required=True, help='Metric name to select best run')
    ap.add_argument('--goal', choices=['maximize','minimize'], default='maximize')
    ap.add_argument('--program', required=True, help='Training entrypoint to relaunch (e.g., VAEModel/vaemodeltest.py)')
    ap.add_argument('--python', default=sys.executable, help='Python interpreter path')
    # Common overrides
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--log_latent_every', type=int, default=None)
    ap.add_argument('--generate_viz', action='store_true')
    ap.add_argument('--out_corrected', type=Path, default=None)
    ap.add_argument('--out_latent', type=Path, default=None)
    ap.add_argument('--viz_pca_before', type=Path, default=None)
    ap.add_argument('--viz_pca_after', type=Path, default=None)
    ap.add_argument('--viz_pca_panel', type=Path, default=None)
    ap.add_argument('--viz_boxplot', type=Path, default=None)
    args = ap.parse_args()

    best = get_best_run(args.entity, args.project, args.sweep_id, args.metric, args.goal)
    cfg = {k: v for k, v in best.config.items() if not k.startswith('_')}

    overrides = {
        'epochs': args.epochs,
        'log_latent_every': args.log_latent_every,
        'generate_viz': args.generate_viz,
        'out_corrected': str(args.out_corrected) if args.out_corrected else None,
        'out_latent': str(args.out_latent) if args.out_latent else None,
        'viz_pca_before': str(args.viz_pca_before) if args.viz_pca_before else None,
        'viz_pca_after': str(args.viz_pca_after) if args.viz_pca_after else None,
        'viz_pca_panel': str(args.viz_pca_panel) if args.viz_pca_panel else None,
        'viz_boxplot': str(args.viz_boxplot) if args.viz_boxplot else None,
    }

    # Choose program; auto-wrap AE residual script to also generate PCA
    program_to_run = args.program
    exclude: set[str] = set()
    if program_to_run.replace('\\', '/').endswith('NN_batch_correct.py'):
        # Use wrapper that runs training then PCA visualiser; drop flags the wrapper doesn't accept
        program_to_run = str(Path('tools') / 'sweep_nn_residual_entry.py')
        exclude.update({'generate_viz', 'viz_boxplot'})

    cmd = build_cmd(args.python, program_to_run, cfg, overrides, exclude_keys=exclude)
    print('[INFO] Best run:', best.id, best.name)
    print('[INFO] Metric', args.metric, '=', best.summary.get(args.metric))
    print('[RUN]', ' '.join(shlex.quote(x) for x in cmd))
    # Execute
    rc = subprocess.call(cmd)
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
