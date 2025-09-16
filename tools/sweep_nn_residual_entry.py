#!/usr/bin/env python3
"""
Sweep entrypoint for residual AE: runs NN_batch_correct.py then generates PCA plots.

This wrapper preserves all original CLI args to the training script and then
invokes visualise_outputs.py to create PCA plots using the provided counts,
metadata and the produced corrected matrix.
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import OUTPUTS_DIR, project_relative


def parse_known():
    ap = argparse.ArgumentParser(add_help=False)
    # We only explicitly read the fields needed for visualisation; everything else is forwarded verbatim
    ap.add_argument('--counts', type=Path, required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--sample_col', type=str, default='sample')
    ap.add_argument('--batch_col', type=str, default='batch')
    ap.add_argument('--label_col', type=str, default=None)
    ap.add_argument('--genes_in_rows', action='store_true')
    ap.add_argument('--out_corrected', type=Path, required=True)
    # Optional visual settings
    ap.add_argument('--viz_hvg_top', type=int, default=2000)
    ap.add_argument('--viz_pca_before', type=Path, default=OUTPUTS_DIR / 'pca_before.png')
    ap.add_argument('--viz_pca_after', type=Path, default=OUTPUTS_DIR / 'pca_after.png')
    ap.add_argument('--viz_pca_panel', type=Path, default=OUTPUTS_DIR / 'pca_panel.png')
    # Allow all other args to pass through
    args, unknown = ap.parse_known_args()
    return args, unknown


def run(cmd):
    print(f"[RUN] {cmd}")
    proc = subprocess.run(cmd, shell=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    args, unknown = parse_known()
    py = sys.executable

    args.counts = project_relative(args.counts)
    args.metadata = project_relative(args.metadata)
    args.out_corrected = project_relative(args.out_corrected)
    args.out_corrected.parent.mkdir(parents=True, exist_ok=True)

    for attr in ('viz_pca_before', 'viz_pca_after', 'viz_pca_panel'):
        value = getattr(args, attr)
        value = project_relative(value)
        value.parent.mkdir(parents=True, exist_ok=True)
        setattr(args, attr, value)

    # 1) Train with original script (forward only training args; strip viz-only flags)
    train_cmd = [
        py, 'NN_batch_correct.py',
        '--counts', str(args.counts),
        '--metadata', str(args.metadata),
        '--out_corrected', str(args.out_corrected),
        '--sample_col', args.sample_col,
        '--batch_col', args.batch_col,
    ]
    if args.label_col:
        train_cmd += ['--label_col', args.label_col]
    if args.genes_in_rows:
        train_cmd += ['--genes_in_rows']
    # Append the remaining (unknown) args untouched (e.g., --use_residual, --epochs, etc.)
    train_cmd += unknown
    run(train_cmd)

    # 2) Visualise PCA (corrected is log-like for AE path)
    viz_cmd = [
        py, 'visualise_outputs.py',
        '--counts', str(args.counts),
        '--metadata', str(args.metadata),
        '--sample_col', args.sample_col,
        '--batch_col', args.batch_col,
        '--viz_hvg_top', str(args.viz_hvg_top),
        '--viz_pca_before', str(args.viz_pca_before),
        '--viz_pca_after', str(args.viz_pca_after),
        '--viz_pca_panel', str(args.viz_pca_panel),
        '--corrected', str(args.out_corrected),
        '--corrected_is_log'
    ]
    if args.label_col:
        viz_cmd.extend(['--label_col', args.label_col])
    if args.genes_in_rows:
        viz_cmd.append('--genes_in_rows')

    run(viz_cmd)


if __name__ == '__main__':
    main()
