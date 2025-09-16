#!/usr/bin/env python3
"""
Sweep entrypoint for residual AE: runs NN_batch_correct.py then generates PCA plots.

Reuses the same Python process to avoid costly interpreter start-up between sweep
runs. All non-visual arguments are forwarded verbatim to the training CLI.
"""
import argparse
import shlex
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import OUTPUTS_DIR, project_relative
from NN_batch_correct import main as train_main
from visualise_outputs import main as viz_main


def parse_known() -> Tuple[argparse.Namespace, List[str]]:
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


def call_cli(fn: Callable[[Optional[Sequence[str]]], None], argv: Sequence[str], label: str) -> None:
    cmd = f"{label} {shlex.join(argv)}"
    print(f"[RUN] {cmd}")
    try:
        fn(list(argv))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
        if code != 0:
            raise SystemExit(code) from exc


def main() -> None:
    args, unknown = parse_known()

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
    train_args: List[str] = [
        '--counts', str(args.counts),
        '--metadata', str(args.metadata),
        '--out_corrected', str(args.out_corrected),
        '--sample_col', args.sample_col,
        '--batch_col', args.batch_col,
    ]
    if args.label_col:
        train_args += ['--label_col', args.label_col]
    if args.genes_in_rows:
        train_args += ['--genes_in_rows']

    extra_args = list(unknown)
    train_args.extend(extra_args)
    call_cli(train_main, train_args, 'NN_batch_correct')

    should_run_viz = '--generate_viz' not in extra_args
    if not should_run_viz:
        return

    viz_args: List[str] = [
        '--counts', str(args.counts),
        '--metadata', str(args.metadata),
        '--sample_col', args.sample_col,
        '--batch_col', args.batch_col,
        '--viz_hvg_top', str(args.viz_hvg_top),
        '--viz_pca_before', str(args.viz_pca_before),
        '--viz_pca_after', str(args.viz_pca_after),
        '--viz_pca_panel', str(args.viz_pca_panel),
        '--corrected', str(args.out_corrected),
        '--corrected_is_log',
    ]
    if args.label_col:
        viz_args += ['--label_col', args.label_col]
    if args.genes_in_rows:
        viz_args.append('--genes_in_rows')

    call_cli(viz_main, viz_args, 'visualise_outputs')


if __name__ == '__main__':
    main()
