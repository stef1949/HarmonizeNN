#!/usr/bin/env python3
"""
Quick PCA visualisations for before/after batch correction.

Inputs:
  - Raw counts CSV and matching sample metadata
  - Optional corrected matrix CSV (after correction)

Outputs:
  - PCA before plot
  - PCA after plot (if corrected provided)
  - Side-by-side PCA panel (if corrected provided)
"""

from pathlib import Path
from typing import Optional

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


from paths import DATA_DIR, OUTPUTS_DIR, project_relative


def library_size_normalize(counts_df: pd.DataFrame, cpm_factor: float = 1e6) -> pd.DataFrame:
    lib_sizes = counts_df.sum(axis=1).replace(0, np.nan)
    x = counts_df.div(lib_sizes, axis=0) * cpm_factor
    return np.log1p(x)


def select_hvg(df: pd.DataFrame, n_hvg: int) -> pd.DataFrame:
    if not n_hvg or n_hvg <= 0 or n_hvg >= df.shape[1]:
        return df
    vars_ = df.var(axis=0)
    top = vars_.nlargest(n_hvg).index
    return df.loc[:, top]


def _pca_plot(df: pd.DataFrame, meta: pd.DataFrame, batch_col: str, label_col: Optional[str], out_path: Path, title: str):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(df.values)
    pc_df = pd.DataFrame(Z, index=df.index, columns=["PC1", "PC2"])
    batches = meta.loc[pc_df.index, batch_col].astype("category")
    batch_codes = batches.cat.codes.values
    batch_names = batches.cat.categories.tolist()
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap('tab10')
    for c in np.unique(batch_codes):
        mask = batch_codes == c
        lbl = batch_names[int(c)] if int(c) < len(batch_names) else str(int(c))
        ax.scatter(pc_df.loc[mask, 'PC1'], pc_df.loc[mask, 'PC2'], s=14, color=cmap(int(c) % cmap.N), label=lbl, alpha=0.85)
    if label_col and label_col in meta.columns:
        labels = meta.loc[pc_df.index, label_col].astype('category')
        if len(labels.cat.categories) > 1:
            markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
            for i, cat in enumerate(labels.cat.categories):
                mask = labels == cat
                ax.scatter(pc_df.loc[mask, 'PC1'], pc_df.loc[mask, 'PC2'], facecolors='none', edgecolors='k', marker=markers[i % len(markers)], s=60, linewidths=0.9, label=f"{cat}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(title='batch', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved PCA plot: {out_path}")


def _pca_panel(before_df: pd.DataFrame, after_df: pd.DataFrame, meta: pd.DataFrame, batch_col: str, out_path: Path, title: str, label_col: Optional[str] = None):
    # Align samples and fit one PCA on concatenated data for comparable axes
    common = before_df.index.intersection(after_df.index)
    if len(common) < 2:
        print("[WARN] PCA panel skipped: fewer than 2 overlapping samples.")
        return
    Bf = before_df.loc[common]
    Af = after_df.loc[common]
    # Align features robustly by inner-joining on gene columns
    Bf, Af = Bf.align(Af, join='inner', axis=1)
    if Bf.shape[1] < 2:
        print("[WARN] PCA panel skipped: fewer than 2 overlapping genes between before/after matrices.")
        return
    batches = meta.loc[common, batch_col].astype('category')
    batch_codes = batches.cat.codes.values
    batch_names = batches.cat.categories.tolist()
    # Fit separate PCAs per view to allow different scales/bases
    pca_b = PCA(n_components=2).fit(Bf.values)
    pca_a = PCA(n_components=2).fit(Af.values)
    Zb = pca_b.transform(Bf.values)
    Za = pca_a.transform(Af.values)
    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=False, sharey=False)
    for ax, Z, ttl in zip(axes, [Zb, Za], ["Before", "After"]):
        for c in np.unique(batch_codes):
            mask = batch_codes == c
            lbl = batch_names[int(c)] if int(c) < len(batch_names) else str(int(c))
            ax.scatter(Z[mask, 0], Z[mask, 1], s=10, color=cmap(int(c) % cmap.N), label=lbl, alpha=0.85)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl)
    # Optional label overlays with distinct marker shapes
    if label_col and (label_col in meta.columns):
        labels = meta.loc[common, label_col].astype('category')
        if len(labels.cat.categories) > 1:
            cats = list(labels.cat.categories)
            markers_cycle = ["o","s","^","D","v","P","X","*","<",">"]
            # Prefer fixed mapping for common pairs tumor/normal
            lower = [c.lower() for c in cats]
            fixed_map = {}
            if 'tumor' in lower and 'normal' in lower:
                for c in cats:
                    if c.lower() == 'tumor': fixed_map[c] = '^'
                    elif c.lower() == 'normal': fixed_map[c] = 'o'
            for i, cat in enumerate(cats):
                m = fixed_map.get(cat, markers_cycle[i % len(markers_cycle)])
                mask = (labels == cat).values
                for ax, Z in ((axes[0], Zb), (axes[1], Za)):
                    ax.scatter(Z[mask,0], Z[mask,1], facecolors='none', edgecolors='k', marker=m, s=60, linewidths=0.9, label=f"{cat} (label)")
    # one legend
    handles = []
    labels = []
    for c in np.unique(batch_codes):
        lbl = batch_names[int(c)] if int(c) < len(batch_names) else str(int(c))
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=lbl, markerfacecolor=cmap(int(c) % cmap.N), markersize=6))
        labels.append(lbl)
    axes[0].legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    evr_b = pca_b.explained_variance_ratio_
    evr_a = pca_a.explained_variance_ratio_
    fig.suptitle(
        f"{title} — Before PC1 {evr_b[0]*100:.1f}%, PC2 {evr_b[1]*100:.1f}% | After PC1 {evr_a[0]*100:.1f}%, PC2 {evr_a[1]*100:.1f}%"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved PCA panel: {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Generate PCA plots before/after correction")
    ap.add_argument('--counts', default=None, type=Path, help='Raw counts CSV (samples x genes, or use --genes_in_rows). If omitted, defaults to data/bulk_counts.csv')
    ap.add_argument('--metadata', default=None, type=Path, help='Metadata CSV with sample, batch, and optional label cols. If omitted, defaults to data/sample_meta.csv')
    ap.add_argument('--sample_col', default='sample', type=str)
    ap.add_argument('--batch_col', default='batch', type=str)
    ap.add_argument('--label_col', default=None, type=str)
    ap.add_argument('--genes_in_rows', action='store_true', help='Set if counts CSV has genes as rows')
    ap.add_argument('--corrected', default=None, type=Path, help='Optional corrected matrix CSV (samples x genes). Defaults to artifacts/outputs/corrected_logcpm.csv if present')
    ap.add_argument('--corrected_is_log', action='store_true', help='Set if corrected values are already log-transformed (skip log1p)')
    ap.add_argument('--viz_hvg_top', default=2000, type=int, help='Top-N genes for PCA (0=all)')
    ap.add_argument('--viz_pca_before', default=OUTPUTS_DIR / 'pca_before.png', type=Path)
    ap.add_argument('--viz_pca_after', default=OUTPUTS_DIR / 'pca_after.png', type=Path)
    ap.add_argument('--viz_pca_panel', default=OUTPUTS_DIR / 'pca_panel.png', type=Path)
    return ap.parse_args()


def main():
    args = parse_args()

    if args.counts is None:
        default_counts = DATA_DIR / 'bulk_counts.csv'
        if default_counts.exists():
            args.counts = default_counts
            print(f"[INFO] Using default counts file: {default_counts}")
        else:
            raise SystemExit('ERROR: --counts not provided and data/bulk_counts.csv not found')
    else:
        args.counts = project_relative(args.counts)

    if args.metadata is None:
        default_meta = DATA_DIR / 'sample_meta.csv'
        if default_meta.exists():
            args.metadata = default_meta
            print(f"[INFO] Using default metadata file: {default_meta}")
        else:
            raise SystemExit('ERROR: --metadata not provided and data/sample_meta.csv not found')
    else:
        args.metadata = project_relative(args.metadata)

    if args.corrected is None:
        default_corrected = OUTPUTS_DIR / 'corrected_logcpm.csv'
        if default_corrected.exists():
            args.corrected = default_corrected
            print(f"[INFO] Using default corrected matrix: {default_corrected}")
    else:
        args.corrected = project_relative(args.corrected)

    for attr in ('viz_pca_before', 'viz_pca_after', 'viz_pca_panel'):
        value = getattr(args, attr)
        value = project_relative(value)
        value.parent.mkdir(parents=True, exist_ok=True)
        setattr(args, attr, value)

    counts = pd.read_csv(args.counts, index_col=0)
    if args.genes_in_rows:
        counts = counts.T
    meta = pd.read_csv(args.metadata)
    if args.sample_col not in meta.columns:
        raise ValueError(f"Sample column '{args.sample_col}' not found in metadata")
    meta = meta.set_index(args.sample_col)
    common = counts.index.intersection(meta.index)
    if len(common) < 2:
        raise ValueError('Fewer than 2 overlapping samples between counts and metadata')
    counts = counts.loc[common]
    meta = meta.loc[common]

    # Before: logCPM
    logcpm_before = library_size_normalize(counts)
    before_sel = select_hvg(logcpm_before, args.viz_hvg_top)
    _pca_plot(before_sel, meta, args.batch_col, args.label_col, args.viz_pca_before, title=f"PCA (before) HVG={args.viz_hvg_top}")

    # After: corrected provided?
    if args.corrected:
        corrected = pd.read_csv(args.corrected, index_col=0)
        # Align samples
        corrected = corrected.loc[corrected.index.intersection(meta.index)]
        # If after has strictly non-negative values, assume counts-like and log1p unless flagged as already log
        if not args.corrected_is_log and (corrected.values >= 0).all():
            corrected_log = np.log1p(corrected)
        else:
            corrected_log = corrected
        # Ensure same gene set as before selection if possible
        after_sel = corrected_log.loc[:, [g for g in before_sel.columns if g in corrected_log.columns]]
        if after_sel.shape[1] < 2:
            after_sel = corrected_log
        _pca_plot(after_sel, meta.loc[after_sel.index], args.batch_col, args.label_col, args.viz_pca_after, title=f"PCA (after) HVG={args.viz_hvg_top}")
        # Panel
        common_samples = before_sel.index.intersection(after_sel.index)
        if len(common_samples) >= 2:
            _pca_panel(before_sel.loc[common_samples], after_sel.loc[common_samples], meta.loc[common_samples], args.batch_col, args.viz_pca_panel, title=f"PCA Before vs After (HVG {args.viz_hvg_top})", label_col=args.label_col)


if __name__ == '__main__':
    main()
