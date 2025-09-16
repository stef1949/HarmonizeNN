#!/usr/bin/env python3
"""
Compute quantitative stats on batch-corrected data.

Inputs:
  - Corrected matrix CSV (samples x genes)
  - Metadata CSV (with sample IDs, batch column, and optional label/condition column)

Metrics (computed on a PCA embedding of corrected data):
  - batch_silhouette (↓): Silhouette score by batch (lower is better; 0 ~ mixed)
  - label_silhouette (↑): Silhouette score by label (higher is better; optional)
  - knn_batch_entropy (↑): Normalized entropy of neighbor batch composition (0..1; higher is better mixing)
  - ilisi_batch (↑): Inverse Simpson index of neighbor batch composition (>1; higher is better mixing)
  - batch_eta2 (↓): Proportion of variance explained by batch groups in embedding (lower is better)
  - knn_label_acc (↑): 1-NN accuracy for labels (higher is better; optional)
  - label_eta2 (↑): Variance explained by label groups in embedding (higher is better; optional)

Outputs:
  - JSON or CSV file with the metrics.
  - Optional: the PCA embedding as CSV (samples x components).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def _select_hvg(df: pd.DataFrame, n_hvg: int) -> pd.DataFrame:
    if not n_hvg or n_hvg <= 0 or n_hvg >= df.shape[1]:
        return df
    vars_ = df.var(axis=0)
    top = vars_.nlargest(n_hvg).index
    return df.loc[:, top]


def _knn_stats(Z: np.ndarray, labels: np.ndarray, k: int = 15) -> Tuple[float, float, float]:
    try:
        n = Z.shape[0]
        if n < 3 or labels is None:
            return np.nan, np.nan, np.nan
        labels = np.asarray(labels)
        classes, inv = np.unique(labels, return_inverse=True)
        C = int(len(classes))
        if C <= 1:
            return np.nan, np.nan, np.nan
        k_eff = min(int(k), max(2, n - 1))
        nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto").fit(Z)
        idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
        neigh_codes = inv[idx]
        H = []
        L = []
        correct = 0
        for i in range(n):
            counts = np.bincount(neigh_codes[i], minlength=C).astype(float)
            total = counts.sum()
            if total <= 0:
                H.append(np.nan); L.append(np.nan)
                continue
            p = counts / total
            lisi = 1.0 / np.sum(p * p)
            L.append(lisi)
            mask = p > 0
            h = -np.sum(p[mask] * np.log(p[mask] + 1e-12))
            h_norm = h / np.log(C) if C > 1 else np.nan
            H.append(h_norm)
            maj = int(np.argmax(counts))
            correct += (maj == int(inv[i]))
        return float(np.nanmean(H)), float(np.nanmean(L)), float(correct / n)
    except Exception:
        return np.nan, np.nan, np.nan


def _eta2_by_group(Z: np.ndarray, groups: np.ndarray) -> float:
    try:
        n, d = Z.shape
        if n < 3:
            return np.nan
        overall_mean = Z.mean(axis=0)
        var_total = Z.var(axis=0, ddof=1) + 1e-12
        ssb = np.zeros(d, dtype=float)
        for g in np.unique(groups):
            idx = (groups == g)
            ng = idx.sum()
            if ng == 0:
                continue
            diff = Z[idx].mean(axis=0) - overall_mean
            ssb += ng * (diff ** 2)
        eta2 = (ssb / (n * var_total)).mean()
        return float(eta2)
    except Exception:
        return np.nan


def parse_args():
    ap = argparse.ArgumentParser(description="Compute stats on batch-corrected data")
    ap.add_argument('--corrected', required=True, type=Path, help='Corrected matrix CSV (samples x genes)')
    ap.add_argument('--metadata', required=True, type=Path, help='Metadata CSV with sample IDs and batch/label columns')
    ap.add_argument('--sample_col', default='sample', type=str)
    ap.add_argument('--batch_col', default='batch', type=str)
    ap.add_argument('--label_col', default=None, type=str)
    ap.add_argument('--corrected_is_log', action='store_true', help='Set if corrected is already log-like (skip log1p)')
    ap.add_argument('--hvg', default=2000, type=int, help='Top-N genes to use (0=all)')
    ap.add_argument('--n_components', default=30, type=int, help='PCA components for the embedding')
    ap.add_argument('--k', default=15, type=int, help='k for kNN-based stats')
    ap.add_argument('--out', default='corrected_stats.json', type=Path, help='Output path (.json or .csv)')
    ap.add_argument('--save_embedding', default=None, type=Path, help='Optional: save PCA embedding CSV here')
    return ap.parse_args()


def main():
    args = parse_args()

    # Load corrected and metadata
    X = pd.read_csv(args.corrected, index_col=0)
    meta = pd.read_csv(args.metadata)
    if args.sample_col not in meta.columns:
        raise ValueError(f"Sample column '{args.sample_col}' not in metadata")
    meta = meta.set_index(args.sample_col)

    # Align samples
    common = X.index.intersection(meta.index)
    if len(common) < 3:
        raise ValueError("Too few overlapping samples between corrected and metadata")
    X = X.loc[common]
    meta = meta.loc[common]

    # Log transform if needed
    if not args.corrected_is_log and (X.values >= 0).all():
        X = np.log1p(X)

    # HVG selection
    X = _select_hvg(X, args.hvg)

    # PCA embedding
    nc = max(2, int(args.n_components))
    pca = PCA(n_components=nc)
    Z = pca.fit_transform(X.values)
    if args.save_embedding:
        emb_df = pd.DataFrame(Z, index=X.index, columns=[f'PC{i+1}' for i in range(nc)])
        emb_df.to_csv(args.save_embedding)

    # Prepare groups
    batches = meta[args.batch_col].astype('category')
    batch_codes = batches.cat.codes.values

    # Metrics
    stats = {}
    # Silhouettes
    try:
        stats['batch_silhouette'] = float(silhouette_score(Z, batch_codes)) if len(np.unique(batch_codes)) > 1 else np.nan
    except Exception:
        stats['batch_silhouette'] = np.nan

    if args.label_col and args.label_col in meta.columns:
        labels = meta[args.label_col].astype('category')
        label_codes = labels.cat.codes.values
        try:
            stats['label_silhouette'] = float(silhouette_score(Z, label_codes)) if len(np.unique(label_codes)) > 1 else np.nan
        except Exception:
            stats['label_silhouette'] = np.nan
        h, lisi, acc = _knn_stats(Z, label_codes, k=args.k)
        stats['knn_label_entropy'] = float(h) if h == h else np.nan
        stats['knn_label_acc'] = float(acc) if acc == acc else np.nan
        stats['label_eta2'] = float(_eta2_by_group(Z, label_codes))
    else:
        stats['label_silhouette'] = np.nan
        stats['knn_label_entropy'] = np.nan
        stats['knn_label_acc'] = np.nan
        stats['label_eta2'] = np.nan

    # Batch kNN-based
    h, lisi, _ = _knn_stats(Z, batch_codes, k=args.k)
    stats['knn_batch_entropy'] = float(h) if h == h else np.nan
    stats['ilisi_batch'] = float(lisi) if lisi == lisi else np.nan
    stats['batch_eta2'] = float(_eta2_by_group(Z, batch_codes))

    # Save
    out_path = args.out
    if out_path.suffix.lower() == '.csv':
        pd.DataFrame([stats]).to_csv(out_path, index=False)
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    print(f"[OK] Wrote stats to: {out_path}")
    # Also print concise summary with arrows indicating direction
    print("Summary:")
    directions = {
        'batch_silhouette': 'lower',
        'label_silhouette': 'higher',
        'knn_batch_entropy': 'higher',
        'ilisi_batch': 'higher',
        'batch_eta2': 'lower',
        'knn_label_acc': 'higher',
        'label_eta2': 'higher',
    }
    arrows = {'higher': '↑', 'lower': '↓'}
    # Output in a stable order
    order = [
        'batch_silhouette', 'label_silhouette',
        'knn_batch_entropy', 'ilisi_batch', 'batch_eta2',
        'knn_label_acc', 'label_eta2',
    ]
    for k in order:
        if k in stats:
            dirn = directions.get(k, '')
            arrow = arrows.get(dirn, '')
            print(f"  {k}: {stats[k]} {arrow}")
    # Print any additional fields not in default order
    for k in stats:
        if k not in order:
            print(f"  {k}: {stats[k]}")


if __name__ == '__main__':
    main()
