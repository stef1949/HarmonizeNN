# Visualize the neural network architecture of the adversarial autoencoder for bulk RNA-seq.
# This creates:
# 1) A clean diagram (PNG) of the modules and data flow
# 2) A parameter table per submodule
#
# The diagram is generic (independent of dataset) and reflects the default sizes used in nn_batch_correct.py
# You can re-run this cell after editing the sizes below.

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.decomposition import PCA
from matplotlib import gridspec
import argparse
from typing import Optional

# ----------------------------
# Define a lightweight version of the architecture config
# ----------------------------

n_genes   = 5000       # input gene features (after HVG selection)
latent_dim = 32
enc_hidden = (1024, 256)
dec_hidden = (256, 1024)
adv_hidden = (128,)
sup_hidden = (64,)
n_batches  = 4         # e.g., 4 batches
n_labels   = 2         # e.g., 2 biological classes (optional head)

# ----------------------------
# Helper: MLP param counting
# ----------------------------
def mlp_param_table(sizes, name_prefix):
    rows = []
    for i in range(len(sizes)-1):
        inp, out = sizes[i], sizes[i+1]
        # Linear layer params: W (out x inp) + b (out)
        params = out*inp + out
        rows.append({
            "Layer": f"{name_prefix}/Linear_{i+1}",
            "Input dim": inp,
            "Output dim": out,
            "Parameters": params
        })
    return rows

enc_sizes = (n_genes, ) + enc_hidden + (latent_dim, )
dec_sizes = (latent_dim, ) + dec_hidden + (n_genes, )
adv_sizes = (latent_dim, ) + adv_hidden + (n_batches, )
sup_sizes = (latent_dim, ) + sup_hidden + (n_labels, )

rows = []
rows += mlp_param_table(enc_sizes, "Encoder")
rows += mlp_param_table(dec_sizes, "Decoder")
rows += mlp_param_table(adv_sizes, "Adversary(batch)")
rows += mlp_param_table(sup_sizes, "Supervised(label)")

df = pd.DataFrame(rows)
df["Parameters"] = df["Parameters"].astype(int)
df_totals = (
    df.assign(Module=df["Layer"].str.split("/").str[0])
      .groupby("Module")["Parameters"]
      .sum()
      .reset_index()
      .rename(columns={"Parameters":"Module Parameters"})
)

# Display the tables to the user (console output)
print("\nNN layer-by-layer parameters:")
print(df.to_string(index=False))
print("\nNN parameters per module:")
print(df_totals.to_string(index=False))

# ----------------------------
# Draw a clean block diagram using matplotlib
# ----------------------------

def draw_box(ax, xy, w, h, text, fontsize=11, facecolor=None, edgecolor="#333"):
    """Draw a rounded box (FancyBboxPatch) with centered text and return the patch."""
    box = FancyBboxPatch((xy[0], xy[1]), w, h,
                         boxstyle="round,pad=0.02,rounding_size=6",
                         linewidth=1.2, edgecolor=edgecolor, facecolor=facecolor, alpha=0.9)
    ax.add_patch(box)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha='center', va='center', fontsize=fontsize)
    return box

def draw_arrow(ax, start_xy, end_xy, color="#555"):
    ax.annotate("",
                xy=end_xy, xycoords='data',
                xytext=start_xy, textcoords='data',
                arrowprops=dict(arrowstyle="->", lw=1.6, color=color, connectionstyle='arc3'))

# Canvas (slightly larger for labels)
fig_w, fig_h = 13, 6.5
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

ax.set_xlim(0, 13)
ax.set_ylim(0, 6.5)
ax.axis('off')

# Layout coordinates
# Encoder chain (left to center)
x0, y0 = 0.5, 3.5
box_w, box_h = 1.6, 0.9
gap = 0.6

# Color scheme per module
colors = {
    'Encoder': '#c6dbef',
    'Decoder': '#fde0dd',
    'Adversary(batch)': '#e5f5e0',
    'Supervised(label)': '#fff2cc',
    'Latent': '#d9d9d9'
}

# Encoder boxes
enc_dims = [n_genes] + list(enc_hidden) + [latent_dim]
enc_boxes = []
for i, (inp, out) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
    bx = x0 + i*(box_w + gap)
    b = draw_box(ax, (bx, y0), box_w, box_h, f"Enc\n{inp}→{out}", facecolor=colors['Encoder'], edgecolor='#2b6ca3')
    enc_boxes.append(b)
    if i > 0:
        # Draw arrow from previous box to this box
        prev = enc_boxes[i-1]
        draw_arrow(ax, (prev.get_x()+box_w, prev.get_y()+box_h/2),
                        (b.get_x(), b.get_y()+box_h/2))

# Latent node
latent_x = enc_boxes[-1].get_x() + box_w + 0.6
latent_y = y0 + box_h/2 - 0.2
latent = draw_box(ax, (latent_x, y0), 1.2, box_h, f"Latent\n{latent_dim}", facecolor=colors['Latent'], edgecolor='#666')

# Arrow from encoder to latent
draw_arrow(ax, (enc_boxes[-1].get_x()+box_w, enc_boxes[-1].get_y()+box_h/2),
                (latent.get_x(), latent.get_y()+box_h/2))

# Decoder chain (center to right)
dec_dims = [latent_dim] + list(dec_hidden) + [n_genes]
dec_boxes = []
for i, (inp, out) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
    bx = latent.get_x() + 1.2 + i*(box_w + gap)
    by = y0
    b = draw_box(ax, (bx, by), box_w, box_h, f"Dec\n{inp}→{out}", facecolor=colors['Decoder'], edgecolor='#b3504a')
    dec_boxes.append(b)
    # arrows
    if i == 0:
        draw_arrow(ax, (latent.get_x()+1.2, latent.get_y()+box_h/2),
                        (b.get_x(), by+box_h/2))
    else:
        prev = dec_boxes[i-1]
        draw_arrow(ax, (prev.get_x()+box_w, prev.get_y()+box_h/2),
                        (b.get_x(), b.get_y()+box_h/2))

# Adversary branch (downwards)
adv_dims = [latent_dim] + list(adv_hidden) + [n_batches]
adv_boxes = []
branch_x = latent.get_x() + 0.6 - box_w/2
branch_y = y0 - (box_h + 1.2)
for i, (inp, out) in enumerate(zip(adv_dims[:-1], adv_dims[1:])):
    by = branch_y - i*(box_h + 0.4)
    b = draw_box(ax, (branch_x, by), box_w, box_h, f"Adv\n{inp}→{out}", facecolor=colors['Adversary(batch)'], edgecolor='#2b7a3b')
    adv_boxes.append(b)
    if i == 0:
        draw_arrow(ax, (latent.get_x()+0.6, latent.get_y()),
                        (b.get_x()+box_w/2, b.get_y()+box_h))
    else:
        prev = adv_boxes[i-1]
        draw_arrow(ax, (prev.get_x()+box_w/2, prev.get_y()),
                        (b.get_x()+box_w/2, b.get_y()+box_h))

# Supervised branch (upwards)
sup_dims = [latent_dim] + list(sup_hidden) + [n_labels]
sup_boxes = []
branch_y_up = y0 + (box_h + 1.2)
for i, (inp, out) in enumerate(zip(sup_dims[:-1], sup_dims[1:])):
    by = branch_y_up + i*(box_h + 0.4)
    b = draw_box(ax, (branch_x, by), box_w, box_h, f"Sup\n{inp}→{out}", facecolor=colors['Supervised(label)'], edgecolor='#b8860b')
    sup_boxes.append(b)
    if i == 0:
        draw_arrow(ax, (latent.get_x()+0.6, latent.get_y()+box_h),
                        (b.get_x()+box_w/2, b.get_y()))
    else:
        prev = sup_boxes[i-1]
        draw_arrow(ax, (prev.get_x()+box_w/2, prev.get_y()+box_h),
                        (b.get_x()+box_w/2, b.get_y()))

# Titles
ax.text(0.5, 5.4, "Adversarial Autoencoder for Bulk RNA-seq Batch Correction", fontsize=14, ha='left', va='center')
ax.text(latent.get_x()+0.6, y0+box_h+0.3, "Gradient Reversal → discourages batch info in latent", ha='center')

out_path = Path("nn_architecture.png")
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches="tight")
out_path.__str__()


# ----------------------------
# Additional visualisations: PCA and logCPM boxplots
# ----------------------------


def library_size_normalize(counts_df: pd.DataFrame, cpm_factor: float = 1e6) -> pd.DataFrame:
    lib_sizes = counts_df.sum(axis=1).replace(0, np.nan)
    x = counts_df.div(lib_sizes, axis=0) * cpm_factor
    x = np.log1p(x)
    return x


def plot_pca(logcpm: pd.DataFrame, meta: pd.DataFrame, batch_col: str, label_col: Optional[str], out_path: str, title: str = "PCA"):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(logcpm.values)
    df = pd.DataFrame(Z, index=logcpm.index, columns=["PC1", "PC2"])
    batches = meta[batch_col].astype('category')
    batch_codes = batches.cat.codes.loc[df.index].values
    batch_names = batches.cat.categories.tolist()

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap('tab10')
    for c in np.unique(batch_codes):
        mask = batch_codes == c
        label = batch_names[int(c)] if int(c) < len(batch_names) else str(int(c))
        ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'], s=18, color=cmap(int(c) % cmap.N), label=label, alpha=0.85)

    if label_col is not None and label_col in meta.columns:
        labels = meta.loc[df.index, label_col].astype('category')
        markers = ['o', 's', '^', 'D', 'v']
        for i, lvl in enumerate(labels.cat.categories):
            mask = labels.cat.codes.values == i
            ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'], s=18, facecolors='none', edgecolors='k', marker=markers[i % len(markers)], linewidths=0.6)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(title='batch', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved PCA: {out_path}")


def select_hvg(logcpm: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Select top-N most variable genes (columns) from a samples x genes logCPM matrix."""
    if top_n <= 0 or top_n >= logcpm.shape[1]:
        return logcpm
    vars_ = logcpm.var(axis=0)
    top = vars_.nlargest(top_n).index.tolist()
    return logcpm.loc[:, top]


def plot_boxplots(logcpm_before: pd.DataFrame, logcpm_after: Optional[pd.DataFrame], meta: pd.DataFrame, batch_col: str, out_path: str):
    batches = meta[batch_col].astype('category')
    batch_names = batches.cat.categories.tolist()
    before_groups = []
    after_groups = []
    labels = []
    for b in batch_names:
        samp_ids = meta.index[meta[batch_col] == b].tolist()
        if len(samp_ids) == 0:
            before_groups.append([])
            after_groups.append([])
            labels.append(b)
            continue
        vals_before = logcpm_before.loc[samp_ids].values.flatten()
        before_groups.append(vals_before)
        if logcpm_after is not None:
            vals_after = logcpm_after.loc[samp_ids].values.flatten()
            after_groups.append(vals_after)
        else:
            after_groups.append(None)
        labels.append(b)

    n = len(batch_names)
    fig = plt.figure(figsize=(max(6, n*1.2), 5))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])

    positions = []
    data = []
    tick_pos = []
    width = 0.35
    for i in range(n):
        left = i*2
        positions.append(left)
        data.append(before_groups[i])
        positions.append(left + width)
        if after_groups[i] is not None:
            data.append(after_groups[i])
        else:
            data.append([])
        tick_pos.append(left + width/2)

    bplots = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=False)
    for idx, patch in enumerate(bplots['boxes']):
        if idx % 2 == 0:
            patch.set_facecolor('#a6cee3')
        else:
            patch.set_facecolor('#b2df8a')

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('logCPM')
    ax.set_title('logCPM distributions by batch (before=blue, after=green)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved boxplots: {out_path}")


def main():
    ap = argparse.ArgumentParser(description='Visualise PCA and logCPM boxplots before/after correction')
    ap.add_argument('--counts', required=True, help='Raw counts CSV (genes x samples)')
    ap.add_argument('--metadata', required=True, help='Sample metadata CSV (sample,batch[,condition])')
    ap.add_argument('--corrected', default=None, help='Corrected logCPM CSV (samples x genes) produced by NN_batch_correct.py')
    ap.add_argument('--genes_in_rows', action='store_true', help='Set if counts CSV is genes in rows (default for generator)')
    ap.add_argument('--sample_col', default='sample')
    ap.add_argument('--batch_col', default='batch')
    ap.add_argument('--label_col', default='condition')
    ap.add_argument('--hvg_top', type=int, default=0, help='Top-N most variable genes to use for PCA (0 = use all)')
    ap.add_argument('--pca_before', default='pca_before.png')
    ap.add_argument('--pca_after', default='pca_after.png')
    ap.add_argument('--boxplot', default='logCPM_boxplots.png')
    args = ap.parse_args()

    counts = pd.read_csv(args.counts, index_col=0)
    if args.genes_in_rows:
        counts = counts.T
    meta = pd.read_csv(args.metadata)
    if args.sample_col not in meta.columns:
        raise ValueError(f"metadata must contain sample column '{args.sample_col}'")
    meta = meta.set_index(args.sample_col)

    if counts.shape[0] != meta.shape[0]:
        if counts.shape[1] == meta.shape[0]:
            counts = counts.T
        else:
            print('[WARN] counts/sample-metadata shape mismatch; continuing with matching intersection')
    common = counts.index.intersection(meta.index)
    if len(common) < 1:
        raise ValueError('No overlapping samples between counts and metadata')
    counts = counts.loc[common]
    meta = meta.loc[common]

    logcpm_before = library_size_normalize(counts)

    selected_genes = None
    if args.hvg_top and args.hvg_top > 0:
        selected = select_hvg(logcpm_before, args.hvg_top)
        selected_genes = selected.columns.tolist()
        logcpm_for_pca = selected
    else:
        logcpm_for_pca = logcpm_before

    plot_pca(logcpm_for_pca, meta, args.batch_col, args.label_col if args.label_col in meta.columns else None, args.pca_before, title=f'PCA (before correction){" - HVG="+str(args.hvg_top) if args.hvg_top and args.hvg_top>0 else ""}')

    if args.corrected is not None:
        corrected = pd.read_csv(args.corrected, index_col=0)
        if corrected.shape[0] != meta.shape[0] and corrected.shape[1] == meta.shape[0]:
            corrected = corrected.T
        common2 = corrected.index.intersection(meta.index)
        corrected = corrected.loc[common2]
        meta2 = meta.loc[common2]
        # If HVG selection was used for before-PCA, restrict corrected to the same genes when possible
        if selected_genes is not None:
            genes_present = [g for g in selected_genes if g in corrected.columns]
            if len(genes_present) < 2:
                print(f"[WARN] Fewer than 2 selected HVG genes found in corrected data; skipping HVG restriction")
                corrected_for_pca = corrected
            else:
                corrected_for_pca = corrected.loc[:, genes_present]
        else:
            corrected_for_pca = corrected

        plot_pca(corrected_for_pca, meta2, args.batch_col, args.label_col if args.label_col in meta2.columns else None, args.pca_after, title=f'PCA (after correction){" - HVG="+str(args.hvg_top) if args.hvg_top and args.hvg_top>0 else ""}')
        both_idx = logcpm_before.index.intersection(corrected.index)
        plot_boxplots(logcpm_before.loc[both_idx], corrected.loc[both_idx], meta.loc[both_idx], args.batch_col, args.boxplot)
    else:
        plot_boxplots(logcpm_before, None, meta, args.batch_col, args.boxplot)


if __name__ == '__main__':
    main()
