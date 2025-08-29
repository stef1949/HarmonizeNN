#!/usr/bin/env Rscript
# Synthetic bulk RNA-Seq generator for batch-correction testing
# Outputs:
#   - bulk_counts.csv   (genes x samples; raw counts)
#   - sample_meta.csv   (sample,batch,condition)
#
# Design:
# - Negative Binomial (gene-wise mean & dispersion)
# - Balanced, NON-CONFOUNDED batches and conditions
# - Multiplicative batch effects (per-gene LFC ~ N(0, batch_sd))
# - Optional additive library-size shifts per batch & per sample
# - Differential expression in a subset of genes with random LFCs

suppressPackageStartupMessages({
  # no special packages required; base R only
})

set.seed(1234)

# -----------------------------
# Tunable knobs (edit as needed)
# -----------------------------
n_genes        <- 50000        # total genes
n_batches      <- 4            # number of batches
n_cond         <- 2            # number of conditions (e.g., case/control)
n_per_batch    <- 30           # samples per batch
de_prop        <- 0.10         # proportion of genes that are DE
de_lfc_sd      <- 1.0          # SD of DE log2 fold-change
# (bigger => stronger biology)
batch_sd       <- 0.30         # SD of per-gene batch
# log2-effect (bigger => stronger batch)
libsize_sd     <- 0.25         # SD of per-sample library size log-normal factor
batch_lib_l2fc <- 0.20         # log2 library-size shift
# per batch (site throughput)
min_count_frac <- 0.00         # drop genes with
# zero fraction > this (0 keeps all)
out_counts     <- "bulk_counts.csv"
out_meta       <- "sample_meta.csv"

# -----------------------------
# Sample design (balanced)
# -----------------------------
n_samples <- n_batches * n_per_batch
batches   <- factor(rep(paste0("B", seq_len(n_batches)), each = n_per_batch))
# conditions balanced within each batch (non-confounded)
conds     <- unlist(lapply(seq_len(n_batches), function(b) {
  rep(paste0("C", seq_len(n_cond)), length.out = n_per_batch)
}))
condition <- factor(conds)

# Map C1/C2 -> normal/tumor by default when there are 2 conditions
if (n_cond == 2) {
  # ensure levels are C1, C2 then map
  levels(condition) <- c("normal", "tumor")
}

# -----------------------------
# Gene-level parameters
# -----------------------------
# Base gene means (on counts) — heavy-tailed like RNA-seq:
# Draw base log2 mean, then convert

# ~ 2^1.5 ≈ 2.8 counts baseline (pre library factor)
base_log2_mu <- rnorm(n_genes, mean = 1.5, sd = 1.25)
base_mu      <- pmax(2^(base_log2_mu), 1e-6)

# Gene-wise dispersion (NB): variance = mu + mu^2/size; size ~= 1/phi
# Let phi vary (typical bulk ~ 0.05–0.2). We sample phi log-normally.
phi <- rlnorm(n_genes, meanlog = log(0.10), sdlog = 0.5)  # dispersion
size <- 1 / pmax(phi, 1e-8)

# -----------------------------
# Differential expression (biology)
# -----------------------------
is_de <- rbinom(n_genes, 1, de_prop) == 1
lfc_mat <- matrix(0, nrow = n_genes, ncol = n_cond)
colnames(lfc_mat) <- paste0("C", seq_len(n_cond))
# Reference condition C1 has 0 by construction;
# others get random LFCs on DE genes
for (k in 2:n_cond) {
  # symmetric up/down regulation; continuous LFCs
  lfc <- rnorm(n_genes, mean = 0, sd = de_lfc_sd)
  lfc[!is_de] <- 0
  lfc_mat[, k] <- lfc
}
lfc_mat[, 1] <- 0  # reference

# -----------------------------
# Batch effects (technical)
# -----------------------------
# Per-gene multiplicative batch effects (log2-scale)
batch_lfc_gene <- matrix(rnorm(n_genes * n_batches, mean = 0, sd = batch_sd),
                         nrow = n_genes, ncol = n_batches)
colnames(batch_lfc_gene) <- paste0("B", seq_len(n_batches))

# Library-size multipliers
# - Per-batch shift (site throughput)
batch_lib_shift <- 2^(rnorm(n_batches, mean = batch_lib_l2fc, sd = 0.05))
names(batch_lib_shift) <- paste0("B", seq_len(n_batches))
# - Per-sample random factor
sample_lib_factor <- 2^(rnorm(n_samples, mean = 0, sd = libsize_sd))

# -----------------------------
# Build per-sample gene-wise means (mu_ij)
# -----------------------------
# mu_ij = base_mu * 2^(lfc_condition[gene, cond_j]) * 2^(batch_lfc_gene[gene, batch_j]) * libsize_batch_j * libsize_sample_j
mu <- matrix(0, nrow = n_genes, ncol = n_samples)
for (j in seq_len(n_samples)) {
  bj <- as.integer(batches[j])
  cj <- as.integer(condition[j])
  # gene-specific effects
  log2_mu_j <- log2(base_mu) + lfc_mat[, cj] + batch_lfc_gene[, bj]
  mu[, j] <- pmax(2^(log2_mu_j), 1e-12) * batch_lib_shift[bj] * sample_lib_factor[j]
}

# -----------------------------
# Simulate counts via NB
# -----------------------------
counts <- matrix(0L, nrow = n_genes, ncol = n_samples)
for (j in seq_len(n_samples)) {
  # rnbinom uses 'size' and 'mu'
  counts[, j] <- rnbinom(n_genes, size = size, mu = mu[, j])
}
rownames(counts) <- paste0("Gene", sprintf("%05d", seq_len(n_genes)))
colnames(counts) <- paste0("S", sprintf("%03d", seq_len(n_samples)))

# -----------------------------
# Optional gene filtering (speed)
# -----------------------------
if (min_count_frac > 0) {
  keep <- rowMeans(counts > 0) >= min_count_frac
  counts <- counts[keep, , drop = FALSE]
}

# -----------------------------
# Write outputs
# -----------------------------
# bulk_counts.csv: genes x samples
write.csv(as.data.frame(counts), file = out_counts, row.names = TRUE)

# sample_meta.csv
meta <- data.frame(
  sample    = colnames(counts),
  batch     = batches,
  condition = condition
)
write.csv(meta, file = out_meta, row.names = FALSE)

message(sprintf("Wrote %s  (genes x samples = %d x %d)", out_counts, nrow(counts), ncol(counts)))
message(sprintf("Wrote %s  (batches=%d, conditions=%d, samples=%d)",
                out_meta, nlevels(batches), nlevels(condition), nrow(meta)))
message("Non-confounded by construction: each batch contains all conditions.")
