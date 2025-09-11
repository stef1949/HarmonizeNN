import numpy as np
import pandas as pd

from NN_batch_correct import (
    library_size_normalize,
    select_hvg_train_only,
    standardize_per_gene_fit_transform,
    inverse_standardize,
)


def test_library_size_normalize_basic():
    # Two samples, three genes with simple counts
    df = pd.DataFrame(
        [[10, 20, 30], [5, 5, 10]],
        index=["s1", "s2"],
        columns=["g1", "g2", "g3"],
        dtype=float,
    )
    out = library_size_normalize(df, cpm_factor=1e6)
    # CPM for s1: total=60 -> [166666.7, 333333.3, 500000.0]
    # CPM for s2: total=20 -> [250000.0, 250000.0, 500000.0]
    expected = pd.DataFrame(
        [
            np.log1p([1e6 * 10 / 60, 1e6 * 20 / 60, 1e6 * 30 / 60]),
            np.log1p([1e6 * 5 / 20, 1e6 * 5 / 20, 1e6 * 10 / 20]),
        ],
        index=df.index,
        columns=df.columns,
    )
    assert np.allclose(out.values, expected.values, rtol=1e-6, atol=1e-6)


def test_select_hvg_train_only_uses_train_variance():
    # Construct 4 samples (0..3), 4 genes (g0..g3)
    # Make g0 vary only in train, g1 vary only in val, g2 moderate in both, g3 least
    X = np.array(
        [
            [0.0, 0.0, 0.5, 0.1],  # s0 train
            [3.0, 0.0, 0.4, 0.1],  # s1 train -> g0 high var in train
            [0.1, 5.0, 0.6, 0.1],  # s2 val   -> g1 high var in val only
            [0.2, 5.0, 0.7, 0.1],  # s3 val
        ]
    )
    df = pd.DataFrame(X, index=["s0", "s1", "s2", "s3"], columns=["g0", "g1", "g2", "g3"])
    train_idx = np.array([0, 1])
    sel = select_hvg_train_only(df, train_idx, n_hvg=2)
    # Expect g0 (var in train) and g2 (moderate overall) rather than g1 (var only in val)
    assert set(sel.columns) == {"g0", "g2"}


def test_standardize_and_inverse_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=10.0, scale=2.0, size=(10, 5))
    df = pd.DataFrame(X, index=[f"s{i}" for i in range(10)], columns=[f"g{j}" for j in range(5)])
    train_idx = np.arange(6)  # fit on first 6 samples
    X_train, X_all, scaler = standardize_per_gene_fit_transform(df, train_idx)

    # Train set should be mean 0, std 1 per feature (approximately for small N)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=0)
    assert np.allclose(mu, 0.0, atol=1e-7)
    # Allow small numerical deviation; std should be 1
    assert np.allclose(sd, 1.0, atol=1e-6)

    # Inverse-transform should approximately recover original values
    X_round = inverse_standardize(X_all, scaler)
    assert np.allclose(X_round, df.values, rtol=1e-7, atol=1e-7)

