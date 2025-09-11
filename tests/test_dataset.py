import numpy as np
import torch

from NN_batch_correct import RNADataset


def test_rna_dataset_no_labels():
    X = np.random.RandomState(0).randn(7, 4).astype(np.float32)
    B = np.array([0, 1, 0, 1, 0, 1, 1], dtype=np.int64)
    ds = RNADataset(X, B)
    assert len(ds) == 7
    xb, bb = ds[3]
    assert isinstance(xb, torch.Tensor) and xb.shape == (4,)
    assert isinstance(bb, torch.Tensor) and bb.shape == ()


def test_rna_dataset_with_labels():
    X = np.random.RandomState(1).randn(5, 3).astype(np.float32)
    B = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    L = np.array([1, 0, 1, 0, 1], dtype=np.int64)
    ds = RNADataset(X, B, L)
    x0, b0, l0 = ds[0]
    assert x0.shape == (3,)
    assert b0.dtype == torch.long
    assert l0.dtype == torch.long

