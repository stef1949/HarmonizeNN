#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for HarmonizeNN batch correction models.

This script loads a trained model and applies it to new data for batch correction.
It handles both AE and VAE+Attention model types.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Import model classes and utilities from the main script
from NN_batch_correct import (
    library_size_normalize,
    inverse_standardize,
    AEBatchCorrector,
    ResidualBlock,
    make_mlp,
    GradReverse,
    GradReverseLayer
)

try:
    from vae_attention_model import VaeAttentionBatchCorrector
except Exception:
    VaeAttentionBatchCorrector = None


def load_model_checkpoint(model_path: Path):
    """Load model checkpoint and return components."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    required_keys = ['state_dict', 'scaler', 'hvg_genes']
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing required key: {key}")
    
    return checkpoint


def preprocess_new_data(counts_df: pd.DataFrame, hvg_genes: list, scaler: StandardScaler, 
                       genes_in_rows: bool = False) -> np.ndarray:
    """
    Preprocess new count data for inference.
    
    Args:
        counts_df: Raw count matrix
        hvg_genes: List of highly variable genes the model was trained on
        scaler: Fitted StandardScaler from training
        genes_in_rows: If True, transpose the data (genes in rows instead of columns)
        
    Returns:
        Preprocessed data ready for model inference
    """
    # Transpose if genes are in rows
    if genes_in_rows:
        counts_df = counts_df.T
    
    # Library size normalization -> logCPM
    print(f"[INFO] Input data shape: {counts_df.shape}")
    logcpm_df = library_size_normalize(counts_df)
    
    # Filter for HVG genes that were used in training
    missing_genes = [g for g in hvg_genes if g not in logcpm_df.columns]
    if missing_genes:
        print(f"[WARNING] {len(missing_genes)} genes from training not found in new data. "
              f"These will be filled with zeros.")
        
        # Add missing genes as zero columns
        for gene in missing_genes:
            logcpm_df[gene] = 0.0
    
    # Reorder columns to match training order
    logcpm_df = logcpm_df.reindex(columns=hvg_genes, fill_value=0.0)
    
    print(f"[INFO] After HVG filtering: {logcpm_df.shape}")
    
    # Apply the same standardization as training
    X_standardized = scaler.transform(logcpm_df.values)
    
    return X_standardized


def reconstruct_model_from_checkpoint(checkpoint: dict) -> nn.Module:
    """Reconstruct model from checkpoint state_dict."""
    state_dict = checkpoint['state_dict']
    
    # Try to infer model type from state_dict keys
    if any('attention' in key for key in state_dict.keys()):
        if VaeAttentionBatchCorrector is None:
            raise RuntimeError("VAE+Attention model detected but vae_attention_model not available")
        
        # For VAE models, we need to reconstruct with the right parameters
        # This is tricky without the original args, so we'll make reasonable defaults
        num_genes = len(checkpoint['hvg_genes'])
        num_batches = len(checkpoint['batch_classes'])
        
        # Try to infer dimensions from state_dict
        model = VaeAttentionBatchCorrector(
            num_genes=num_genes,
            num_batches=num_batches,
            latent_dim=32,  # Default, may need adjustment
            hidden_dim=128,  # Default
            attention_dim=64,  # Default
            n_heads=4,  # Default
            dropout=0.1,
        )
    else:
        # AE model
        num_genes = len(checkpoint['hvg_genes'])
        num_batches = len(checkpoint['batch_classes'])
        num_labels = len(checkpoint['label_classes']) if checkpoint['label_classes'] else None
        
        # Try to infer dimensions from state_dict
        # Look for encoder layers
        enc_dims = []
        for key in state_dict.keys():
            if 'encoder' in key and 'weight' in key:
                layer_idx = int(key.split('.')[1])
                if layer_idx == 0:  # First encoder layer
                    enc_dims.append(state_dict[key].shape[0])
        
        model = AEBatchCorrector(
            n_genes=num_genes,
            latent_dim=32,  # Will be corrected by load_state_dict
            n_batches=num_batches,
            n_labels=num_labels
        )
    
    model.load_state_dict(state_dict)
    return model


def run_inference(model: nn.Module, X_input: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Run model inference and return corrected data."""
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
        
        # Handle different model types
        if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
            # AE model - get reconstruction
            x_hat, _, _, _ = model(X_tensor, adv_lambda=0.0)  # No adversarial during inference
            corrected = x_hat.cpu().numpy()
        else:
            # VAE model - get reconstruction mean
            recon_mu, _, _, _, _, _, _ = model(X_tensor)
            corrected = recon_mu.cpu().numpy()
    
    return corrected


def main():
    parser = argparse.ArgumentParser(description='Apply trained HarmonizeNN model for batch correction')
    parser.add_argument('--model_path', type=Path, required=True,
                       help='Path to saved model checkpoint (.pt file)')
    parser.add_argument('--counts_path', type=Path, required=True,
                       help='Path to new counts matrix (CSV)')
    parser.add_argument('--out_path', type=Path, required=True,
                       help='Path for corrected output (CSV)')
    parser.add_argument('--genes_in_rows', action='store_true',
                       help='If set, genes are in rows instead of columns')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Check if CUDA is available when requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available, using CPU")
        args.device = 'cpu'
    
    print(f"[INFO] Loading model from {args.model_path}")
    checkpoint = load_model_checkpoint(args.model_path)
    
    print(f"[INFO] Loading counts data from {args.counts_path}")
    counts_df = pd.read_csv(args.counts_path, index_col=0)
    
    print(f"[INFO] Preprocessing data...")
    X_input = preprocess_new_data(
        counts_df, 
        checkpoint['hvg_genes'], 
        checkpoint['scaler'], 
        args.genes_in_rows
    )
    
    print(f"[INFO] Reconstructing model...")
    model = reconstruct_model_from_checkpoint(checkpoint)
    
    print(f"[INFO] Running inference...")
    corrected = run_inference(model, X_input, args.device)
    
    print(f"[INFO] Inverse-transforming to logCPM scale...")
    corrected_logcpm = inverse_standardize(corrected, checkpoint['scaler'])
    
    # Create output DataFrame with original sample names and HVG gene names
    sample_names = counts_df.index if not args.genes_in_rows else counts_df.columns
    corrected_df = pd.DataFrame(
        corrected_logcpm, 
        index=sample_names, 
        columns=checkpoint['hvg_genes']
    )
    
    print(f"[INFO] Saving corrected data to {args.out_path}")
    corrected_df.to_csv(args.out_path)
    
    print(f"[OK] Batch correction complete!")
    print(f"[OK] Input shape: {counts_df.shape}")
    print(f"[OK] Output shape: {corrected_df.shape}")


if __name__ == "__main__":
    main()