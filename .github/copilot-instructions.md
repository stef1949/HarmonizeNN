# HarmonizeNN - Neural Network Batch Correction for Bulk RNA-seq

Always reference these instructions first and fallback to search or additional context gathering only when you encounter unexpected information that does not match the info here.

## Project Overview
HarmonizeNN is a Python-based neural network tool for bulk RNA-seq batch effect correction using adversarial autoencoders. The tool supports both standard autoencoder (AE) and VAE+Attention models with optional negative binomial loss for count data.

## Working Effectively

### Environment Setup & Dependencies
- **Python Requirements**: Python 3.10+ (tested with 3.10, 3.11, 3.12, 3.13)
- **Install dependencies**: `pip install -r requirements.txt` - takes ~2.5 minutes, NEVER CANCEL
- **Install testing framework**: `pip install pytest` - takes ~3 seconds
- **Key dependencies**: PyTorch (CPU/CUDA), pandas, scikit-learn, matplotlib, wandb (optional), shap

### Build & Test Process
- **Run unit tests**: `pytest -v` - takes ~7 seconds, runs 8 tests across 3 test files
- **GitHub Actions CI**: Uses pytest, tests on Python 3.10-3.13, includes offline W&B mode
- **No separate build step required** - Python project with direct script execution

### Core Training & Correction Commands
**Quick validation run** (2 seconds):
```bash
python NN_batch_correct.py --counts bulk_counts.csv --metadata sample_meta.csv --genes_in_rows --epochs 2 --batch_size 8 --hvg 1000 --out_corrected test_output.csv --patience 999
```

**Standard training run** (10 seconds for 20 epochs):
```bash
python NN_batch_correct.py \
  --counts bulk_counts.csv \
  --metadata sample_meta.csv \
  --genes_in_rows \
  --label_col condition \
  --hvg 5000 \
  --epochs 20 \
  --batch_size 32 \
  --latent_dim 32 \
  --enc_hidden 1024,256 \
  --dec_hidden 256,1024 \
  --adv_hidden 128 \
  --sup_hidden 64 \
  --adv_lambda_schedule adaptive \
  --out_corrected corrected_output.csv \
  --out_latent latent_output.csv \
  --save_model trained_model.pt \
  --generate_viz
```

**With Weights & Biases logging** (26 seconds):
```bash
WANDB_MODE=offline WANDB_SILENT=true python NN_batch_correct.py \
  --counts bulk_counts.csv \
  --metadata sample_meta.csv \
  --genes_in_rows \
  --epochs 10 \
  --use_wandb \
  --out_corrected corrected_wandb.csv
```

### Visualization
**Generate PCA plots and boxplots** (3 seconds):
```bash
python visualise.py \
  --counts bulk_counts.csv \
  --metadata sample_meta.csv \
  --genes_in_rows \
  --corrected corrected_output.csv \
  --hvg_top 2000
```

### Performance & Timing Expectations
- **Dependency installation**: 2.5 minutes (140 seconds) - NEVER CANCEL, set timeout to 300+ seconds
- **Unit tests**: 7 seconds - set timeout to 30+ seconds  
- **Quick training (2-3 epochs)**: 3-5 seconds
- **Standard training (10-20 epochs)**: 8-12 seconds  
- **Full training (50+ epochs)**: 15-30 seconds (early stopping often triggers)
- **Visualization generation**: 3 seconds
- **W&B logging adds overhead**: ~3x longer execution time

## Validation Scenarios

### Always Test These User Scenarios After Changes:
1. **Basic training workflow**: Run quick training with example data, verify output files are created
2. **Visualization workflow**: Generate PCA plots and boxplots using example or corrected data
3. **Unit test suite**: Run `pytest -v` to ensure no regressions in core functionality
4. **Help system**: Verify `python NN_batch_correct.py -h` shows complete help

### Manual Validation Steps:
- **Check output files exist**: corrected CSV, latent CSV (if requested), model PT file (if requested)
- **Verify output format**: CSV files should have samples as rows, genes/latent dimensions as columns
- **Validate visualizations**: PNG files should be generated (PCA plots, boxplots, architecture diagram)
- **Monitor training progress**: Should show epoch progress with loss values and accuracy metrics

## Key Files & Components

### Main Scripts
- **`NN_batch_correct.py`** - Main training and correction script with full argument parser
- **`visualise.py`** - Standalone visualization script for PCA plots and boxplots  
- **`vae_attention_model.py`** - VAE+Attention model implementation (has parameter compatibility issues)

### Data Files (Examples Provided)
- **`bulk_counts.csv`** - Example counts matrix (genes × samples, ~13MB)
- **`sample_meta.csv`** - Example metadata with sample, batch, condition columns
- **Test files**: `test_counts.csv`, `test_meta.csv` for minimal testing

### Configuration & Testing
- **`requirements.txt`** - Python dependencies specification
- **`tests/`** - Unit test directory with pytest configuration
- **`.github/workflows/unit-tests.yml`** - GitHub Actions CI pipeline

### PowerShell Scripts (Windows)
- **`run_training.ps1`** - Example Windows training script
- **`run_wandb_*.ps1`** - W&B sweep scripts for hyperparameter tuning

## Common Issues & Workarounds

### Known Limitations
- **VAE+Attention model (`--model_type vae_attention`)**: Has parameter compatibility issues with `attn_max_tokens` argument. Use standard AE model for reliable results.
- **CUDA acceleration**: AMP (`--amp`) and CUDA features work but provide minimal benefit on CPU-only systems
- **R script**: `generate_synthetic_bulk_rna.R` requires R installation (not available in standard environments)

### Environment-Specific Notes
- **Offline W&B**: Always set `WANDB_MODE=offline WANDB_SILENT=true` in CI/testing environments
- **MPLBACKEND**: Set to `Agg` for headless matplotlib rendering in CI
- **CUDA**: Check availability with `python CUDACheck.py`

### Debugging Commands
- **Check CUDA**: `python CUDACheck.py`
- **Test basic functionality**: Use minimal epochs (2-3) and small batch sizes (8-16) for quick validation
- **View help**: `python NN_batch_correct.py -h` for complete argument reference

## Development Workflow

### Before Making Changes
1. **Install dependencies**: `pip install -r requirements.txt` (2.5 min timeout)
2. **Run baseline tests**: `pytest -v` (30 sec timeout)  
3. **Test basic functionality**: Quick training run to ensure pipeline works

### After Making Changes
1. **Run unit tests**: `pytest -v` - ensure no test regressions
2. **Validate training pipeline**: Run quick test with 2-3 epochs
3. **Test complete workflow**: Full training run with visualization
4. **Check output consistency**: Verify file formats and content structure

### CI Integration
- **GitHub Actions**: Automatically runs on push/PR with Python 3.10-3.13 matrix
- **Test environment**: Ubuntu with offline W&B, headless matplotlib, no CUDA
- **Timeout considerations**: CI uses 120s default timeout for training commands

## Project Structure Reference

### Repository Root Contents
```
├── NN_batch_correct.py          # Main training script
├── visualise.py                 # Visualization utilities  
├── vae_attention_model.py       # VAE+Attention model
├── requirements.txt             # Python dependencies
├── bulk_counts.csv              # Example data (13MB)
├── sample_meta.csv              # Example metadata
├── tests/                       # Unit tests (pytest)
├── .github/workflows/           # CI configuration
├── *.ps1                        # Windows PowerShell scripts
└── generate_synthetic_bulk_rna.R # R synthetic data generator
```

### Generated Output Files
- **Corrected data**: `corrected_*.csv` (logCPM scale, samples × genes)
- **Latent embeddings**: `latent_*.csv` (samples × latent_dimensions)  
- **Trained models**: `*.pt` (PyTorch state dict + metadata)
- **Visualizations**: `pca_*.png`, `logCPM_boxplots.png`, `nn_architecture.png`

## Quick Reference Commands

**Fast validation** (5 seconds total):
```bash
pip install -r requirements.txt && pytest -v && python NN_batch_correct.py --counts bulk_counts.csv --metadata sample_meta.csv --genes_in_rows --epochs 2 --batch_size 8 --hvg 1000 --out_corrected test.csv --patience 999
```

**Complete workflow test** (15 seconds total):
```bash
python NN_batch_correct.py --counts bulk_counts.csv --metadata sample_meta.csv --genes_in_rows --label_col condition --epochs 10 --hvg 5000 --out_corrected output.csv --out_latent latent.csv --generate_viz
```

**Clean up test files**:
```bash
rm -f test*.csv corrected_*.csv latent_*.csv *.pt
```

Always use these exact commands and timeout values to ensure reliable operation in any environment.