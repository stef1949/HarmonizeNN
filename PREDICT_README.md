# HarmonizeNN Inference Script (predict.py)

## Overview
`predict.py` is a standalone inference script for applying trained HarmonizeNN models to new, unseen data. It handles the complete pipeline from raw counts to batch-corrected expression values.

## Requirements
- Python 3.7+
- PyTorch
- pandas
- numpy
- scikit-learn

## Usage

### Basic Usage
```bash
python predict.py --model_path MODEL.pt --counts_path DATA.csv --out_path OUTPUT.csv
```

### Command-line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model_path` | ✅ | Path to trained model checkpoint (.pt file) |
| `--counts_path` | ✅ | Path to new counts matrix (CSV format) |
| `--out_path` | ✅ | Path for corrected output (CSV format) |
| `--genes_in_rows` | ❌ | Set if genes are in rows instead of columns |
| `--device` | ❌ | Device for inference: 'cpu' or 'cuda' (default: 'cpu') |

## Input Data Format

### Default Format (Samples × Genes)
```csv
,GENE1,GENE2,GENE3,...
SAMPLE1,100,250,75,...
SAMPLE2,150,200,90,...
```

### Genes in Rows Format (use `--genes_in_rows`)
```csv
,SAMPLE1,SAMPLE2,SAMPLE3,...
GENE1,100,150,200,...
GENE2,250,200,180,...
```

## Processing Pipeline

1. **Load Model**: Reads checkpoint containing:
   - Model state_dict
   - StandardScaler fitted on training data  
   - Highly Variable Genes (HVG) list
   - Batch and label class mappings

2. **Data Preprocessing**:
   - Library size normalization (CPM → log1p)
   - Filter/reorder genes to match training HVGs
   - Apply z-score standardization using training scaler

3. **Model Inference**:
   - Run batch correction in evaluation mode
   - Handle both AE and VAE model architectures

4. **Output Processing**:
   - Inverse z-score transformation back to logCPM scale
   - Save corrected expression matrix

## Examples

### Example 1: Basic Inference
```bash
# Apply trained model to test data
python predict.py \
    --model_path artifacts/checkpoints/best_model.pt \
    --counts_path data/test_counts.csv \
    --out_path results/corrected_test.csv
```

### Example 2: With GPU Acceleration
```bash
# Use GPU for faster inference on large datasets
python predict.py \
    --model_path models/residual_model.pt \
    --counts_path data/large_dataset.csv \
    --out_path results/corrected_large.csv \
    --device cuda
```

### Example 3: Transposed Input
```bash
# Handle data where genes are in rows
python predict.py \
    --model_path artifacts/checkpoints/trained_model.pt \
    --counts_path data/transposed_counts.csv \
    --out_path results/corrected_output.csv \
    --genes_in_rows
```

## Output

The script produces a CSV file with:
- **Rows**: Same samples as input
- **Columns**: Highly Variable Genes used in training
- **Values**: Batch-corrected logCPM expression values
- **Format**: Same orientation as input data

## Error Handling

The script provides informative error messages for common issues:

- **Missing model file**: Clear path validation
- **Incompatible data**: Gene mismatch warnings with zero-filling
- **Memory issues**: Automatic fallback to CPU if CUDA unavailable
- **Format errors**: Data shape and format validation

## Performance Tips

1. **Memory**: For large datasets, use CPU inference to avoid GPU memory limits
2. **Speed**: Use `--device cuda` for faster processing when available
3. **Batch Size**: Process very large datasets in chunks if memory limited

## Integration with Training

The inference script automatically handles models trained with:
- ✅ Standard autoencoder architecture
- ✅ VAE with attention mechanisms  
- ✅ Residual connections (new feature)
- ✅ Different loss functions (MSE, MAE, Huber, NB)

## Troubleshooting

### Common Issues

1. **"Checkpoint missing required key"**
   - Ensure model was saved with enhanced checkpoint format
   - Re-train model with updated `NN_batch_correct.py`

2. **"X genes from training not found"**  
   - Normal warning when new data has different gene sets
   - Missing genes are automatically zero-filled

3. **"CUDA requested but not available"**
   - Automatic fallback to CPU with warning message
   - Ensure PyTorch was installed with CUDA support

### Debug Mode
Add debug prints by modifying the script to show:
```python
print(f"Input shape: {counts_df.shape}")
print(f"After HVG filtering: {X_input.shape}") 
print(f"Model output shape: {corrected.shape}")
```

## Version Compatibility

- Compatible with models trained using the refactored `NN_batch_correct.py`
- Requires enhanced checkpoint format (includes scaler and hvg_genes)
- Backward compatible with existing model architectures