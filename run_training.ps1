<#
Run training for NN batch correction.
Usage: pwsh .\run_training.ps1
Optional: set $Env:WANDB_API_KEY beforehand for online Weights & Biases logging.
#>

$ErrorActionPreference = 'Stop'

# Activate local venv if present
if (Test-Path .venv\Scripts\Activate.ps1) {
    . .venv\Scripts\Activate.ps1
}

# Ensure project name for W&B (change if desired)
if (-not $Env:WANDB_PROJECT) { $Env:WANDB_PROJECT = 'nn-batch-correction' }

# Toggle offline mode easily (uncomment to force offline)
# $Env:WANDB_MODE = 'offline'

# Core run command
python NN_batch_correct.py `
  --counts bulk_counts.csv `
  --metadata sample_meta.csv `
  --genes_in_rows `
  --sample_col sample `
  --batch_col batch `
  --label_col condition `
  --hvg 10000 `
  --epochs 200 `
  --latent_dim 32 `
  --enc_hidden 1024,256 `
  --dec_hidden 256,1024 `
  --adv_hidden 128 `
  --sup_hidden 64 `
  --lr 3e-4 `
  --batch_size 16 `
  --scheduler cosine `
  --weight_decay 1e-4 `
  --amp `
  --num_workers 4 `
  --pin_memory `
  --adv_lambda_schedule adaptive `
  --grad_accum 1 `
  --out_corrected corrected_logcpm.csv `
  --out_latent latent.csv `
  --use_wandb `
  --wandb_log gradients `
  --wandb_log_freq 1 `
  --generate_viz

if ($LASTEXITCODE -ne 0) {
  Write-Error "Training failed with exit code $LASTEXITCODE"
} else {
  Write-Host "Training completed successfully." -ForegroundColor Green
}
