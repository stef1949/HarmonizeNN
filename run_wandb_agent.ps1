<#!
Launch a single W&B sweep agent (helper script if you prefer manual control per GPU).
Usage:
  pwsh .\run_wandb_agent.ps1 -SweepId <entity/project/hash> [-MaxRuns 20]
#>

param(
  [Parameter(Mandatory=$true)][string]$SweepId,
  [int]$MaxRuns = 0,
  [int]$NumAgents = 1
)

# Activate local venv if present (must happen after param block in PowerShell scripts)
if (Test-Path .venv\Scripts\Activate.ps1) { . .venv\Scripts\Activate.ps1 }

# Ensure project name for W&B (change if desired)
if (-not $Env:WANDB_PROJECT) { $Env:WANDB_PROJECT = 'nn-batch-correction' }
$ErrorActionPreference = 'Stop'
if (-not (Get-Command wandb -ErrorAction SilentlyContinue)) { throw 'wandb CLI not found. pip install wandb' }
if (Test-Path .venv\Scripts\Activate.ps1) { . .venv\Scripts\Activate.ps1 }

if ($SweepId -notmatch '/') {
  if ($Env:WANDB_ENTITY -and $Env:WANDB_PROJECT) {
    Write-Host "[INFO] Expanding short sweep id '$SweepId' with entity/project" -ForegroundColor Cyan
    $SweepId = "$Env:WANDB_ENTITY/$Env:WANDB_PROJECT/$SweepId"
  } else {
    throw 'Provide full sweep id entity/project/hash OR set WANDB_ENTITY & WANDB_PROJECT environment variables.'
  }
}
if ($NumAgents -lt 1) { throw 'NumAgents must be >= 1' }

$baseCmd = "wandb agent $SweepId"
if ($MaxRuns -gt 0) { $baseCmd = "$baseCmd --max-runs $MaxRuns" }

if ($NumAgents -eq 1) {
  Write-Host "Running: $baseCmd" -ForegroundColor Cyan
  & $baseCmd
  return
}

Write-Host "Launching $NumAgents parallel agents..." -ForegroundColor Cyan
if (-not (Test-Path logs)) { New-Item -ItemType Directory -Path logs | Out-Null }
for ($i=1; $i -le $NumAgents; $i++) {
  $cmd = $baseCmd
  $log = Join-Path logs ("agent_${i}.log")
  Write-Host "  Agent $i -> $cmd (log: $log)" -ForegroundColor Yellow
  Start-Process pwsh -ArgumentList "-NoLogo","-NoProfile","-Command","$cmd *>&1 | Tee-Object -FilePath '$log'" -WindowStyle Minimized
  Start-Sleep -Seconds 2  # slight stagger to avoid simultaneous sweep reservation collisions
}
Write-Host "All agents started. Use 'Get-Content -Wait logs/agent_1.log' to tail a log." -ForegroundColor Green
