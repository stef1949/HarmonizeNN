<#!
Run / manage a Weights & Biases sweep for NN_batch_correct.
Usage examples:
  # (Recommended) Set API key once per session:
  $Env:WANDB_API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
  pwsh .\run_wandb_sweep.ps1 -Entity myuser -Project nn-batch-correction

  # Provide key inline (NOT recommended; shows in history / process list):
  pwsh .\run_wandb_sweep.ps1 -Entity myuser -ApiKey 'xxxxxxxx' -Agents 2

  # Resume existing sweep
  pwsh .\run_wandb_sweep.ps1 -Resume -SweepId myuser/nn-batch-correction/abc123

Params:
  -Entity <string>   : W&B entity / team (defaults to $Env:WANDB_ENTITY if set)
  -Project <string>  : W&B project (default nn-batch-correction)
  -Yaml <path>       : Sweep YAML file (default wandb_sweep.yaml)
  -Agents <int>      : Number of agents to spawn (default 1)
  -MaxRuns <int>     : Optional cap; agent(s) stop after this many completed runs
  -Resume            : Skip creation; requires -SweepId
  -SweepId <string>  : Existing sweep ID (short or full)
  -ApiKey <string>   : (Optional) W&B API key (prefer environment variable instead)

Security: DO NOT commit or paste real API keys into version control. Revoke exposed keys via W&B settings.
Requires: wandb CLI (pip install wandb).
#>

# Activate local venv if present
if (Test-Path .venv\Scripts\Activate.ps1) {
    . .venv\Scripts\Activate.ps1
}

# Ensure project name for W&B (change if desired)
if (-not $Env:WANDB_PROJECT) { $Env:WANDB_PROJECT = 'nn-batch-correction' }

param(
  [string]$Entity = $Env:WANDB_ENTITY,
  [string]$Project = 'nn-batch-correction',
  [string]$Yaml = 'wandb_sweep.yaml',
  [int]$Agents = 1,
  [int]$MaxRuns = 0,
  [switch]$Resume,
  [string]$SweepId,
  [string]$ApiKey
)

$ErrorActionPreference = 'Stop'
if (-not (Get-Command wandb -ErrorAction SilentlyContinue)) { throw 'wandb CLI not found. pip install wandb' }

# Inject API key if provided (warn about security)
if ($ApiKey) {
  Write-Warning 'Passing API key via parameter exposes it in process list & history; prefer $Env:WANDB_API_KEY.'
  $Env:WANDB_API_KEY = $ApiKey
}
if (-not $Entity) { throw 'Entity not provided (set -Entity or $Env:WANDB_ENTITY)' }
if ($Entity -eq 'your_entity') { throw "Replace -Entity your_entity with your actual W&B username or team name." }
if (-not (Test-Path $Yaml)) { throw "YAML file not found: $Yaml" }

# Activate venv if present
if (Test-Path .venv\Scripts\Activate.ps1) { . .venv\Scripts\Activate.ps1 }

function Get-SweepIdFromOutput($text) {
  # Handles lines like:
  #  'Created sweep with ID: abc123'
  #  'Created sweep with ID: user/project/abc123'
  $pattern = 'Created sweep with ID:\s*(.+)$'
  foreach ($line in ($text -split "`n")) {
    $m = [regex]::Match($line, $pattern)
    if ($m.Success) { return $m.Groups[1].Value.Trim() }
  }
  return $null
}

function Test-WandbAuth {
  try {
    $info = wandb whoami 2>&1 | Out-String
    $exit = $LASTEXITCODE
    if ($exit -ne 0) { Write-Host "whoami exit code $exit" -ForegroundColor DarkGray; return $false }
    Write-Host "wandb whoami output:`n$info" -ForegroundColor DarkGray
    # Accept if it shows an entity/user or 'You are logged in'
    if ($info -match 'Logged in as' -or $info -match 'entity' -or $info -match 'username' -or $info -match 'You are logged in') { return $true }
    if ($info -match 'Not logged in') { return $false }
    # Fallback: if API key env present, assume ok
    if ($Env:WANDB_API_KEY) { return $true }
    return $false
  } catch { return $false }
}

if (-not (Test-WandbAuth)) {
  if (-not $Env:WANDB_API_KEY) {
    throw "Not authenticated with W&B. Run 'wandb login' or set $Env:WANDB_API_KEY first (or use -ApiKey)."
  }
  Write-Host 'Attempting non-interactive login with provided API key...' -ForegroundColor Cyan
  $loginOut = wandb login --relogin $Env:WANDB_API_KEY 2>&1 | Out-String
  Write-Host "wandb login output:`n$loginOut" -ForegroundColor DarkGray
  if ($loginOut -match 'Invalid') { throw "wandb login failed. Output:`n$loginOut" }
  if (-not (Test-WandbAuth)) {
    Write-Warning 'Authentication still uncertain; proceeding because API key is set. Creation may fail if permissions missing.'
  }
}

if (-not $Resume) {
  Write-Host "Creating new sweep from $Yaml ..."
  $createOut = wandb sweep --entity $Entity --project $Project $Yaml 2>&1 | Out-String
  if ($createOut -match 'permission denied') {
    throw "Permission denied creating sweep. Check: (1) WANDB_API_KEY is set & 'wandb login' succeeded, (2) you have 'create' rights for entity '$Entity', (3) entity name is correct. Raw output:`n$createOut"
  }
  $sweepLocalId = Get-SweepIdFromOutput $createOut
  if (-not $sweepLocalId) { throw "Failed to parse sweep ID. Output:`n$createOut" }
  Write-Host "Sweep ID: $sweepLocalId" -ForegroundColor Green
  $SweepId = $sweepLocalId
} elseif (-not $SweepId) {
  throw 'Resume specified but no -SweepId provided.'
}

# Normalise SweepId (allow short form <hash>)
if ($SweepId -notmatch '/') { $SweepId = "$Entity/$Project/$SweepId" }
Write-Host "Using sweep: $SweepId" -ForegroundColor Cyan

# Build agent command base
$agentCmd = "wandb agent $SweepId"
if ($MaxRuns -gt 0) { $agentCmd = "$agentCmd --max-runs $MaxRuns" }

Write-Host "Launching $Agents agent(s) ..." -ForegroundColor Cyan

$jobs = @()
for ($i = 1; $i -le $Agents; $i++) {
  $jobs += Start-Job -ScriptBlock { param($cmd) & cmd /c $cmd } -ArgumentList $agentCmd
  Start-Sleep -Seconds 2  # slight stagger
}

Write-Host 'Agents started. Use Get-Job / Receive-Job to inspect logs. Waiting for completion...' -ForegroundColor Yellow

# Tail status loop
while (@($jobs | Where-Object { $_.State -eq 'Running' }).Count -gt 0) {
  Start-Sleep -Seconds 30
  $running = @($jobs | Where-Object { $_.State -eq 'Running' }).Count
  Write-Host (Get-Date -Format HH:mm:ss) "Running agents: $running" -ForegroundColor DarkGray
}

foreach ($j in $jobs) {
  Write-Host "--- Agent job output (ID=$($j.Id)) ---" -ForegroundColor Magenta
  Receive-Job $j
  Remove-Job $j
}

Write-Host 'Sweep run(s) finished.' -ForegroundColor Green
