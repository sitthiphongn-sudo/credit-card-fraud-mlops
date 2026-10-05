# One command: raw data -> validate -> train -> register -> gate -> champion -> serving
# Usage (from repo root):  powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
$ErrorActionPreference = "Continue"  # native tools write progress to stderr; exit codes are checked below
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# Use the project venv if it exists
$py = "python"
if (Test-Path ".\.venv\Scripts\python.exe") { $py = (Resolve-Path ".\.venv\Scripts\python.exe").Path }
$env:PYTHONUTF8 = "1"

Step "0/5 Check prerequisites"
if (-not (Test-Path "data/raw/creditcard.csv")) { throw "data/raw/creditcard.csv not found - download it first (see README)" }
docker info 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker is not running - start Docker Desktop first" }

Step "1/5 Start MLflow"
docker compose up -d mlflow
if ($LASTEXITCODE -ne 0) { throw "docker compose up mlflow failed" }
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
  try { Invoke-WebRequest -UseBasicParsing "$env:MLFLOW_TRACKING_URI/health" -TimeoutSec 2 -ErrorAction Stop | Out-Null; $ready = $true; break }
  catch { Start-Sleep -Seconds 2 }
}
if (-not $ready) { throw "MLflow did not become ready on $env:MLFLOW_TRACKING_URI" }

Step "2/5 Run pipeline (ingest -> check -> split -> train -> size -> register -> promote)"
& $py pipelines/flow.py
if ($LASTEXITCODE -ne 0) { throw "Pipeline failed (exit $LASTEXITCODE)" }

Step "3/5 Start serving + monitoring (API, Prometheus, Grafana)"
docker compose up -d --build
if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }
# restart API so it loads the newest @champion
docker compose restart api | Out-Null

Step "4/5 Wait for API health"
$health = $null
for ($i = 0; $i -lt 60; $i++) {
  try { $health = Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction Stop; break }
  catch { Start-Sleep -Seconds 2 }
}
if ($null -eq $health) { throw "API did not become healthy on http://localhost:8000/health" }

Step "5/5 Done"
$health | ConvertTo-Json
Write-Host "MLflow     http://localhost:5001"
Write-Host "API docs   http://localhost:8000/docs"
Write-Host "Prometheus http://localhost:9090"
Write-Host "Grafana    http://localhost:3000"
