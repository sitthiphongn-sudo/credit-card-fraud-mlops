<#
verify_serving.ps1 — ตรวจงาน Serving ของสหรัฐแบบครบวงจร และเก็บหลักฐานไว้ที่ reports/serving/evidence/

รัน (PowerShell ที่ root ของ repo):
    powershell -ExecutionPolicy Bypass -File reports\serving\verify_serving.ps1

ทำอะไร (ไม่แก้ไฟล์ของสมาชิกคนอื่น และไม่แก้ docker-compose.yml):
  1. pytest tests/test_api.py + ruff (ใช้ Python บนเครื่อง)
  2. docker info / docker build -f serving/Dockerfile
  3. เปิด MLflow ชั่วคราวด้วย docker run (ค่าตามที่ อัปงาน.pdf หัวข้อ 6 เสนอ: sqlite + --artifacts-destination)
     ใช้ named volume "fraud-serving-mlflow" จึงไม่สร้างไฟล์ใน repo
  4. รัน pipeline ของทีม (python pipelines/flow.py) "ใน image เดียวกับ API" เพื่อให้ได้ champion
     แล้วคืนไฟล์ reports/experiments/* ที่ pipeline เขียนทับกลับเป็นของ git (ถ้าก่อนรันไม่มีการแก้ค้าง)
  5. ตรวจว่า champion = lightgbm_none และ threshold มาจาก run (ถ้าไม่ใช่จะหยุดก่อน load test)
  6. รัน API container แล้วเก็บ curl ของ /health /predict /metrics และข้อมูลเสีย -> 400
  7. load test concurrency 1/10/100 (WEB_CONCURRENCY=1 และ 2) -> reports/serving/load_test_champion*.json

พารามิเตอร์:
  -SkipPipeline   ข้ามขั้น 4 (ใช้ champion ที่มีอยู่แล้วใน volume จากการรันครั้งก่อน)
  -Requests 1000  จำนวน request ต่อระดับ concurrency
  -ApiPort 8000   port ของ API บนเครื่อง (ต้องว่าง ถ้า docker compose api รันอยู่ให้ docker compose stop api ก่อน)
  -Keep           ไม่ลบ container หลังจบ (API ยังเปิดอยู่ที่ localhost:ApiPort)
#>
param(
    [switch]$SkipPipeline,
    [int]$Requests = 1000,
    [int]$ApiPort = 8000,
    [int]$MlflowPort = 5050,
    [switch]$Keep
)

$ErrorActionPreference = "Continue"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Evidence = "reports/serving/evidence"
$Payloads = "$Evidence/payloads"
New-Item -ItemType Directory -Force -Path $Payloads | Out-Null

$Net = "fraud-serving-net"
$Vol = "fraud-serving-mlflow"
$MlflowName = "fraud-mlflow-verify"
$ApiName = "fraud-api-verify"
$Image = "fraud-api:serving-verify"
$MlflowImage = "ghcr.io/mlflow/mlflow:v2.22.1"
$Api = "http://localhost:$ApiPort"
$Utf8 = New-Object System.Text.UTF8Encoding($false)
$Summary = New-Object System.Collections.ArrayList

function Save-Text([string]$Name, [string]$Text) {
    [System.IO.File]::WriteAllText((Join-Path $RepoRoot "$Evidence/$Name"), $Text, $Utf8)
}

function Note([string]$Line) {
    Write-Host $Line -ForegroundColor Yellow
    [void]$Summary.Add($Line)
}

# รันคำสั่ง แสดงผลสด และบันทึกลงไฟล์ evidence/<Name>.txt พร้อม exit code
function Run-Step([string]$Name, [string]$Shown, [scriptblock]$Block) {
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    Write-Host "> $Shown" -ForegroundColor DarkGray
    $lines = New-Object System.Collections.ArrayList
    $global:LASTEXITCODE = 0
    & $Block 2>&1 | ForEach-Object { $s = "$_"; Write-Host $s; [void]$lines.Add($s) }
    $code = $LASTEXITCODE
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss zzz")
    Save-Text "$Name.txt" ("# $stamp`r`n> $Shown`r`n# exit=$code`r`n" + ($lines -join "`r`n") + "`r`n")
    return $code
}

function Wait-Http([string]$Url, [int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        $code = & curl.exe -s -o NUL -w "%{http_code}" $Url 2>$null
        if ($code -eq "200") { return $true }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Remove-Container([string]$Name) {
    & docker rm -f $Name 2>&1 | Out-Null
}

function Start-Api([int]$Workers) {
    Remove-Container $ApiName
    $code = Run-Step "docker_run_api_workers$Workers" "docker run -d --name $ApiName --network $Net -p ${ApiPort}:8000 -e MLFLOW_TRACKING_URI=http://${MlflowName}:5000 -e WEB_CONCURRENCY=$Workers $Image" {
        docker run -d --name $ApiName --network $Net -p "${ApiPort}:8000" `
            -e "MLFLOW_TRACKING_URI=http://${MlflowName}:5000" -e "WEB_CONCURRENCY=$Workers" $Image
    }
    if ($code -ne 0) { return $false }
    Write-Host "Waiting for $Api/health ..."
    if (-not (Wait-Http "$Api/health" 240)) {
        & docker logs --tail 100 $ApiName 2>&1 | Out-String | Write-Host
        return $false
    }
    return $true
}

# ---------------------------------------------------------------------------
# 0. สภาพแวดล้อม + git
# ---------------------------------------------------------------------------
Run-Step "env" "python --version; docker --version; git branch/status" {
    python --version
    docker --version
    git rev-parse --abbrev-ref HEAD
    git log -1 --oneline
    git status --short
} | Out-Null

# ---------------------------------------------------------------------------
# 1. Unit test + lint
# ---------------------------------------------------------------------------
$c = Run-Step "pytest" "python -m pytest tests/test_api.py -q" { python -m pytest tests/test_api.py -q -p no:cacheprovider }
Note ("pytest tests/test_api.py: " + $(if ($c -eq 0) { "PASS" } else { "FAIL (exit $c)" }))
$c = Run-Step "ruff" "ruff check serving tests/test_api.py" { python -m ruff check serving tests/test_api.py }
Note ("ruff check serving tests/test_api.py: " + $(if ($c -eq 0) { "PASS" } else { "FAIL (exit $c)" }))

# ---------------------------------------------------------------------------
# 2. Docker engine + build
# ---------------------------------------------------------------------------
Run-Step "docker_info" "docker info" { docker info --format "Server {{.ServerVersion}} | OS {{.OperatingSystem}} | CPUs {{.NCPU}} | Mem {{.MemTotal}}" } | Out-Null
# docker info คืน exit 0 แม้ Engine ตอบ 500 จึงเช็ก Server version จริงอีกชั้น
$serverVersion = (& docker version --format "{{.Server.Version}}" 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrEmpty($serverVersion) -or $serverVersion -notmatch "^\d") {
    Note "Docker Engine ไม่พร้อม (docker version ไม่ได้ Server version) -> Quit Docker Desktop, รัน wsl --shutdown, เปิด Docker Desktop ใหม่จนขึ้น Engine running แล้วรันสคริปต์ใหม่"
    Save-Text "summary.txt" ($Summary -join "`r`n")
    exit 1
}
Run-Step "docker_compose_ps" "docker compose ps" { docker compose ps } | Out-Null

Note "Docker Engine: server $serverVersion"
$c = Run-Step "docker_build" "docker build -f serving/Dockerfile -t $Image ." { docker build -f serving/Dockerfile -t $Image . }
Note ("docker build: " + $(if ($c -eq 0) { "PASS" } else { "FAIL (exit $c)" }))
if ($c -ne 0) { Save-Text "summary.txt" ($Summary -join "`r`n"); exit 1 }

# ---------------------------------------------------------------------------
# 3. MLflow ชั่วคราว (ไม่แตะ docker-compose.yml)
# ---------------------------------------------------------------------------
& docker network create $Net 2>&1 | Out-Null
& docker volume create $Vol 2>&1 | Out-Null
Remove-Container $MlflowName
$c = Run-Step "mlflow_start" "docker run -d --name $MlflowName --network $Net -p ${MlflowPort}:5000 -v ${Vol}:/mlflow $MlflowImage mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --artifacts-destination /mlflow/artifacts" {
    docker run -d --name $MlflowName --network $Net -p "${MlflowPort}:5000" -v "${Vol}:/mlflow" $MlflowImage `
        mlflow server --host 0.0.0.0 --port 5000 `
        --backend-store-uri sqlite:///mlflow/mlflow.db --artifacts-destination /mlflow/artifacts
}
if ($c -ne 0 -or -not (Wait-Http "http://localhost:$MlflowPort/health" 120)) {
    Note "MLflow ชั่วคราวเริ่มไม่ได้ (ดู evidence/mlflow_start.txt)"
    Save-Text "summary.txt" ($Summary -join "`r`n"); exit 1
}

# ---------------------------------------------------------------------------
# 4. Pipeline ของทีม -> champion (รันใน image เดียวกับ API = Python 3.12 + requirements ที่ตรึงไว้)
# ---------------------------------------------------------------------------
if (-not $SkipPipeline) {
    $dirtyBefore = (& git status --porcelain -- reports/experiments) -join ""
    $c = Run-Step "pipeline" "docker run --rm --network $Net -v <repo>:/work -w /work -e MLFLOW_TRACKING_URI=http://${MlflowName}:5000 $Image python pipelines/flow.py" {
        docker run --rm --network $Net -v "${RepoRoot}:/work" -w /work `
            -e "MLFLOW_TRACKING_URI=http://${MlflowName}:5000" -e "PYTHONPATH=/work/src:/work" `
            -e "PREFECT_HOME=/tmp/prefect" $Image python pipelines/flow.py
    }
    if (Test-Path "reports/experiments/best_model.json") {
        Copy-Item "reports/experiments/best_model.json" "$Evidence/pipeline_best_model.json" -Force
        Copy-Item "reports/experiments/experiments.md" "$Evidence/pipeline_experiments.md" -Force -ErrorAction SilentlyContinue
    }
    if ([string]::IsNullOrEmpty($dirtyBefore)) {
        & git restore -- reports/experiments 2>&1 | Out-Null
        Note "reports/experiments/* ที่ pipeline เขียนทับ ถูก git restore กลับแล้ว (สำเนาอยู่ใน evidence/pipeline_*)"
    } else {
        Note "WARNING: reports/experiments มีการแก้ค้างก่อนรัน จึงไม่ได้ restore ให้ ตรวจ git status เอง"
    }
    Note ("pipeline (make all ใน container): " + $(if ($c -eq 0) { "PASS" } else { "FAIL (exit $c)" }))
    if ($c -ne 0) { Save-Text "summary.txt" ($Summary -join "`r`n"); exit 1 }
}

# ---------------------------------------------------------------------------
# 5. ตรวจ champion ใน Registry
# ---------------------------------------------------------------------------
$registryPy = @'
import json, os
from mlflow.tracking import MlflowClient
c = MlflowClient()
mv = c.get_model_version_by_alias("fraud-detector", "champion")
run = c.get_run(mv.run_id)
agreed = {"name": "lightgbm_none", "threshold": 0.51, "run_id": "3db9851cb0b64f1e8987cc009d3d49ab"}
out = {
    "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI"),
    "registry_uri": "models:/fraud-detector@champion",
    "model_version": mv.version,
    "run_id": mv.run_id,
    "run_name": run.data.tags.get("mlflow.runName"),
    "threshold_param": run.data.params.get("threshold"),
    "all_versions": [
        {"version": v.version, "run_id": v.run_id, "aliases": list(v.aliases)}
        for v in c.search_model_versions("name='fraud-detector'")
    ],
    "team_agreement": agreed,
}
out["matches_model_name"] = out["run_name"] == agreed["name"]
out["matches_threshold"] = abs(float(out["threshold_param"]) - agreed["threshold"]) < 1e-9
out["matches_original_run_id"] = out["run_id"] == agreed["run_id"]
print(json.dumps(out, indent=2))
'@
[System.IO.File]::WriteAllText((Join-Path $RepoRoot "$Evidence/_registry_check.py"), $registryPy, $Utf8)
$c = Run-Step "registry_champion" "python evidence/_registry_check.py (in container)" {
    docker run --rm --network $Net -v "${RepoRoot}:/work" -w /work `
        -e "MLFLOW_TRACKING_URI=http://${MlflowName}:5000" $Image python "$Evidence/_registry_check.py"
}
$champ = $null
try { $champ = (Get-Content "$Evidence/registry_champion.txt" -Raw -Encoding UTF8) -replace "(?s)^.*?(\{.*\}).*$", '$1' | ConvertFrom-Json } catch { }
if ($null -eq $champ) {
    Note "อ่าน champion จาก Registry ไม่ได้ (ดู evidence/registry_champion.txt)"
    Save-Text "summary.txt" ($Summary -join "`r`n"); exit 1
}
Note "champion: version=$($champ.model_version) run_id=$($champ.run_id) run_name=$($champ.run_name) threshold=$($champ.threshold_param)"
if (-not ($champ.matches_model_name -and $champ.matches_threshold)) {
    Note "STOP: champion ไม่ใช่ lightgbm_none / threshold 0.51 ตามข้อตกลง -> ไม่ทำ load test (ไม่ใช่หลักฐานของ champion)"
    Save-Text "summary.txt" ($Summary -join "`r`n"); exit 1
}

# ---------------------------------------------------------------------------
# 6. API container + curl evidence
# ---------------------------------------------------------------------------
$payloadPy = @'
import json, pathlib
import pandas as pd
cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
src = pathlib.Path("data/processed/test.csv")
if not src.exists():
    src = pathlib.Path("data/raw/creditcard.csv")
df = pd.read_csv(src)
out = pathlib.Path("reports/serving/evidence/payloads")
legit = df[df["Class"] == 0].iloc[0]
fraud = df[df["Class"] == 1].iloc[0]
(out / "normal.json").write_text(json.dumps({"features": [float(legit[c]) for c in cols]}))
(out / "fraud_example.json").write_text(json.dumps({"features": [float(fraud[c]) for c in cols]}))
(out / "source.txt").write_text(f"{src} | normal row index {legit.name} | fraud row index {fraud.name}\n")
zeros = ", ".join(["0.0"] * 29)
(out / "nan.json").write_text('{"features": [' + zeros + ", NaN]}")
(out / "negative_amount.json").write_text('{"features": [' + zeros + ", -10.0]}")
(out / "text_in_feature.json").write_text('{"features": [' + zeros + ', "abc"]}')
(out / "text_only.json").write_text('{"features": ["abc"]}')
(out / "wrong_count_29.json").write_text('{"features": [' + zeros + "]}")
'@
[System.IO.File]::WriteAllText((Join-Path $RepoRoot "$Evidence/_make_payloads.py"), $payloadPy, $Utf8)
Run-Step "payloads" "python evidence/_make_payloads.py (in container)" {
    docker run --rm -v "${RepoRoot}:/work" -w /work $Image python "$Evidence/_make_payloads.py"
} | Out-Null

if (-not (Start-Api 1)) {
    Note "API container ไม่ healthy (ดู docker logs $ApiName)"
    & docker logs --tail 200 $ApiName 2>&1 | Out-String | ForEach-Object { Save-Text "api_logs.txt" $_ }
    Save-Text "summary.txt" ($Summary -join "`r`n"); exit 1
}

$curl = New-Object System.Collections.ArrayList
function Curl-Record([string]$Label, [string]$Method, [string]$Path, [string]$File, [string]$Expect) {
    $bodyFile = "$Evidence/_resp.txt"
    if ($Method -eq "GET") {
        $shown = "curl.exe $Api$Path"
        $code = & curl.exe -s -o $bodyFile -w "%{http_code}" "$Api$Path"
    } else {
        $shown = "curl.exe -X POST $Api$Path -H `"Content-Type: application/json`" --data-binary `"@$File`""
        $code = & curl.exe -s -o $bodyFile -w "%{http_code}" -X POST "$Api$Path" -H "Content-Type: application/json" --data-binary "@$File"
    }
    $body = Get-Content $bodyFile -Raw -Encoding UTF8
    $ok = if ($Expect) { if ($code -eq $Expect) { "OK" } else { "UNEXPECTED (want $Expect)" } } else { "" }
    $block = "### $Label`r`n> $shown`r`nHTTP $code $ok`r`n$body`r`n"
    Write-Host $block
    [void]$curl.Add($block)
    return $code
}

$h = Curl-Record "health" "GET" "/health" "" "200"
$p1 = Curl-Record "predict normal (real legit row)" "POST" "/predict" "$Payloads/normal.json" "200"
$p2 = Curl-Record "predict fraud example (real fraud row)" "POST" "/predict" "$Payloads/fraud_example.json" "200"
$b1 = Curl-Record "bad: NaN" "POST" "/predict" "$Payloads/nan.json" "400"
$b2 = Curl-Record "bad: negative Amount" "POST" "/predict" "$Payloads/negative_amount.json" "400"
$b3 = Curl-Record "bad: text in feature" "POST" "/predict" "$Payloads/text_in_feature.json" "400"
$b4 = Curl-Record "bad: text only ['abc']" "POST" "/predict" "$Payloads/text_only.json" "400"
$b5 = Curl-Record "bad: 29 features" "POST" "/predict" "$Payloads/wrong_count_29.json" "400"
Save-Text "curl_api.txt" ($curl -join "`r`n")
Note "container /health=$h /predict normal=$p1 fraud_example=$p2 | NaN=$b1 negAmount=$b2 text=$b3 textOnly=$b4 wrongCount=$b5"

$m = & curl.exe -s -w "`nHTTP %{http_code}" "$Api/metrics"
Save-Text "metrics_before_load.txt" ("> curl.exe $Api/metrics`r`n" + ($m -join "`r`n"))
$fams = @("fraud_requests_total", "fraud_request_latency_seconds", "fraud_predictions_total", "fraud_score")
$mText = $m -join "`n"
$missing = @($fams | Where-Object { $mText -notmatch "# TYPE $_ " })
Note ("/metrics families: " + $(if ($missing.Count -eq 0) { "all 4 present" } else { "MISSING: " + ($missing -join ", ") }))

# ---------------------------------------------------------------------------
# 7. Load test (champion) — 1 worker แล้ว 2 worker
# ---------------------------------------------------------------------------
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
$c = Run-Step "load_test_workers1" "python serving/load_test.py --url $Api --requests $Requests --levels 1,10,100 --out reports/serving/load_test_champion.json" {
    python serving/load_test.py --url $Api --requests $Requests --levels 1,10,100 `
        --out reports/serving/load_test_champion.json --note "docker, WEB_CONCURRENCY=1, $env:COMPUTERNAME, $cpu"
}
Note ("load test workers=1 -> reports/serving/load_test_champion.json (SLO exit=$c, 0=PASS 1=FAIL 2=not run)")

$m = & curl.exe -s "$Api/metrics"
Save-Text "metrics_after_load.txt" ("> curl.exe $Api/metrics`r`n" + ($m -join "`r`n"))
& docker logs --tail 300 $ApiName 2>&1 | Out-String | ForEach-Object { Save-Text "api_logs_workers1.txt" $_ }
& docker stats --no-stream --format "{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}" $ApiName 2>&1 | Out-String | ForEach-Object { Save-Text "api_mem_workers1.txt" $_ }

if (Start-Api 2) {
    $c2 = Run-Step "load_test_workers2" "python serving/load_test.py --url $Api --requests $Requests --levels 1,10,100 --out reports/serving/load_test_champion_workers2.json" {
        python serving/load_test.py --url $Api --requests $Requests --levels 1,10,100 `
            --out reports/serving/load_test_champion_workers2.json --note "docker, WEB_CONCURRENCY=2, $env:COMPUTERNAME, $cpu"
    }
    & docker stats --no-stream --format "{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}" $ApiName 2>&1 | Out-String | ForEach-Object { Save-Text "api_mem_workers2.txt" $_ }
    $m = & curl.exe -s "$Api/metrics"
    Save-Text "metrics_after_load_workers2.txt" ("> curl.exe $Api/metrics`r`n" + ($m -join "`r`n"))
    Note ("load test workers=2 -> reports/serving/load_test_champion_workers2.json (SLO exit=$c2)")
} else {
    Note "API workers=2 start ไม่สำเร็จ"
}

# ---------------------------------------------------------------------------
# สรุป + เก็บกวาด
# ---------------------------------------------------------------------------
if (-not $Keep) {
    Remove-Container $ApiName
    Remove-Container $MlflowName
    & docker network rm $Net 2>&1 | Out-Null
    Note "ลบ container ชั่วคราวแล้ว (volume $Vol ยังเก็บ registry ไว้ ใช้ -SkipPipeline รอบหน้าได้)"
} else {
    Note "Keep: API ยังรันที่ $Api และ MLflow ที่ http://localhost:$MlflowPort"
}
Remove-Item "$Evidence/_resp.txt" -ErrorAction SilentlyContinue
Save-Text "summary.txt" ($Summary -join "`r`n")
Write-Host ""
Write-Host "=== SUMMARY (reports/serving/evidence/summary.txt) ===" -ForegroundColor Green
$Summary | ForEach-Object { Write-Host $_ }
