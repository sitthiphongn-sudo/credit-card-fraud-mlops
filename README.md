# Credit Card Fraud Detection — MLOps

โครงงานรายวิชา CP413008 Machine Learning Engineering for Production · กลุ่ม 15 **The bid**

ระบบให้คะแนนความเสี่ยงธุรกรรมบัตรเครดิตแบบเรียลไทม์ ครบวงจรตั้งแต่ข้อมูลดิบ → ตรวจคุณภาพ → เทรน → ประเมิน → ทะเบียนโมเดล → ด่านตรวจ → ให้บริการ → เฝ้าระวัง → เทรนใหม่

- แผนภาพสถาปัตยกรรม: [docs/architecture.md](docs/architecture.md)
- การปล่อยโมเดล, gate และ rollback: [docs/model_release.md](docs/model_release.md)
- AI Project Canvas: [docs/ai_project_canvas.md](docs/ai_project_canvas.md)

## ผลลัพธ์ปัจจุบัน

| รายการ | ค่า |
|---|---|
| โมเดลที่เลือก | `lightgbm_none` |
| PR-AUC บนชุด test (95% CI) | 0.796 (0.707–0.880) |
| Recall ที่ Precision >= 0.80 | 0.773 (gate >= 0.75 ผ่าน) |
| threshold ที่ใช้ตัดสิน | 0.51 · precision 0.902 / recall 0.733 |
| เงินที่ประหยัดได้บนชุด test | 5,042 EUR (ประมาณ 191,764 บาทตามอัตราที่ใช้ในรายงาน) |
| SLO ของ API | p95 <= 100 ms ที่ <= 10 คำขอพร้อมกัน, error < 1% (ผลในรายงาน serving: p95 72 ms) |

รายละเอียด: [reports/experiments/experiments.md](reports/experiments/experiments.md) · [reports/serving/](reports/serving/)

เวอร์ชัน champion และ threshold ที่ใช้งานจริงตรวจได้จาก `/health`
ผล latency อาจเปลี่ยนตามเครื่องและภาระงาน

## สมาชิกและส่วนที่รับผิดชอบ

| สมาชิก | ส่วนที่รับผิดชอบ | ไฟล์ / โฟลเดอร์หลัก |
|---|---|---|
| นายกันตพัฒน์ โชติเจริญวัฒนะกุล 673380305-2 | หัวหน้ากลุ่ม · Repo & CI/CD | `.github/workflows/ci.yml`, `scripts/make_ci_data.py`, `scripts/ci_checks.py` |
| นายเปรมสิริวัฒณ์ วราธิกานนท์ 673380360-4 | Data & Validation | `src/fraud/data.py`, `src/fraud/validate.py`, `data/sample/`, `scripts/eda.py`, `scripts/make_samples.py`, `scripts/schema_stats.py`, `reports/eda.md`, `tests/test_data.py`, `tests/test_validate.py` |
| นายนาคินทร์ ประเสริฐยิ่ง 673380324-8 | Feature Engineering · AI Project Canvas | `src/fraud/features.py`, `tests/test_features.py`, `docs/ai_project_canvas.md` |
| นายสิทธิพงษ์ นครขวาง 673380350-7 | Modeling & Experiments | `src/fraud/train.py`, `src/fraud/evaluate.py`, `tests/test_train.py`, `tests/test_evaluate.py`, `reports/experiments/` |
| นายสหรัฐ งามเลิศ 673380349-2 | Serving & Performance | `serving/`, `tests/test_api.py`, `reports/serving/` |
| นายโชติกานต์ วิลาชัย 673380362-0 | Monitoring & Drift | `monitoring/` |
| นายธรรมรักษ์ บุตราช 673380077-9 | Pipeline & Model Registry · Docs | `pipelines/`, `src/fraud/gate.py`, `tests/test_gate.py`, `tests/test_flow.py`, `Makefile`, `docker-compose.yml`, `docs/architecture.md`, `docs/model_release.md` |

## ชุดข้อมูล

[Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 รายการ, fraud 492 (0.172%)

ดาวน์โหลด `creditcard.csv` แล้ววางที่ `data/raw/creditcard.csv`
ไฟล์ข้อมูลดิบไม่ถูก commit และต้องเตรียมเองหลัง clone

แบ่งข้อมูลตามเวลาเป็น train/validation/test สัดส่วน 60/20/20

## เริ่มใช้งาน

ต้องมี Python 3.12 และ Docker Desktop หรือ Docker Engine พร้อม Docker Compose

### ติดตั้งบน Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

หาก PowerShell บล็อกการเปิด environment ให้รันคำสั่งนี้ใน Terminal ปัจจุบันก่อน:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### ติดตั้งบน Bash

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

ถ้าไม่มี Make ให้ใช้คำสั่ง Python ที่ระบุไว้ด้านล่าง

### 1. ตรวจคุณภาพโค้ดและข้อมูล

```powershell
ruff check .
python -m pytest -q
python -m fraud.validate data/sample/valid.csv
python -m fraud.validate data/sample/bad_missing_column.csv
```

ไฟล์ `valid.csv` ต้องผ่าน ส่วนไฟล์ข้อมูลเสียต้องแจ้ง
`DATA VALIDATION FAILED` และคืน exit code `1`

ตรวจ exit code ทันทีหลังคำสั่งใน PowerShell:

```powershell
$LASTEXITCODE
```

### 2. รันทั้งระบบด้วย Docker

เปิด Docker ให้พร้อมก่อนรันคำสั่ง

บน PowerShell:

```powershell
docker compose up -d mlflow
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
python pipelines/flow.py
docker compose up -d --build
```

บน Bash:

```bash
docker compose up -d mlflow
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
make all
docker compose up -d --build
```

ต้องมี `data/raw/creditcard.csv` ก่อนรัน pipeline
และต้องรอให้ MLflow พร้อมรับการเชื่อมต่อ

Pipeline รับข้อมูล → ตรวจ schema → แบ่งชุด → รันการทดลอง →
เลือกโมเดลจาก validation → ลงทะเบียน challenger → ตรวจ Gate →
เลื่อนเป็น champion เมื่อผ่าน

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5001 |
| API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Grafana ใช้บัญชี `admin/admin` สำหรับการสาธิตในเครื่อง

Compose เปิด MLflow ที่พอร์ต `5001` บนเครื่อง และพอร์ต `5000`
ภายใน container โดย API เชื่อมผ่าน `http://mlflow:5000`

MLflow เก็บฐานข้อมูลและ artifacts ใน `./mlflow_data`
และรับส่ง artifacts ผ่าน MLflow server

ตั้ง `MLFLOW_TRACKING_URI` ใหม่ทุกครั้งที่เปิด Terminal ใหม่

ตรวจว่า API โหลด champion แล้ว:

```powershell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

ตัวอย่างผล:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_uri": "models:/fraud-detector@champion",
  "model_version": "1",
  "threshold": 0.51
}
```

หมายเลขเวอร์ชันขึ้นกับประวัติ Registry ในเครื่องนั้น
หลังเริ่มหรือ restart API ให้รอจนโหลดโมเดลเสร็จก่อนตรวจ

### 3. เรียกใช้ API

บน PowerShell สร้างธุรกรรมตัวอย่าง 30 ค่า:

```powershell
$features = @(0) * 30
$features[29] = 100.0
$body = @{ features = $features } | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://localhost:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json
```

บน Bash:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100.0]}'
```

ลำดับ 30 ค่าคือ `Time, V1, ..., V28, Amount`

API ใช้ `predict_proba` และ threshold จาก MLflow run ของ champion
ตอบ `is_fraud`, `fraud_score`, `threshold` และ `model_version`

ข้อมูลผิดปกติ เช่น Amount ติดลบ ค่าว่าง หรือข้อความแทนตัวเลข
ต้องตอบ HTTP 400

| Endpoint | หน้าที่ |
|---|---|
| `POST /predict` | รับ `{"features": [Time, V1..V28, Amount]}` จำนวน 30 ค่าและทำนาย fraud |
| `GET /health` | สถานะและเวอร์ชันโมเดล หากยังไม่พร้อมตอบ 503 |
| `GET /metrics` | Prometheus metrics: `fraud_requests_total`, `fraud_request_latency_seconds`, `fraud_predictions_total`, `fraud_score` |

### 4. ด่านตรวจ, rollback และเทรนใหม่

Model Gate ตรวจ:

- `recall_at_p80` >= 0.75
- PR-AUC ต่ำกว่า championได้ไม่เกิน 0.005
- inference p95 <= 100 ms
- model artifact <= 50 MB

ค่า inference p95 ของโมเดลแยกจาก latency ของ API load test

ตรวจ candidate บน PowerShell:

```powershell
python -m fraud.gate check --candidate reports/experiments/best_model.json
$LASTEXITCODE
```

PASS คืน exit code `0` และ FAIL คืน `1`

เทรนใหม่เมื่อได้รับสัญญาณ RETRAIN:

```powershell
python pipelines/flow.py --signal RETRAIN
```

หาก Gate ผ่านจะเลื่อนเวอร์ชันใหม่เป็น champion
จากนั้น restart API เพื่อโหลดโมเดลใหม่:

```powershell
docker compose restart api
```

ย้อน champion ไปยังเวอร์ชันที่มีอยู่ เช่น Version 1:

```powershell
python -m fraud.gate rollback 1
docker compose restart api
```

รอให้ API พร้อม แล้วตรวจเวอร์ชัน:

```powershell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

API โหลดโมเดลและ threshold ตอนเริ่ม service
จึงต้อง restart หลัง promote หรือ rollback

คำสั่งที่เทียบเท่าเมื่อมี Make:

```bash
make gate CANDIDATE=reports/experiments/best_model.json
make retrain
make rollback VERSION=1
```

รายละเอียดและรายการหลักฐาน:
[docs/model_release.md](docs/model_release.md)

### 5. เฝ้าระวัง drift

```powershell
python monitoring/drift_check.py --scenario data
```

Data drift ให้ผล `WATCH` และ exit code `1` ตามสถานการณ์จำลอง

```powershell
python monitoring/drift_check.py --scenario concept
```

Concept drift ให้ผล `RETRAIN` และ exit code `2` ตามสถานการณ์จำลอง

เมื่อได้ RETRAIN เรียก:

```powershell
python pipelines/flow.py --signal RETRAIN
```

Prometheus เชื่อม drift exporter ที่ `host.docker.internal:9108`
โดย exporter ต้องรันบนเครื่องตามขั้นตอนของ monitoring

รายละเอียดการเริ่ม exporter เกณฑ์แจ้งเตือนและนโยบายเทรนใหม่:
[monitoring/README.md](monitoring/README.md)

### ดูผลการทดลอง

```powershell
python -m fraud.train
```

รันการทดลองทั้ง 8 แบบและเขียนผลที่ `reports/experiments/`
หากตั้ง `MLFLOW_TRACKING_URI` ไว้ จะบันทึก runs ไปยัง server นั้น

กรณีเทรนโดยไม่ได้ตั้ง `MLFLOW_TRACKING_URI` ให้เปิด UI
ของ local tracking store แยกจาก Docker:

```powershell
mlflow ui --port 5002
```

เปิดที่ http://127.0.0.1:5002

## โครงสร้าง

```text
configs/params.yaml     ค่าต้นทุน, gate, monitoring และ registry
src/fraud/              โค้ดหลัก: data, validate, features, train, evaluate, gate
pipelines/flow.py       Prefect DAG: ingest → check → split → train → register → gate → champion
serving/                FastAPI, Dockerfile, health, metrics และ load test
monitoring/             จำลอง drift, ตรวจ data/concept drift, exporter, alerts และ Grafana
tests/                  unit tests
scripts/                EDA, ไฟล์ตัวอย่าง, ข้อมูลสังเคราะห์และตัวตรวจของ CI
data/raw/               ข้อมูลดิบที่ต้องเตรียมเอง ไม่ commit
data/sample/            ตัวอย่างปกติ 1 ไฟล์และข้อมูลเสีย 5 แบบ
reports/                ผล EDA, ผลการทดลองและหลักฐาน
docs/                   AI Project Canvas, สถาปัตยกรรมและ Model Release Contract
mlflow_data/            ฐานข้อมูลและ artifacts ของ MLflow ใน Docker ไม่ commit
.github/workflows/      CI: คุณภาพโค้ด, ความถูกต้องของข้อมูลและคุณภาพโมเดล
```

## วิธีทำงานร่วมกัน

1. ทำงานผ่าน branch และ Pull Request เช่น `feat/<ส่วน>-<เรื่อง>` หรือ `fix/<ส่วน>-<เรื่อง>`
2. Commit ด้วยบัญชีตัวเองเพื่อให้ประวัติแสดงผลงานรายบุคคล
3. เปิด Pull Request รอ CI ผ่านและเพื่อน approve อย่างน้อย 1 คนก่อน merge
4. แก้เฉพาะไฟล์ที่รับผิดชอบ หากต้องแก้ `configs/params.yaml` หรือ `requirements.txt` ให้ตกลงกับกลุ่มก่อน
5. บันทึกการใช้ AI ใน [AI_USAGE.md](AI_USAGE.md) และอธิบายโค้ดของตัวเองได้