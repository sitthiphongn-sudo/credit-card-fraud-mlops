# Credit Card Fraud Detection — MLOps

โครงงานรายวิชา CP413008 Machine Learning Engineering for Production · กลุ่ม 15 **The bid**

ระบบให้คะแนนความเสี่ยงธุรกรรมบัตรเครดิตแบบเรียลไทม์ ครบวงจรตั้งแต่ข้อมูลดิบ → ตรวจคุณภาพ → เทรน → ประเมิน → ด่านตรวจ → ทะเบียนโมเดล → ให้บริการ → เฝ้าระวัง → เทรนใหม่

- แผนภาพสถาปัตยกรรม: [docs/architecture.md](docs/architecture.md)
- การปล่อยโมเดล, gate และ rollback: [docs/model_release.md](docs/model_release.md)
- AI Project Canvas: [docs/ai_project_canvas.md](docs/ai_project_canvas.md)

## ผลลัพธ์ปัจจุบัน

| รายการ | ค่า |
|---|---|
| โมเดลที่ใช้งาน (champion) | `lightgbm_none` |
| PR-AUC บนชุด test (95% CI) | 0.796 (0.707–0.880) |
| Recall ที่ Precision ≥ 0.80 | 0.773 (gate ≥ 0.75 ผ่าน) |
| threshold ที่ใช้ตัดสิน | 0.51 · precision 0.902 / recall 0.733 |
| เงินที่ประหยัดได้บนชุด test | 5,042 EUR (≈ 191,764 บาท) |
| SLO ของ API | p95 ≤ 100 ms ที่ ≤ 10 คำขอพร้อมกัน, error < 1% (วัดได้ p95 72 ms) |

รายละเอียด: [reports/experiments/experiments.md](reports/experiments/experiments.md) · [reports/serving/](reports/serving/)

## สมาชิกและส่วนที่รับผิดชอบ

| สมาชิก | ส่วนที่รับผิดชอบ | ไฟล์ / โฟลเดอร์หลัก |
|---|---|---|
| กันตพัฒน์ | หัวหน้ากลุ่ม · Repo & CI/CD | `.github/workflows/ci.yml`, `scripts/make_ci_data.py`, `scripts/ci_checks.py` |
| เปรมสิริวัฒณ์ | Data & Validation | `src/fraud/data.py`, `src/fraud/validate.py`, `data/sample/`, `scripts/eda.py`, `scripts/make_samples.py`, `scripts/schema_stats.py`, `reports/eda.md`, `tests/test_data.py`, `tests/test_validate.py` |
| นาคินทร์ | Feature Engineering · AI Project Canvas | `src/fraud/features.py`, `tests/test_features.py`, `docs/ai_project_canvas.md` |
| สิทธิพงษ์ | Modeling & Experiments | `src/fraud/train.py`, `src/fraud/evaluate.py`, `tests/test_train.py`, `tests/test_evaluate.py`, `reports/experiments/` |
| สหรัฐ | Serving & Performance | `serving/`, `tests/test_api.py`, `reports/serving/` |
| โชติกานต์ | Monitoring & Drift | `monitoring/` |
| ธรรมรักษ์ | Pipeline & Model Registry · Docs | `pipelines/`, `src/fraud/gate.py`, `tests/test_gate.py`, `tests/test_flow.py`, `Makefile`, `docker-compose.yml`, `docs/architecture.md`, `docs/model_release.md` |

## ชุดข้อมูล

[Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 รายการ, fraud 492 (0.172%)

ดาวน์โหลด `creditcard.csv` แล้ววางที่ `data/raw/creditcard.csv` (ไฟล์ข้อมูลไม่ถูก commit) · แบ่งข้อมูลตามเวลา 60/20/20

## เริ่มใช้งาน

ต้องมี Python 3.12 และ Docker (สำหรับรันทั้งระบบ)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

> **Windows:** ถ้าไม่มีคำสั่ง `make` ใช้คำสั่งในวงเล็บแทน · ตั้งตัวแปรด้วย `set` (cmd / Anaconda Prompt) หรือ `$env:ชื่อ="ค่า"` (PowerShell)

### 1. ตรวจคุณภาพโค้ดและข้อมูล

```bash
ruff check .
python -m pytest -q
python -m fraud.validate data/sample/valid.csv              # ต้องผ่าน
python -m fraud.validate data/sample/bad_missing_column.csv # ต้องหยุดพร้อมแจ้ง DATA VALIDATION FAILED (exit 1)
```

### 2. รันทั้งระบบด้วย Docker (MLflow + pipeline + API + monitoring)

```bash
docker compose up -d mlflow                      # MLflow UI: http://localhost:5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000 # Windows cmd: set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
make all                                         # (python pipelines/flow.py) ข้อมูลดิบ → validate → split → เทรน → gate → register → champion
docker compose up -d --build                     # api :8000 · prometheus :9090 · grafana :3000 (admin/admin)
```

ตรวจว่า API โหลด champion แล้ว:

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_version":"1","threshold":0.51,...}
```

### 3. เรียกใช้ API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100.0]}'
# {"is_fraud":0,"fraud_score":...,"threshold":0.51,"model_version":"1"}
```

ลำดับ 30 ค่าคือ `Time, V1, ..., V28, Amount` · ส่ง Amount ติดลบ, ค่าว่าง หรือข้อความ จะได้ 400 พร้อมบอกว่าผิดที่ฟีเจอร์ไหน

| Endpoint | หน้าที่ |
|---|---|
| `POST /predict` | รับ `{"features": [Time, V1..V28, Amount]}` (30 ค่า) ตอบ `is_fraud`, `fraud_score`, `threshold`, `model_version` · ข้อมูลผิดปกติตอบ 400 |
| `GET /health` | สถานะและเวอร์ชันโมเดล (ยังโหลดไม่เสร็จตอบ 503) |
| `GET /metrics` | Prometheus metrics: `fraud_requests_total`, `fraud_request_latency_seconds`, `fraud_predictions_total`, `fraud_score` |

### 4. ด่านตรวจ, rollback และเทรนใหม่

```bash
make gate CANDIDATE=reports/experiments/best_model.json   # (python -m fraud.gate check --candidate ...)
make rollback VERSION=1                                   # (python -m fraud.gate rollback 1) ย้าย alias champion กลับ
docker compose restart api                                # ให้ API โหลด champion ใหม่
make retrain                                              # (python pipelines/flow.py --signal RETRAIN)
```

### 5. เฝ้าระวัง drift

```bash
python monitoring/drift_check.py --scenario data      # data drift → WATCH (exit 1)
python monitoring/drift_check.py --scenario concept   # concept drift → RETRAIN (exit 2)
```

รายละเอียดเกณฑ์แจ้งเตือนและนโยบายเทรนใหม่: [monitoring/README.md](monitoring/README.md)

### ดูผลการทดลอง

```bash
python -m fraud.train        # รันการทดลองทั้ง 8 แบบ เขียนผลที่ reports/experiments/
mlflow ui                    # เปิด http://127.0.0.1:5000 (กรณีเทรนโดยไม่ตั้ง MLFLOW_TRACKING_URI)
```

## โครงสร้าง

```
configs/params.yaml     ค่าคงที่และเกณฑ์ทั้งหมด (ต้นทุน, gate, monitoring, registry) ที่เดียว
src/fraud/              โค้ดหลัก: data, validate, features, train, evaluate, gate
pipelines/flow.py       Prefect DAG: ingest → validate → split → train → evaluate → gate → register → champion
serving/                FastAPI + Dockerfile (/predict /health /metrics) + load test
monitoring/             จำลอง drift, ตรวจ data/concept drift, Prometheus alert rules, Grafana dashboard
tests/                  unit tests (รันใน CI)
scripts/                EDA, สร้างไฟล์ตัวอย่าง, ข้อมูลสังเคราะห์และตัวตรวจของ CI
data/sample/            ไฟล์ตัวอย่างปกติ 1 ไฟล์ + ไฟล์เสีย 5 แบบ สำหรับสาธิตการตรวจข้อมูล
reports/                ผล EDA, ผลการทดลอง, หลักฐาน serving
docs/                   AI Project Canvas, แผนภาพสถาปัตยกรรม, การปล่อยโมเดล
.github/workflows/      CI 3 ด้าน: คุณภาพโค้ด · ความถูกต้องของข้อมูล · เกณฑ์คุณภาพโมเดล
```

## วิธีทำงานร่วมกัน

1. ห้าม push เข้า `main` ตรง (ตั้ง branch protection แล้ว) แตก branch ตามรูปแบบ `feat/<ส่วน>-<เรื่อง>` เช่น `feat/data-schema`
2. commit บ่อย ๆ ด้วยบัญชีตัวเอง (คะแนนรายบุคคลดูจากประวัติ commit)
3. เปิด Pull Request → CI ทั้ง 3 job ต้องผ่าน → มีเพื่อน approve อย่างน้อย 1 คน → merge
4. แก้เฉพาะไฟล์ของตัวเอง · `configs/params.yaml` และ `requirements.txt` ต้องแจ้งกลุ่มก่อนแก้
5. ถ้าใช้ AI ช่วยเขียนโค้ด ให้บันทึกใน [AI_USAGE.md](AI_USAGE.md) และต้องอธิบายโค้ดของตัวเองได้ทุกบรรทัด
