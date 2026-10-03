# Monitoring & Drift — owner: โชติกานต์

ส่วนนี้ทำ production simulation, Evidently, NannyML CBPE, Prometheus/Grafana, alert และสรุปผลเป็น
`OK / WATCH / RETRAIN` โดยไม่แก้โค้ดของ Modeling/Serving/Pipeline.

## Contract ที่ใช้ร่วมกับทีม

- Model threshold: อ่านจาก `reports/experiments/best_model.json` (ไม่ hardcode 0.51)
- Evidently reference: `data/processed/train.csv`
- Production source: `data/processed/test.csv`
- API metrics: `fraud_requests_total`, `fraud_request_latency_seconds`, `fraud_predictions_total`, `fraud_score`
- RETRAIN hand-off: พิมพ์ `RETRAIN` + exit code `2` ให้ pipeline นำไปใช้ต่อ

## Flow

```text
simulate_drift.py
  -> Period 1-2 normal / Period 3 data drift / Period 4 concept drift
  -> run_evidently.py
  -> run_nannyml.py
  -> drift_check.py / decide_action.py
  -> monitoring_summary.json
  -> drift_exporter.py :9108
  -> Prometheus -> Grafana + alert rules
```

## 1. Production simulation 4 ช่วง

```bash
python monitoring/simulate_drift.py
```

ผลอยู่ใน `monitoring/generated/`:

- `period_1.csv`, `period_2.csv` — normal
- `period_3_data_drift.csv` — `Amount` สูงขึ้น, `V14/V17` shift, label ไม่เปลี่ยน
- `period_4_concept_drift.csv` — features เหมือนเดิม แต่ low-Amount normal บางส่วนถูก flip เป็น fraud
- `simulation_summary.json` — audit หลักฐานว่าฉีด drift ถูกชนิด

## 2. คำสั่งหลักสำหรับ demo

ก่อนรันต้องมี 3 อย่าง:

1. champion ใน MLflow (รัน pipeline ตาม README หลัก) และตั้ง `MLFLOW_TRACKING_URI` เป็น `http://127.0.0.1:5001`
2. ข้อมูลที่แบ่งแล้วใน `data/processed/` สร้างด้วย `python -m fraud.data`
3. environment `.venv-nannyml` ตามข้อ 4 ถ้ายังไม่มี ให้เติม `--skip-nannyml` ท้ายคำสั่ง (ใช้ recall จริงอย่างเดียว)

```bash
python monitoring/drift_check.py --scenario normal1
python monitoring/drift_check.py --scenario normal2
python monitoring/drift_check.py --scenario data
python monitoring/drift_check.py --scenario concept
```

สถานะ/exit code:

- `OK` -> `0`
- `WATCH` -> `1`
- `RETRAIN` -> `2`

`--scenario data` ใช้ Period 3 และควรแสดง root cause ด้าน feature เช่น Amount/V14/V17.
`--scenario concept` ใช้ Period 4 และเมื่อ actual recall / CBPE gap เกิน config จะพิมพ์ `RETRAIN`.

คำสั่งตัวอย่างที่เอกสารกลุ่มต้องการโดยตรงคือ:

```bash
python monitoring/drift_check.py --scenario data
python monitoring/drift_check.py --scenario concept
```

## 3. Evidently แยกรัน

```bash
python monitoring/run_evidently.py
```

Default reference คือ `data/processed/train.csv`. HTML/JSON ราย period อยู่ที่
`reports/monitoring/evidently/` และสรุปอยู่ที่ `reports/monitoring/evidently_summary.json`.

รายงานเก็บ:

- drift share
- drifted features / root causes
- prediction ratio และ probability distribution summary
- actual recall/precision เมื่อ label มีแล้ว

## 4. NannyML CBPE แยกรัน

```bash
python monitoring/run_nannyml.py
```

CBPE fit จาก healthy Period 1+2 แล้ว estimate recall โดยไม่ใช้ `Class` ของ analysis period. เมื่อ label
กลับมา PerformanceCalculator/โค้ด realized metric จะเทียบ estimated กับ actual และคำนวณ concept gap.

NannyML ไม่รองรับ `lightgbm==4.6.0` ของโครงการ จึงรันใน environment แยก `.venv-nannyml`
(`run_nannyml.py` อ่านเฉพาะ CSV ที่ `prepare_nannyml_inputs.py` เตรียมไว้ ไม่โหลดโมเดล)

สร้างครั้งเดียว บน Windows PowerShell:

```powershell
python -m venv .venv-nannyml
.\.venv-nannyml\Scripts\python -m pip install nannyml==0.13.1
```

บน Bash:

```bash
python -m venv .venv-nannyml
.venv-nannyml/bin/python -m pip install nannyml==0.13.1
```

`drift_check.py` หา Python ของ environment นี้เองที่ `.venv-nannyml/Scripts/python.exe` (Windows)
หรือ `.venv-nannyml/bin/python` ระบุที่อื่นได้ด้วย `--nannyml-python <path>`
ถ้าไม่มี environment นี้ ให้ใช้ `--skip-nannyml`

## 5. Decision แยกรัน

```bash
python monitoring/decide_action.py --period period_4
```

เกณฑ์ `drift_share`, `max_recall_drop`, `concept_gap` อ่านจาก `configs/params.yaml` เท่านั้น.
Baseline recall จะใช้ `metrics.test_recall` จาก model hand-off ก่อน; ถ้าไม่มีจึง fallback ไป healthy periods.

## 6. Prometheus exporter

หลังมี `reports/monitoring/monitoring_summary.json`:

```bash
python monitoring/drift_exporter.py
```

เปิด `http://localhost:9108/metrics` และส่ง batch gauges เช่น drift share, actual/estimated recall,
concept gap, action และ last successful run timestamp.

## 7. Prometheus + Grafana

```bash
docker compose up -d prometheus grafana
```

Dashboard มี 2 แถว:

- System Health: p95 latency, error rate, requests
- Model Health: drift share, estimated vs actual recall, fraud/normal prediction share, fraud score, decision

Grafana queries ใช้ metric contract ของสหรัฐที่ระบุด้านบน.

## 8. Alert / retraining policy

- Prometheus rules: `monitoring/alert.rules.yml`
- ตารางอธิบาย: `reports/monitoring/alert_thresholds.md`
- Retraining policy: `reports/monitoring/retraining_policy.md`

มี stale-monitoring alert เพื่อไม่ให้ค่ารอบเก่าค้างเป็นสีเขียวเมื่อ batch monitoring ไม่ได้รัน.

## ก่อนเปิด PR

อ่าน `monitoring/required_team_changes.md` ก่อน เพราะยังมี shared dependency/config ที่โชติกานต์แก้เองไม่ได้.
จากนั้นรัน:

```bash
ruff check .
pytest -q
```

และเก็บภาพ demo อย่างน้อย: normal เงียบ, data drift root cause, concept drift -> RETRAIN,
Prometheus targets/alerts และ Grafana dashboard 2 แถว.
