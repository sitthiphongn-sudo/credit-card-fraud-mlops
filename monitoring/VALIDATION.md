# Monitoring validation (final)

ตรวจบน main `f3c182a` วันที่ 3 ต.ค. 2569 ด้วย champion model version `1`
และ model threshold `0.51`.

## Commands

| คำสั่ง | ผล | exit code |
|---|---|---:|
| `ruff check .` | All checks passed | 0 |
| `python -m pytest -q` | 123 passed, 15 warnings | 0 |
| `python monitoring/drift_check.py --scenario normal1` | OK · drift share 0.0000 | 0 |
| `python monitoring/drift_check.py --scenario normal2` | OK · drift share 0.0333 | 0 |
| `python monitoring/drift_check.py --scenario data` | WATCH · drift share 0.1667 · drifted features: Amount, Time, V14, V17, V2 | 1 |
| `python monitoring/drift_check.py --scenario concept` | RETRAIN · actual recall 0.0284 vs CBPE 0.5694 · concept gap 0.5409 | 2 |

## Monitoring results

### Normal periods

- Period 1: drift share = 0.0000
- Period 1 actual recall = 0.7826
- Period 1 CBPE estimated recall = 0.7473
- Period 2: drift share = 0.0333
- Period 2 actual recall = 0.7667
- Period 2 CBPE estimated recall = 0.7960

ทั้งสองช่วงอยู่ภายใน threshold จึงได้สถานะ `OK`.

### Data drift

Period 3 ได้สถานะ `WATCH`.

- drift share = 0.1667
- drifted features = `Amount`, `Time`, `V14`, `V17`, `V2`
- actual recall = 0.9091
- CBPE estimated recall = 0.8111

feature distribution เปลี่ยนเกิน `drift_share` แต่ realized recall ไม่ลดลง
จึงแจ้งเตือนเป็น `WATCH` แทนการ retrain ทันที.

### Concept drift

Period 4 ได้สถานะ `RETRAIN`.

- actual recall = 0.0284
- CBPE estimated recall = 0.5694
- recall drop = 0.7316
- concept gap = 0.5409

ทั้ง recall drop และ estimated-vs-actual recall gap เกิน threshold
จึงคืน `RETRAIN` และ exit code `2`.

## Config (`configs/params.yaml` ส่วน monitoring)

- `drift_share`: 0.10
- `max_recall_drop`: 0.10
- `concept_gap`: 0.10
- `max_error_rate`: 0.01
- `stale_after_seconds`: 86400

`max_fraud_prediction_rate` ไม่ได้ตั้งค่า และ alert นี้ถูกปิดไว้เมื่อ threshold ไม่ได้ถูก configure.

## NannyML

NannyML CBPE ใช้ healthy Period 1 และ Period 2 เป็น reference
และซ่อน analysis labels ระหว่างการ estimate recall.

NannyML รันผ่านด้วย environment แยก `.venv-nannyml`.
มี warning เรื่องจำนวน chunk ต่ำ แต่ไม่ทำให้การรันล้มเหลว.

## Prometheus / Grafana

Metric contract ล่าสุด:

- `fraud_requests_total{status="success|bad_request|error"}`
- `fraud_predictions_total{label="fraud|not_fraud"}`
- `fraud_request_latency_seconds`
- `fraud_score`

Prometheus และ Grafana ใช้ metric contract นี้ในการแสดง System Health,
Model Health และ alert rules.

## Prometheus target validation

ตรวจ Prometheus Targets แล้วพบว่า service ที่ monitoring ต้องใช้พร้อมทำงานทั้งคู่:

- `fraud-api` = `UP`
- `fraud-monitoring-batch` = `UP`

จึงยืนยันได้ว่า Prometheus สามารถ scrape ทั้ง API metrics และ batch monitoring metrics ได้สำเร็จ.

## Prometheus alert validation

หลังรัน scenario `concept` พบว่า Prometheus โหลด alert rules ครบและ alert ที่เกี่ยวข้องกับ model degradation ขึ้น `FIRING` ได้แก่:

- `FeatureDriftShareHigh`
- `RecallDropCritical`
- `ConceptDriftSuspected`

ขณะที่ `MonitoringConfigInvalid` ยังเป็น `INACTIVE`
จึงยืนยันว่า monitoring config ใช้งานได้และ alert ตอบสนองต่อ concept drift ตามที่ออกแบบไว้.

## Grafana dashboard validation

ตรวจ dashboard `Credit Card Fraud - System & Model Health` แล้วพบว่าแผงหลักแสดงข้อมูลได้ครบ ได้แก่:

- System Health: p95 latency, error rate, requests
- Model Health: drift share, estimated vs actual recall, prediction share, fraud score และ monitoring decision
- Prediction Share ใช้ label `fraud` และ `not_fraud` ตาม metric contract ของ API

หลังแก้ PromQL ของ Prediction Share ให้ใช้ scalar denominator แล้ว pie chart แสดงข้อมูลได้ถูกต้อง.

## End-to-end retraining validation

หลัง scenario `concept` คืน `RETRAIN` และ exit code `2`
ได้รันคำสั่ง:

`python pipelines/flow.py --signal RETRAIN`

ผลคือ retraining pipeline ทำงานครบจนถึงขั้น release,
สร้าง `fraud-detector` model version `2`
และย้าย alias `champion` ไปยัง version `2` สำเร็จ.

จึงยืนยันได้ว่า monitoring signal สามารถเชื่อมต่อไปยัง retraining pipeline
และ model promotion flow ได้แบบ end-to-end.

## NannyML environment validation

NannyML รันผ่าน environment แยก `.venv-nannyml`
เพื่อหลีกเลี่ยง dependency conflict กับ environment หลักของโครงการ.

ขั้นตอน monitoring เตรียม input ใน main environment ก่อน
แล้วให้ `run_nannyml.py` อ่านเฉพาะไฟล์ที่เตรียมไว้สำหรับ CBPE
โดยไม่ต้องโหลด training stack ทั้งชุด.

แนวทางนี้ช่วยแยก dependency ของ monitoring ออกจาก serving/training
และลดความเสี่ยงที่การติดตั้ง NannyML จะกระทบ package หลักของระบบ.
