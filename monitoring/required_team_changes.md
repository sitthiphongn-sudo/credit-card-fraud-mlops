# สิ่งที่โชติกานต์ต้องประสานกับทีมก่อน Monitoring รันครบ End-to-End

เอกสารอัปเดตของกลุ่มวันที่ 1 ต.ค. 2569 กำหนด contract ระหว่าง Modeling, Serving, Pipeline และ Monitoring
ชัดขึ้นแล้ว ดังนั้นโค้ดใน `monitoring/` เวอร์ชันนี้ยึด contract ใหม่นี้เป็นหลัก แต่ไม่แก้ไฟล์ของเพื่อนเอง.

## 1. ต้องรอ `reports/experiments/best_model.json` จากสิทธิพงษ์

Monitoring ใช้ threshold เดียวกับ production โดยอ่านจาก `reports/experiments/best_model.json` แทน
`model.predict()` ที่ตัด 0.5. เอกสารกลุ่มระบุโมเดลปัจจุบันเป็น `lightgbm_none` และ threshold ที่เลือกคือ
0.51 แต่โค้ด monitoring **ไม่ hardcode 0.51**; จะอ่านจาก hand-off file หลัง PR ของสิทธิพงษ์ merge.

Monitoring ยังพยายามอ่าน `metrics.test_recall` จากไฟล์เดียวกันเพื่อใช้เป็น baseline recall ที่ threshold จริง.
เอกสารอัปเดตระบุค่าปัจจุบันประมาณ 0.733. ถ้า key นี้ไม่มี จะ fallback ไปใช้ healthy Period 1+2.

## 2. NannyML ยังต้องแก้ dependency ร่วมกัน

โจทย์รายบุคคลเดิมกำหนดให้ใช้ NannyML CBPE แต่ `requirements.txt` ปัจจุบันยังไม่มี NannyML.
เวอร์ชัน `nannyml==0.13.1` รองรับ Python 3.12 แต่ dependency metadata ของ PyPI ระบุ
`lightgbm>=3.3,<4.6` ขณะที่ repo ตรึง `lightgbm==4.6.0`.

ดังนั้นโชติกานต์ไม่ควรแก้ `requirements.txt` หรือ downgrade LightGBM เอง. ให้สิทธิพงษ์/กลุ่มเลือก dependency
set ที่ทดสอบกับ champion แล้วก่อน จากนั้นจึงเพิ่ม NannyML ใน shared requirements.

## 3. ต้องเพิ่ม threshold สำหรับ alert ที่ยังไม่มีใน `configs/params.yaml`

ค่าที่มีแล้ว:

```yaml
monitoring:
  drift_share: 0.30
  max_recall_drop: 0.10
  concept_gap: 0.10
```

แต่ alert system ยังต้องการค่าอีกสามตัว โดยกติกาของงานห้าม hardcode ซ้ำใน PromQL:

```yaml
monitoring:
  max_error_rate: 0.01
  max_fraud_prediction_rate: <ค่าที่ทีม calibrate จาก healthy traffic>
  stale_after_seconds: <ค่าตาม cadence ของ batch monitoring>
```

`max_error_rate: 0.01` มาจากเอกสารอัปเดตของกลุ่ม. ส่วน fraud prediction rate และ stale window ต้องให้ทีม
ตกลงจาก Period 1-2 / healthy traffic และความถี่ที่ตั้งใจรัน batch monitoring.

ถ้าค่ายังไม่ครบ exporter จะปล่อย `fraud_monitor_config_valid 0` และ Alertmanager/Prometheus rule
`MonitoringConfigInvalid` จะเตือนแทนการแอบใช้ default.

## 4. Metric contract ของสหรัฐ — ชื่อต้องตรงนี้

เอกสารอัปเดตกำหนดชื่อ Prometheus metrics ฝั่ง API แล้ว:

- `fraud_requests_total{status="ok|bad_request|error"}`
- `fraud_request_latency_seconds` (Histogram)
- `fraud_predictions_total{label="fraud|normal"}`
- `fraud_score` (Histogram)

Grafana และ `alert.rules.yml` ใน branch monitoring ปรับตาม contract นี้แล้ว. หลังสหรัฐ merge ให้เปิด
`/metrics` และตรวจว่าชื่อ/labels ตรงก่อน demo.

## 5. Reference data ของ Evidently

ตามเอกสารอัปเดต `data/processed/train.csv` คือ default reference สำหรับ Data Drift และ current มาจาก
production/test window. `run_evidently.py` จึง default ไปที่ train.csv แล้ว. ใน local demo ที่ไฟล์ processed
ยังไม่มา script สามารถ fallback ไป `monitoring/generated/reference.csv` พร้อม warning แต่ห้ามใช้ fallback
นั้นเป็นหลักฐาน final.

## 6. ทบทวน `monitoring.drift_share: 0.30`

Scenario ของโจทย์ inject โดยตรงที่ `Amount`, `V14`, `V17` = 3 จาก 30 model features หรือประมาณ 10%.
ดังนั้นอาจเกิดกรณี Evidently ชี้ root cause สามตัวนี้ชัด แต่ dataset drift share ยังไม่เกิน 0.30.

อย่าเปลี่ยนค่าเอง. ให้กลุ่มยืนยันว่าต้องการ:

- คง 0.30 เป็น broad-drift alert และใช้ feature-level result อธิบาย Period 3; หรือ
- recalibrate threshold จาก healthy windows เพื่อให้ตรง operational behavior ที่ต้องการ.

## 7. Pipeline contract กับธรรมรักษ์

เมื่อ monitoring ตัดสิน `RETRAIN` จะคืน exit code `2` และพิมพ์ `RETRAIN`. Pipeline ของธรรมรักษ์รับต่อด้วย:

```bash
python pipelines/flow.py --signal RETRAIN
```

จากนั้น challenger ต้องผ่าน gate ก่อน promote เป็น champion; ถ้าไม่ผ่าน champion เดิมต้องอยู่ต่อ.
Monitoring ไม่แก้ `pipelines/**` เอง.

## 8. รายงานใน `reports/monitoring/` อาจถูก `.gitignore`

ถ้า repo ยัง ignore `reports/*` ให้คุยหัวหน้ากลุ่มว่าจะเพิ่ม exception หรือใช้ `git add -f` เฉพาะไฟล์รายงาน
ที่ต้องส่ง. อย่าแก้ `.gitignore` จาก branch monitoring ถ้ายังไม่ได้รับอนุญาต.

## 9. AI_USAGE.md

งานกำหนดให้ระบุว่าใช้ AI ช่วยไฟล์ใดบ้าง แต่ `AI_USAGE.md` อยู่นอกขอบเขตของโชติกานต์ในรอบนี้.
ก่อน PR ให้เพิ่มรายการสำหรับ `simulate_drift.py`, `drift_check.py`, Evidently/NannyML, exporter,
Prometheus alerts และ Grafana dashboard และต้องอธิบายโค้ดเองได้ทุกส่วน.
