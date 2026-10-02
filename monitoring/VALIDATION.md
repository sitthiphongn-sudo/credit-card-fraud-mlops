# Monitoring validation notes

## ตรวจแล้วใน environment นี้

- `python -m compileall monitoring` — PASS.
- Python ทุกบรรทัดใน `monitoring/*.py` ไม่เกิน 110 ตัวอักษร — PASS.
- YAML ของ Prometheus/alerts/Grafana provisioning และ `docker-compose.yml` parse ได้ — PASS.
- Grafana dashboard JSON parse ได้ — PASS.
- `simulate_drift.py --input data/sample/valid.csv` — PASS:
  - Period 3 เปลี่ยน label = 0 แถว
  - Period 4 features เหมือนเดิมและ flip label เป็น fraud = 8 แถว
- Decision logic synthetic check — PASS:
  - healthy -> `OK`
  - drift only -> `WATCH`
  - concept/performance drop -> `RETRAIN`
  - exit code mapping เป็น `0 / 1 / 2`
- Prometheus exporter อ่าน config ปัจจุบันได้ และตั้ง `fraud_monitor_config_valid=0` ตามที่ตั้งใจ เพราะ shared
  config ยังไม่มี `max_error_rate`, `max_fraud_prediction_rate`, `stale_after_seconds`.

## สิ่งที่ยังรัน end-to-end ไม่ได้จาก ZIP ปัจจุบัน

- `reports/experiments/best_model.json` ยังไม่มีใน ZIP นี้. เอกสารกลุ่มระบุว่าจะมาจาก PR ของสิทธิพงษ์ก่อน
  monitoring merge. Monitoring ต้องใช้ไฟล์นี้เพื่ออ่าน threshold เดียวกับ production.
- `data/processed/train.csv` และ `test.csv` ยังไม่มี จึงยังทำ report final ที่ใช้ reference/current จริงไม่ได้.
- Evidently อยู่ใน `requirements.txt` แต่ package ไม่ได้ติดตั้งใน sandbox นี้ จึงยังไม่ได้ execute HTML report จริง.
- NannyML ยังไม่อยู่ใน `requirements.txt`. `nannyml==0.13.1` รองรับ Python 3.12 แต่ต้องการ
  `lightgbm>=3.3,<4.6` ขณะที่ project ตรึง `lightgbm==4.6.0`; ต้องให้ทีมแก้ dependency ร่วมกันก่อน.
- `ruff` อยู่ใน requirements แต่ไม่ได้ติดตั้งใน sandbox และ network ของ container ติดตั้งเพิ่มไม่ได้ จึงตรวจ
  actual `ruff check .` ไม่ได้ในที่นี่.
- `pytest -q` ของ repo ปัจจุบันหยุดตอน collect `tests/test_api.py` ด้วย `ModuleNotFoundError: serving`.
  ปัญหาเดียวกันเกิดกับ untouched repo ที่อัปโหลด จึงไม่ใช่ regression จาก monitoring branch.
- Serving `/metrics` ใน ZIP เก่ายังไม่ใช่ contract ล่าสุด. ต้องรอสหรัฐ merge metrics สี่ตัวตามเอกสารกลุ่ม.
- `models_cache/champion_model` ใน ZIP เก่ายังเป็น artifact รุ่นก่อน; เอกสารอัปเดตระบุ champion ใหม่เป็น `lightgbm_none`. อย่าใช้ cache เก่านี้เป็นผล final หลัง PR ใหม่ merge.

## สิ่งที่แก้ตามเอกสารอัปเดต 1 ต.ค. 2569 แล้ว

- เพิ่ม command หลัก `monitoring/drift_check.py --scenario data|concept`.
- RETRAIN ใช้ exit code `2`.
- prediction label ใช้ probability เทียบ model threshold จาก `best_model.json`; ไม่ใช้ `model.predict()` 0.5.
- Evidently default reference เป็น `data/processed/train.csv`.
- Baseline recall พยายามอ่าน `metrics.test_recall` จาก model hand-off ก่อน.
- Grafana เปลี่ยนไปใช้ `fraud_predictions_total{label=fraud|normal}` และ `fraud_score` ตาม metric contract ใหม่.
- Alert rules เพิ่ม `HighErrorRate` และ `FraudAlertRateSpike` โดย threshold มาจาก exporter/config ไม่ hardcode.

## ก่อนเปิด PR จริง

หลัง PR ของสิทธิพงษ์, ธรรมรักษ์ และสหรัฐ merge/rebase ตามลำดับ ให้รันใหม่ทั้งหมด:

```bash
ruff check .
pytest -q
python monitoring/drift_check.py --scenario normal1
python monitoring/drift_check.py --scenario normal2
python monitoring/drift_check.py --scenario data
python monitoring/drift_check.py --scenario concept
```

จากนั้นเปิด Prometheus/Grafana และเก็บหลักฐาน normal เงียบ, data drift root cause และ concept drift -> RETRAIN.
