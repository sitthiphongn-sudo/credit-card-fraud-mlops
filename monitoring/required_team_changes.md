# ข้อตกลงกับทีม (ปิดแล้ว)

| หัวข้อ | สถานะสุดท้าย |
|---|---|
| Model hand-off | มี `best_model.json` และ champion แล้ว |
| NannyML dependency | ใช้ environment แยก `.venv-nannyml` เพื่อหลีกเลี่ยง dependency conflict กับ main environment |
| Monitoring thresholds | ใช้ `drift_share: 0.10`, `max_recall_drop: 0.10`, `concept_gap: 0.10`, `max_error_rate: 0.01`, `stale_after_seconds: 86400` |
| API request metric | ใช้ `fraud_requests_total{status="success|bad_request|error"}` |
| Prediction metric | ใช้ `fraud_predictions_total{label="fraud|not_fraud"}` |
| Reference strategy | ใช้ fixed healthy Period 1 เป็น operational drift baseline |
| RETRAIN contract | `RETRAIN` ส่ง exit code `2` ให้ pipeline |
| Generated data | ไฟล์ใน `monitoring/generated/` ไม่ commit |
| AI usage | บันทึกใน `AI_USAGE.md` แล้ว |