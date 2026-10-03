# เกณฑ์แจ้งเตือน (Alert thresholds)

ค่า threshold ทั้งหมดอ่านจาก `configs/params.yaml` ผ่าน `drift_exporter.py`
ไม่ hardcode ใน PromQL

## สถานะระบบ (System health)

| Alert | ดูจาก | เกณฑ์ | ระดับ | ต้องทำ | เหตุผล |
|---|---|---|---|---|---|
| FraudApiMetricsUnavailable | `up{job="fraud-api"}` | = 0 | critical | ตรวจ container API และ `/metrics` | ถ้า API metrics หาย จะไม่สามารถประเมินสถานะ production และคุณภาพการให้บริการได้ |
| MonitoringExporterUnavailable | `up{job="fraud-monitoring-batch"}` | = 0 | critical | ตรวจ `drift_exporter.py` | ถ้า exporter หยุด Prometheus จะไม่ได้ค่า drift/recall ล่าสุด ทำให้ dashboard อาจแสดงข้อมูลเก่า |
| HighLatencyP95 | p95 ของ `fraud_request_latency_seconds` | > 100 ms (`gate.max_p95_latency_ms`) | warning | ตรวจโหลด, worker, ทรัพยากร | ใช้ SLO เดียวกับ model gate เพื่อให้ latency ที่อนุมัติตอน release สอดคล้องกับ production monitoring |
| HighErrorRate | สัดส่วน `fraud_requests_total{status="error"}` | > 1% (`max_error_rate: 0.01`) | critical | ดู log API ก่อนเชื่อผลทำนาย | error เกิน 1% อาจทำให้ผู้ใช้ไม่ได้รับผลทำนายและทำให้ production metrics ไม่น่าเชื่อถือ |
| MonitoringConfigInvalid | `fraud_monitor_config_valid` | = 0 | critical | ตรวจส่วน monitoring ใน params | ป้องกันระบบ monitoring ทำงานด้วย config ที่ไม่ครบหรือ fallback ไปใช้ค่าที่ไม่ได้ตกลงกับทีม |
| DriftMonitoringStale | เวลาตั้งแต่ batch รันล่าสุด | > 86400 วินาที (`stale_after_seconds`) | critical | ตรวจ batch job | ป้องกัน dashboard เขียวค้างจากผล monitoring รอบเก่าที่ไม่ได้สะท้อนสถานะปัจจุบัน |

## คุณภาพการทำนาย (Model health)

| Alert | ดูจาก | เกณฑ์ | ระดับ | ต้องทำ | เหตุผล |
|---|---|---|---|---|---|
| FeatureDriftShareHigh | `fraud_monitor_drift_share` | ≥ 0.10 (`drift_share`) | warning | ดู feature ที่ drift ใน Evidently | ใช้เป็นสัญญาณเตือนเมื่อ feature หลายตัวเริ่มเปลี่ยน แต่ยังไม่ retrain ทันทีเพราะ performance อาจยังดี |
| RecallDropCritical | `fraud_monitor_recall_drop` | ≥ 0.10 (`max_recall_drop`) | critical | ส่ง RETRAIN | Recall สำคัญกับ fraud detection เพราะถ้าลดลงจะพลาด fraud จริงมากขึ้น จึงใช้เป็นเงื่อนไขสำคัญในการ retrain |
| ConceptDriftSuspected | `fraud_monitor_concept_gap` (CBPE − recall จริง) | ≥ 0.10 (`concept_gap`) | critical | ยืนยัน label แล้ว RETRAIN | gap สูงหมายถึง performance จริงต่างจากค่าที่คาดมาก เป็นสัญญาณว่าความสัมพันธ์ระหว่าง feature กับ label อาจเปลี่ยน |
| FraudAlertRateSpike | สัดส่วน `fraud_predictions_total{label="fraud"}` | `max_fraud_prediction_rate` | warning | ดู `fraud_score` + รัน drift check | ปิดอยู่เพราะยังไม่ได้ตั้ง threshold ใน params เพื่อหลีกเลี่ยงการสร้าง false alert จากค่าที่ทีมยังไม่ยืนยัน |
