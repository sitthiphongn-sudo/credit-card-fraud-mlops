# คู่มือนำเสนอ Monitoring & Drift — โชติกานต์

## สคริปต์ประมาณ 1.5 นาที

ผมรับผิดชอบ Monitoring ทั้งฝั่งระบบและคุณภาพโมเดล. ผมจำลอง production 4 ช่วงตามเวลา: ช่วง 1-2 ปกติ,
ช่วง 3 สร้าง Data Drift โดยเพิ่ม Amount และ shift V14/V17 แต่ไม่เปลี่ยน label, ส่วนช่วง 4 สร้าง Concept
Drift โดยคง feature เดิมและเปลี่ยน label ของรายการยอดเล็กบางส่วนเป็น fraud.

ผมใช้ Evidently เทียบ current กับ reference training data เพื่อดู drift share และ feature ต้นเหตุ รวมถึงดู
prediction distribution และ realized performance เมื่อ label กลับมา. ตอน label ยังไม่มาใช้ NannyML CBPE
ประเมิน recall ล่วงหน้า. เมื่อ label กลับมา ผมเทียบ estimated กับ actual; ถ้า actual recall ลดจาก baseline
เกิน `max_recall_drop` หรือ estimated-actual gap เกิน `concept_gap` จะเป็น `RETRAIN`.

Model threshold ไม่ได้ใช้ 0.5 แต่ monitoring อ่าน threshold เดียวกับ production จาก `best_model.json`.
ผล batch ถูก export เป็น Prometheus Gauge แล้ว Grafana แสดง System Health กับ Model Health. ถ้าต้อง retrain
สคริปต์พิมพ์ `RETRAIN` และคืน exit code 2 เพื่อให้ pipeline ของธรรมรักษ์รับไปเทรน challenger และผ่าน gate
ก่อนขึ้น champion.

## คำถามสำคัญ

### Data Drift กับ Concept Drift ต่างกันอย่างไร?

Data Drift คือ `P(X)` เปลี่ยน: input distribution เปลี่ยน แต่ความสัมพันธ์ X->y อาจยังใช้ได้. Concept Drift คือ
`P(y|X)` เปลี่ยน: feature อาจหน้าตาเดิม แต่ mapping ไป label เปลี่ยนจนโมเดลเดิมผิดมากขึ้น. Period 3 เปลี่ยน X
โดย Class เดิม ส่วน Period 4 verify ว่า X เท่าเดิมแบบ exact equality แล้วเปลี่ยนเฉพาะ Class.

### ทำไม CBPE จับ Concept Drift ไม่ได้สมบูรณ์?

CBPE ไม่มี `y_true` ตอน estimate จึงอาศัย prediction probability/confidence และ behavior ที่เคยเรียนจาก
reference. ถ้า attacker เปลี่ยนความสัมพันธ์ X->y โดย X ยังเหมือนเดิม CBPE อาจยังดูปกติ. เพราะฉะนั้น CBPE
เป็น early estimate ไม่ใช่ ground truth; concept drift ต้องยืนยันด้วย realized performance เมื่อ label กลับมา.

### ถ้า label ช้าเป็นสัปดาห์จะรู้ได้อย่างไร?

ระหว่างรอ label ดู Evidently feature drift, `fraud_predictions_total`, `fraud_score` และ CBPE estimated
performance. ถ้าเริ่มผิดปกติให้ WATCH/ตรวจเพิ่ม. เมื่อ label จริงมา จึงคำนวณ recall/precision และตัดสิน RETRAIN.

### Threshold มาจากไหน?

Operational threshold อ่านจาก `configs/params.yaml` ที่เดียว ไม่ hardcode ซ้ำใน scripts/PromQL. ค่าที่ทีม
กำหนดต้องอธิบายจาก healthy Period 1-2, business impact หรือ SLO. Model threshold อ่านจาก model hand-off
เพื่อให้ monitoring ตัดสินเหมือน API จริง.

### ทำไมมี stale monitoring alert?

Prometheus อาจยังแสดงค่ารอบเก่าแม้ batch monitoring ตาย. จึง export `last_run_timestamp` แล้วเตือนเมื่อ
ผลเก่าเกิน freshness window; ไม่อย่างนั้น dashboard สีเขียวอาจทำให้เข้าใจผิด.

### Exit code มีความหมายอย่างไร?

`0=OK`, `1=WATCH`, `2=RETRAIN`. เอกสารกลุ่มกำหนด RETRAIN เป็น exit code 2 เพื่อส่งต่อ pipeline.

## Metric contract ที่ควรจำ

- `fraud_requests_total{status=success|bad_request|error}`
- `fraud_request_latency_seconds`
- `fraud_predictions_total{label=fraud|not_fraud}`
- `fraud_score`
