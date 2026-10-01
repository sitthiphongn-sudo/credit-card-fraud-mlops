# Model Release Contract

เอกสารนี้กำหนดข้อมูลที่ใช้เชื่อมขั้น evaluate, load test, model gate และ MLflow Model Registry

## Release flow

1. ขั้น train คืนค่า MLflow `run_id`
2. ขั้น evaluate ส่ง `recall` และ `pr_auc` จากชุด test
3. ขั้น load test ส่ง `p95_ms`
4. Pipeline คำนวณ `model_mb` จาก MLflow model artifact
5. ลงทะเบียนโมเดลใหม่เป็น `challenger`
6. ตรวจ challenger ด้วย Model Gate
7. ถ้าผ่าน ให้เลื่อนเป็น `champion`
8. ถ้าไม่ผ่าน ให้คง champion เดิม

## Candidate metrics

ขั้น evaluate และ load test ส่ง `recall`, `pr_auc` และ `p95_ms`
เข้า release flow ส่วน Pipeline คำนวณ `model_mb` และเติมให้ก่อนตรวจ gate

| ชื่อ | แหล่งข้อมูล | ความหมาย |
|---|---|---|
| `recall` | ผลประเมินบนชุด test | สัดส่วน fraud ที่ตรวจพบ |
| `pr_auc` | ผลประเมินบนชุด test | พื้นที่ใต้ Precision-Recall curve |
| `p95_ms` | ผล load test ล่าสุด | เวลาตอบกลับ percentile 95 หน่วยมิลลิวินาที |
| `model_mb` | Pipeline คำนวณ | ขนาดรวมของ MLflow model artifact |

ข้อมูลที่ส่งเข้า release flow มีรูปแบบ:
`{"recall": 0.80, "pr_auc": 0.90, "p95_ms": 80}`

ข้อมูลที่ Model Gate ตรวจมีรูปแบบ:
`{"recall": 0.80, "pr_auc": 0.90, "p95_ms": 80, "model_mb": 20}`

## Champion comparison

Candidate และ champion ต้องประเมินบนชุด test เดียวกัน
เพื่อให้ค่า `pr_auc` เปรียบเทียบกันอย่างยุติธรรม

## Gate configuration

เกณฑ์ทั้งหมดอ่านจาก `configs/params.yaml`:

- `min_recall`
- `max_pr_auc_drop`
- `max_p95_latency_ms`
- `max_model_mb`

## Failure behavior

เมื่อ gate ไม่ผ่าน ระบบต้อง:

- แสดงเหตุผลทุกข้อที่ไม่ผ่าน
- คืน exit code `1`
- ไม่เปลี่ยน alias `champion`
- เก็บโมเดลใหม่เป็น `challenger` สำหรับตรวจย้อนหลัง

## Commands

ตรวจ candidate: `make gate CANDIDATE=reports/candidate_metrics.json`

เปรียบเทียบกับ champion:
`make gate CANDIDATE=reports/candidate_metrics.json CHAMPION=reports/champion_metrics.json`

ย้อนเวอร์ชัน: `make rollback VERSION=4`

## Automatic retraining

เมื่อระบบ monitoring ตรวจพบความผิดปกติตามนโยบาย ระบบจะส่งสัญญาณ
`RETRAIN` เพื่อเรียกวงจรฝึกโมเดลใหม่:

`make retrain`

คำสั่งที่เทียบเท่าบนเครื่องที่ไม่มี Make:

`python pipelines/flow.py --signal RETRAIN`

Pipeline จะรันตั้งแต่รับข้อมูล ตรวจคุณภาพ แบ่งชุด เทรน ประเมิน
ลงทะเบียนเป็น challenger และผ่าน Model Gate หากผ่านจึงเลื่อนเป็น
champion หากไม่ผ่าน Flow จะจบด้วยสถานะล้มเหลวและคง champion เดิม

## Current limitation

API ปัจจุบันโหลดโมเดลตอนเริ่ม service เท่านั้น หลังเปลี่ยน alias
`champion` ต้องมีขั้น reload หรือ restart ก่อน API จะใช้โมเดล version ใหม่