# Model Release Contract

เอกสารนี้กำหนดข้อมูลที่เชื่อมการฝึกและประเมินโมเดลกับ Model Gate
และ MLflow Model Registry

## Release flow

1. Pipeline รับข้อมูล ตรวจคุณภาพ และแบ่ง train/validation/test
2. รันการทดลองและเลือกโมเดลจากผล validation
3. บันทึกผลการทดลองและโมเดลที่เลือกใน `reports/experiments/best_model.json`
4. คืนค่า MLflow `run_id` และแปลงผลประเมินเป็น candidate metrics
5. เติม `model_mb` จาก MLflow model artifact หากผลการทดลองยังไม่มี
6. ลงทะเบียนโมเดลใหม่เป็น `challenger`
7. ตรวจ candidate ด้วย Model Gate และเปรียบเทียบกับ champion ปัจจุบัน
8. ถ้าผ่าน ให้ย้าย alias `champion` ไปยังเวอร์ชันใหม่
9. ถ้าไม่ผ่าน ให้คง champion เดิมและแสดงเหตุผล

## Candidate metrics

Pipeline ใช้ `candidate_from_best_model()` แปลงข้อมูลในช่อง `metrics`
ของ `best_model.json` เป็นชื่อที่ Model Gate ใช้:

| ค่าใน best_model.json | ชื่อที่ Gate ใช้ | ความหมาย |
|---|---|---|
| `test_recall_at_p80` | `recall_at_p80` | recall สูงสุดบนชุด test ที่ precision >= 0.80 |
| `test_pr_auc` | `pr_auc` | พื้นที่ใต้ Precision-Recall curve บนชุด test |
| `latency_p95_ms` | `p95_ms` | เวลา inference percentile 95 หน่วยมิลลิวินาที |
| `model_mb` | `model_mb` | ขนาดรวมของ MLflow model artifact หน่วย MB |

ตัวอย่างข้อมูลที่ Model Gate ตรวจ:

```json
{
  "recall_at_p80": 0.80,
  "pr_auc": 0.90,
  "p95_ms": 80,
  "model_mb": 20
}
```

ต้องมี metrics ครบทั้ง 4 ค่า หากขาดค่าใด Gate จะไม่ผ่าน

`p95_ms` ใน release pipeline มาจากการวัด inference ของโมเดล
ส่วน latency และ throughput ของ API ต้องวัดด้วย serving load test แยกต่างหาก

`recall_at_p80` เป็นตัวชี้วัดคุณภาพสำหรับ Gate ส่วน threshold ที่ API
ใช้ทำนายอ่านจาก parameter `threshold` ของ MLflow run ของ champion

## Champion comparison

Candidate และ champion ควรประเมินบนชุด test เดียวกัน
เพื่อให้ค่า `pr_auc` เปรียบเทียบกันอย่างยุติธรรม

Release pipeline โหลด metrics ของ champion ปัจจุบันจาก MLflow
หากยังไม่มี champion จะตรวจเฉพาะเกณฑ์ของ candidate

## Gate configuration

เกณฑ์อ่านจากส่วน `gate` ใน `configs/params.yaml`:

| Configuration | เกณฑ์ |
|---|---|
| `min_recall_at_p80` | `recall_at_p80` >= 0.75 |
| `max_pr_auc_drop` | PR-AUC ต่ำกว่า champion ได้ไม่เกิน 0.005 |
| `max_p95_latency_ms` | p95 <= 100 ms |
| `max_model_mb` | ขนาด artifact <= 50 MB |

## Failure behavior

เมื่อ Gate ไม่ผ่าน:

- แสดงเหตุผลที่ไม่ผ่าน
- คำสั่ง CLI `check` คืน exit code `1`
- Release flow จบด้วยสถานะล้มเหลว
- ไม่เปลี่ยน alias `champion`
- โมเดลที่ลงทะเบียนใน release flow ยังคงเป็น `challenger` สำหรับตรวจย้อนหลัง

คำสั่ง CLI `check` ตรวจ metrics เท่านั้น ไม่ลงทะเบียนหรือเลื่อนโมเดล

## Start MLflow and run the pipeline

เปิด MLflow:

```powershell
docker compose up -d mlflow
```

Compose นี้ใช้พอร์ต `5001` บนเครื่อง เพื่อหลีกเลี่ยงปัญหาพอร์ต `5000`
ส่วน MLflow ภายใน container ยังใช้พอร์ต `5000`

ตั้งค่าใน PowerShell ทุกครั้งที่เปิด Terminal ใหม่:

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
```

ต้องมีข้อมูลดิบที่ `data/raw/creditcard.csv` ก่อนรัน:

```powershell
python pipelines/flow.py
```

คำสั่งที่เทียบเท่าเมื่อมี Make:

```text
make all
```

เปิด MLflow UI ที่ `http://localhost:5001`
และตรวจโมเดลที่ Models → fraud-detector

หลังมี champion แล้ว เริ่ม API:

```powershell
docker compose up -d --build api
```

ตรวจว่า API โหลดโมเดลสำเร็จ:

```powershell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

## Gate commands

ตรวจโมเดลที่เลือก:

```powershell
python -m fraud.gate check --candidate reports/experiments/best_model.json
```

คำสั่งที่เทียบเท่าเมื่อมี Make:

```text
make gate CANDIDATE=reports/experiments/best_model.json
```

หากต้องการระบุ metrics ของ champion จากไฟล์:

```powershell
python -m fraud.gate check --candidate reports/experiments/best_model.json --champion reports/champion_metrics.json
```

ต้องเตรียม `reports/champion_metrics.json` เองก่อนใช้คำสั่งนี้
ส่วน release pipeline โหลด champion จาก MLflow ให้อัตโนมัติ

ตรวจ exit code ทันทีหลังคำสั่งใน PowerShell:

```powershell
$LASTEXITCODE
```

PASS คืน `0` และ FAIL คืน `1`

## Automatic retraining

Monitoring ส่งสัญญาณ `RETRAIN` เมื่อเข้าเงื่อนไขตามนโยบาย
จากนั้นเรียกคำสั่ง:

```powershell
python pipelines/flow.py --signal RETRAIN
```

คำสั่งที่เทียบเท่าเมื่อมี Make:

```text
make retrain
```

Pipeline รันตั้งแต่รับข้อมูลจนถึง Model Gate
หากผ่านจึงเลื่อนเวอร์ชันใหม่เป็น champion
หากไม่ผ่านจะคง champion เดิม

## Rollback

ต้องมีโมเดลอย่างน้อย 2 เวอร์ชันเพื่อสาธิตการย้อนกลับ

1. ตรวจ `/health` และบันทึก `model_version` ปัจจุบัน
2. ย้อน alias champion ไปยังเวอร์ชันที่ต้องการ เช่น Version 1:

```powershell
python -m fraud.gate rollback 1
```

คำสั่งที่เทียบเท่าเมื่อมี Make:

```text
make rollback VERSION=1
```

3. ตรวจใน MLflow ว่า alias `champion` ย้ายไป Version 1
4. Restart API เพื่อโหลด champion ใหม่:

```powershell
docker compose restart api
```

5. รอให้ API พร้อม แล้วตรวจ:

```powershell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

ผลต้องเป็น `healthy` และ `model_version` ตรงกับเวอร์ชันที่ย้อนกลับ
threshold จะอ่านจาก MLflow run ของเวอร์ชันนั้นด้วย

## Current limitation

API โหลดโมเดลและ threshold ตอนเริ่ม service
หลัง promote หรือ rollback ต้อง restart API ก่อนใช้ champion ที่เปลี่ยนไป

## Evidence

เก็บหลักฐานต่อไปนี้:

- Pipeline จบด้วยสถานะ Completed
- MLflow Registry แสดงเวอร์ชันและ alias champion
- API health แสดง healthy, model_version และ threshold
- Gate FAIL พร้อม exit code 1 และ PASS พร้อม exit code 0
- RETRAIN สร้างเวอร์ชันใหม่และผ่าน Gate
- Rollback: API ก่อนย้อน, alias ใน MLflow หลังย้อน และ API หลังย้อน