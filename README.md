# Credit Card Fraud Detection — MLOps

โครงงานรายวิชา CP413008 Machine Learning Engineering for Production · กลุ่ม 15 **The bid**

ระบบให้คะแนนความเสี่ยงธุรกรรมบัตรเครดิตแบบเรียลไทม์ ครบวงจรตั้งแต่ข้อมูลดิบ → ตรวจคุณภาพ → เทรน → ประเมิน → ด่านตรวจ → ทะเบียนโมเดล → ให้บริการ → เฝ้าระวัง → เทรนใหม่

![pipeline](docs/pipeline_diagram.png)

## สมาชิกและส่วนที่รับผิดชอบ

| สมาชิก | ส่วนที่รับผิดชอบ | โฟลเดอร์หลัก |
|---|---|---|
| กันตพัฒน์ | หัวหน้ากลุ่ม · Repo & CI/CD | `.github/` |
| เปรมสิริวัฒณ์ | Data & Validation | `src/fraud/data.py`, `src/fraud/validate.py` |
| นาคินทร์ | Feature Engineering | `src/fraud/features.py` |
| สิทธิพงษ์ | Modeling & Experiments | `src/fraud/train.py`, `src/fraud/evaluate.py` |
| สหรัฐ | Serving & Performance | `serving/` |
| โชติกานต์ | Monitoring & Drift | `monitoring/` |
| ธรรมรักษ์ | Pipeline & Model Registry | `pipelines/`, `src/fraud/gate.py` |

## ชุดข้อมูล

[Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 รายการ, fraud 492 (0.172%)
ดาวน์โหลด `creditcard.csv` แล้ววางที่ `data/raw/creditcard.csv` (ไฟล์ข้อมูลไม่ถูก commit)

## เริ่มใช้งาน

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

make validate   # ตรวจ schema ข้อมูลดิบ
make all        # รันทั้ง pipeline ด้วยคำสั่งเดียว
make retrain    # รับสัญญาณ RETRAIN แล้วรันวงจรฝึกและตรวจโมเดลใหม่
docker compose up -d --build   # mlflow :5000 · api :8000 · prometheus :9090 · grafana :3000
```

## โครงสร้าง

```
configs/params.yaml     เกณฑ์และค่าคงที่ทั้งหมด (ที่เดียว)
src/fraud/              โค้ดหลัก: data, validate, features, train, evaluate, gate
pipelines/flow.py       Prefect DAG: ingest → validate → split → train → evaluate → gate → register
serving/                FastAPI + Dockerfile (/predict /health /metrics)
monitoring/             Prometheus, alert rules, Grafana, จำลอง drift
tests/                  unit tests (รันใน CI)
docs/                   แผนงาน และแผนภาพสถาปัตยกรรม
```

## วิธีทำงานร่วมกัน

1. ห้าม push เข้า `main` ตรง — แตก branch ตามรูปแบบ `feat/<ส่วน>-<เรื่อง>` เช่น `feat/data-schema`
2. commit บ่อย ๆ ด้วยบัญชีตัวเอง (คะแนนรายบุคคลดูจากประวัติ commit)
3. เปิด Pull Request → CI ต้องผ่าน → มีเพื่อน review อย่างน้อย 1 คน → merge
4. ถ้าใช้ AI ช่วยเขียนโค้ด ให้บันทึกใน `AI_USAGE.md`
