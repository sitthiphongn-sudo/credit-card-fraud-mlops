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

## รันจากเครื่องใหม่

ต้องมี Docker Engine พร้อม Compose plugin, GNU Make, Git, อินเทอร์เน็ตสำหรับดาวน์โหลด image/แพ็กเกจ/ข้อมูล และพอร์ต 5000, 8000, 9090, 3000 ที่ว่าง จากโฟลเดอร์รีโปให้รัน:

```bash
make all
```

คำสั่งนี้สั่ง Compose สร้าง MLflow → ดาวน์โหลด [Credit Card Fraud Detection (ULB) จากแหล่งที่ TensorFlow ใช้](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) ถ้ายังไม่มี `data/raw/creditcard.csv` → ตรวจ schema → แบ่งตามเวลา → เทรน → ประเมินบนชุดทดสอบ → ตรวจ gate → ขึ้นทะเบียนและตั้ง alias `champion` → เริ่ม API, Prometheus และ Grafana เมื่อ API healthy ข้อมูลดิบไม่ถูก commit; hash และจำนวนแถวแยกชุดบันทึกใน `data/processed/data_version.json` ส่วน MLflow ใช้ named volume `mlflow-data` เพื่อคงทะเบียนและ artifacts หลัง restart

ถ้าเครื่องไม่มี Make ใช้คำสั่ง Compose ที่เทียบเท่า:

```bash
docker compose up -d --build --force-recreate --wait
```

`docker compose up -d --build` ยกทั้งระบบจากเครื่องใหม่ได้เช่นกัน แต่ `--wait` ทำให้ shell รอจนบริการพร้อมและคืนสถานะล้มเหลวหากเริ่มไม่สำเร็จ หากมี CSV อยู่แล้วและต้องการใช้ไฟล์นั้น ให้คัดลอกไปยัง `data/raw/creditcard.csv` ก่อนรัน; เวอร์ชันข้อมูลบันทึกจาก SHA-256 ของไฟล์ดิบ

ตรวจผล:

```bash
docker compose ps
curl http://localhost:8000/health
docker compose logs trainer
```

API อยู่ที่ `:8000`, MLflow `:5000`, Prometheus `:9090`, Grafana `:3000` (รหัสเริ่มต้นใน Compose สำหรับการสาธิตเท่านั้น) ถ้า gate ไม่ผ่าน trainer จะออกด้วย code 1 และ API ไม่เริ่ม; ดูเหตุผลใน `docker compose logs trainer` การหยุดใช้ `make down`

## Pull Request CI

ทุก PR รันสาม job แยกกัน: `code-quality` ตรวจ exact pins, Ruff, pytest; `data-validation` สร้างข้อมูลตัวอย่างสังเคราะห์ที่ทำซ้ำได้แล้วตรวจไฟล์ดีต้องผ่านและไฟล์เสียต้องออก code 1; `model-gate` เทรน Logistic Regression บนตัวอย่างนั้น ตรวจ recall, PR-AUC, p95 ของการทำนายหนึ่งรายการ และขนาดโมเดล ผ่าน `passes_gate()` ไฟล์ `reports/ci-model-gate.json` ถูกอัปโหลดเป็น artifact แม้ gate ล้มเหลว ตัวอย่างสังเคราะห์เป็น smoke test ของสาย CI; ผล gate บนข้อมูลจริงเกิดซ้ำใน trainer ก่อนขึ้นทะเบียนโมเดล

ตั้ง branch protection ใน GitHub Settings → Branches สำหรับ `main`: Require a pull request, 1 approval, และ required status checks `code-quality`, `data-validation`, `model-gate` หลังมี run แรก การตั้งค่านี้ต้องทำโดยผู้ดูแลรีโป แล้วเก็บ URL ของ run สีแดงและสีเขียวพร้อม artifact ไว้ในรายงาน ห้ามใช้ผล local แทนหลักฐาน GitHub Actions

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
