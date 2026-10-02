# บันทึกการใช้ AI ช่วยเขียนโค้ด

| วันที่ | ผู้ใช้ | เครื่องมือ | ส่วนที่ช่วย | ไฟล์ |
|---|---|---|---|---|
| 2026-09-16 | กันตพัฒน์ | Claude | สร้างโครง repo เริ่มต้น (โครงโฟลเดอร์, CI, schema, pipeline skeleton) | ทั้ง repo |

| 2026-09-27 | นาคินทร์ | Claude | เขียนโมดูลแปลงข้อมูล: เข้ารหัสเวลาแบบวงกลม (sin/cos), log1p+scale สำหรับ Amount, StandardScaler สำหรับ V1-V28 ที่ fit จากชุดฝึกเท่านั้น | src/fraud/features.py |
| 2026-09-27 | นาคินทร์ | Claude | เขียน unit test 13 ข้อ (3 ข้อบังคับตามเกณฑ์: DataFrame vs JSON, สลับลำดับคอลัมน์, joblib roundtrip + เทสต์เสริมเรื่องลด skew และกัน data leakage) | tests/test_features.py |
| 2026-09-27 | กันตพัฒน์ | Codex | ช่วยต่อ CI สาม job, ชุดข้อมูลทดสอบสังเคราะห์, model gate, pipeline จากข้อมูลดิบถึงทะเบียน, Compose และเอกสารการรันซ้ำ; ผู้ใช้ต้องตรวจและอธิบายโค้ดก่อนส่ง | `.github/workflows/ci.yml`, `scripts/`, `src/fraud/bootstrap.py`, `src/fraud/data.py`, `src/fraud/gate.py`, `pipelines/flow.py`, `docker-compose.yml`, `serving/Dockerfile`, `Makefile`, `.dockerignore`, `.gitattributes`, `tests/test_gate.py`, `README.md` |
| 2026-09-28 | นาคินทร์ | Claude | แก้ CI job model-gate ที่ติด ImportError: เพิ่มฟังก์ชัน `split_xy` และ `build_pipeline` (ต่อ FraudFeatureTransformer เข้ากับโมเดลเป็น Pipeline) ให้ตรงกับที่ train.py, pipelines/flow.py และ scripts/ci_model_gate.py เรียกใช้ และแก้ลำดับ import ตาม ruff (I001) ในไฟล์ทดสอบ | `src/fraud/features.py, tests/test_features.py` |
| 2026-09-28 | นาคินทร์ | Claude | เพิ่มการตรวจค่าที่หายไป (NaN/None) ใน `_validate_columns` ให้ raise ValueError ระบุคอลัมน์และจำนวนแถว พร้อมเทสต์ 3 ข้อ (DataFrame, JSON ทีละรายการ, fit) | `src/fraud/features.py, tests/test_features.py` |
| 2026-09-27 | เปรมสิริวัฒน์ | ChatGPT | ช่วยวางขั้นตอน EDA, data split/versioning, คำนวณ schema จาก train data, ปรับ validation, สร้าง sample data ปกติ/ผิดปกติ และตรวจ ruff/pytest ในไฟล์ `src/fraud/data.py`, `src/fraud/validate.py`, `scripts/eda.py`, `scripts/schema_stats.py`, `scripts/make_samples.py` | นำคำแนะนำมาปรับใช้และทดสอบจริงด้วย `ruff`, `pytest` และ validation test |

| 2026-09-28 | ธรรมรักษ์ | Codex | แนะนำการเขียน test, ตรวจ metrics ที่ขาด และเพิ่มคำสั่ง check/rollback สำหรับ model gate | `src/fraud/gate.py`, `tests/test_gate.py`, `Makefile` |
| 2026-09-28 | ธรรมรักษ์ | Codex | แนะนำฟังก์ชันลงทะเบียน challenger, promote champion และ unit tests แบบจำลอง MLflow | `src/fraud/gate.py`, `tests/test_gate.py` |
| 2026-09-28 | ธรรมรักษ์ | Codex | แนะนำการต่อ Prefect release flow สำหรับ register challenger, ตรวจ gate และ promote champion พร้อม unit tests | `pipelines/flow.py`, `pipelines/__init__.py`, `tests/test_flow.py`, `pyproject.toml` |
| 2026-09-28 | ธรรมรักษ์ | Codex | แนะนำ unit test สำหรับตรวจว่า rollback ย้าย alias champion ไปยัง version ที่ระบุ | `tests/test_gate.py` |
| 2026-09-28 | ธรรมรักษ์ | Codex | แนะนำการคำนวณขนาด MLflow model artifact และเชื่อม `model_mb` เข้า release flow พร้อม unit tests | `src/fraud/gate.py`, `pipelines/flow.py`, `tests/test_gate.py`, `tests/test_flow.py`, `docs/model_release.md` |
| 2026-09-28 | ธรรมรักษ์ | Codex | ช่วยร่างแผนภาพสถาปัตยกรรม MLOps และขอบเขตการเชื่อมงานของสมาชิก | `docs/architecture.md` |
| 2026-09-16 | ใหญ่ | Claude | ออกแบบโค้ด | ทั้ง repo |
| 2026-09-29 | ธรรมรักษ์ | Codex | ช่วยแก้ Pipeline ให้ตรวจข้อมูลดิบด้วย `validate_raw` และเพิ่ม unit test ยืนยันการเรียกใช้ | `pipelines/flow.py`, `tests/test_flow.py` |
| 2026-09-29 | ธรรมรักษ์ | Codex | ช่วยเชื่อม XGBoost experiment และแปลง test metrics จากงานโมเดลเข้าสู่ Model Gate พร้อม unit tests | `pipelines/flow.py`, `tests/test_flow.py` |
| 2026-09-29 | ธรรมรักษ์ | Codex | ช่วยเพิ่มการโหลด metrics ของ champion จาก MLflow เพื่อเปรียบเทียบกับ candidate ก่อนเลื่อน alias พร้อม unit tests | `src/fraud/gate.py`, `pipelines/flow.py`, `tests/test_gate.py`, `tests/test_flow.py` |

| 2026-10-01 | ธรรมรักษ์ | Codex | ช่วยออกแบบการรับสัญญาณ RETRAIN เพื่อเรียก Prefect training pipeline เพิ่มคำสั่ง Makefile เอกสาร และ unit tests โดยจำลองการเรียก DAG ไม่ให้เทรนจริง | `pipelines/flow.py`, `tests/test_flow.py`, `Makefile`, `README.md`, `docs/model_release.md` |
| 2026-10-02 | ธรรมรักษ์ | Codex | ช่วยปรับ Model Gate ให้ใช้ Recall at Precision >= 0.80 เชื่อมผลโมเดลที่ดีที่สุดเข้ากับ Prefect pipeline และปรับ unit tests | `configs/params.yaml`, `src/fraud/gate.py`, `pipelines/flow.py`, `tests/test_gate.py`, `tests/test_flow.py` |