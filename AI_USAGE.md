# บันทึกการใช้ AI ช่วยเขียนโค้ด

| วันที่ | ผู้ใช้ | เครื่องมือ | ส่วนที่ช่วย | ไฟล์ |
|---|---|---|---|---|
| 2026-09-16 | กันตพัฒน์ | Claude | สร้างโครง repo เริ่มต้น (โครงโฟลเดอร์, CI, schema, pipeline skeleton) | ทั้ง repo |
| 2026-09-27 | เปรมสิริวัฒน์ | ChatGPT | ช่วยวางขั้นตอน EDA, data split/versioning, คำนวณ schema จาก train data, ปรับ validation, สร้าง sample data ปกติ/ผิดปกติ และตรวจ ruff/pytest ในไฟล์ `src/fraud/data.py`, `src/fraud/validate.py`, `scripts/eda.py`, `scripts/schema_stats.py`, `scripts/make_samples.py` | นำคำแนะนำมาปรับใช้และทดสอบจริงด้วย `ruff`, `pytest` และ validation test |
