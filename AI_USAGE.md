# บันทึกการใช้ AI ช่วยเขียนโค้ด

| วันที่ | ผู้ใช้ | เครื่องมือ | ส่วนที่ช่วย | ไฟล์ |
|---|---|---|---|---|
| 2026-09-16 | กันตพัฒน์ | Claude | สร้างโครง repo เริ่มต้น (โครงโฟลเดอร์, CI, schema, pipeline skeleton) | ทั้ง repo |
| 2026-09-27 | กันตพัฒน์ | Codex | ช่วยต่อ CI สาม job, ชุดข้อมูลทดสอบสังเคราะห์, model gate, pipeline จากข้อมูลดิบถึงทะเบียน, Compose และเอกสารการรันซ้ำ; ผู้ใช้ต้องตรวจและอธิบายโค้ดก่อนส่ง | `.github/workflows/ci.yml`, `scripts/`, `src/fraud/bootstrap.py`, `src/fraud/data.py`, `src/fraud/gate.py`, `pipelines/flow.py`, `docker-compose.yml`, `serving/Dockerfile`, `Makefile`, `.dockerignore`, `.gitattributes`, `tests/test_gate.py`, `README.md` |
