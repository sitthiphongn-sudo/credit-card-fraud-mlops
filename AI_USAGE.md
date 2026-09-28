# บันทึกการใช้ AI ช่วยเขียนโค้ด

| วันที่ | ผู้ใช้ | เครื่องมือ | ส่วนที่ช่วย | ไฟล์ |
|---|---|---|---|---|
| 2026-09-16 | กันตพัฒน์ | Claude | สร้างโครง repo เริ่มต้น (โครงโฟลเดอร์, CI, schema, pipeline skeleton) | ทั้ง repo |
| 2026-09-28 | ธรรมรักษ์ | Codex | แนะนำการเขียน test, ตรวจ metrics ที่ขาด และเพิ่มคำสั่ง check/rollback สำหรับ model gate | `src/fraud/gate.py`, `tests/test_gate.py` |