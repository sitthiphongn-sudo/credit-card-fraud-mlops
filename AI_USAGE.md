# บันทึกการใช้ AI ช่วยเขียนโค้ด

| วันที่ | ผู้ใช้ | เครื่องมือ | ส่วนที่ช่วย | ไฟล์ |
|---|---|---|---|---|
| 2026-09-16 | กันตพัฒน์ | Claude | สร้างโครง repo เริ่มต้น (โครงโฟลเดอร์, CI, schema, pipeline skeleton) | ทั้ง repo |
| 2026-09-27 | นาคินทร์ | Claude | เขียนโมดูลแปลงข้อมูล: เข้ารหัสเวลาแบบวงกลม (sin/cos), log1p+scale สำหรับ Amount, StandardScaler สำหรับ V1-V28 ที่ fit จากชุดฝึกเท่านั้น | src/fraud/features.py |
| 2026-09-27 | นาคินทร์ | Claude | เขียน unit test 13 ข้อ (3 ข้อบังคับตามเกณฑ์: DataFrame vs JSON, สลับลำดับคอลัมน์, joblib roundtrip + เทสต์เสริมเรื่องลด skew และกัน data leakage) | tests/test_features.py |