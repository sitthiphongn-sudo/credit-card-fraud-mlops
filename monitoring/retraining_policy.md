# นโยบายการเทรนใหม่ (Retraining policy)

## การตัดสินใจของ monitoring

| สถานะ | เงื่อนไข | exit code | ผล |
|---|---|---|---|
| OK | drift share < 0.10 และ recall ไม่ตก | 0 | ไม่ทำอะไร |
| WATCH | drift share ≥ 0.10 แต่ recall จริงไม่ตก | 1 | แจ้งเตือน เฝ้าดูต่อ **ยังไม่ retrain** |
| RETRAIN | recall drop ≥ 0.10 หรือ concept gap ≥ 0.10 | 2 | เรียก pipeline เทรนใหม่ |

## ทำไม data drift อย่างเดียวไม่ retrain ทันที

Data drift หมายถึง distribution ของ input เปลี่ยน แต่ไม่ได้แปลว่าคุณภาพโมเดลลดลงเสมอไป

ในการทดสอบ Period 3 พบ feature drift แต่ recall จริงยังอยู่ประมาณ 0.91
จึงเลือกสถานะ `WATCH` แทน `RETRAIN` เพราะ input distribution เปลี่ยน แต่ realized recall ยังไม่ลดลงเกิน `max_recall_drop`

การ retrain ทุกครั้งที่ input เปลี่ยนทำให้ใช้ทรัพยากรโดยไม่จำเป็น
และโมเดลใหม่อาจไม่ได้ดีกว่า champion เดิม
จึงใช้ realized performance และ concept gap ประกอบการตัดสินใจด้วย

## ขั้นตอนเมื่อได้ RETRAIN

1. `python pipelines/flow.py --signal RETRAIN`
2. Pipeline รับข้อมูล → ตรวจ schema → แบ่งชุด → เทรน → ลงทะเบียนเป็น challenger
3. Model Gate ตรวจ: recall_at_p80 ≥ 0.75, PR-AUC ไม่แย่กว่า champion เกิน 0.005, p95 ≤ 100 ms, ขนาด ≤ 50 MB
4. ผ่าน → ย้าย alias `champion` ไปเวอร์ชันใหม่ · ไม่ผ่าน → คง champion เดิม
5. `docker compose restart api` เพื่อให้ API โหลดเวอร์ชันใหม่

## ถ้าโมเดลใหม่มีปัญหาหลังขึ้นใช้งาน

ใช้:

`python -m fraud.gate rollback <version>`

จากนั้น:

`docker compose restart api`

## หลักฐาน

ทดสอบ concept drift แล้ว Monitoring คืน `RETRAIN` พร้อม exit code `2`
จากนั้น `python pipelines/flow.py --signal RETRAIN` รันสำเร็จ
สร้าง `fraud-detector` version 2 และเลื่อน alias `champion` ไป version 2
