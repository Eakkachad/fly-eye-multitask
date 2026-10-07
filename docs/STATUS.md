# สถานะโปรเจกต์ (อัปเดต 2026-10-07)

## ✅ เสร็จแล้ว
| งาน | หลักฐาน |
|---|---|
| แผนงานและสมมติฐานที่ลงทะเบียนไว้ล่วงหน้า | `PLAN.md` |
| ดาวน์โหลดข้อมูล Sintel + depth (23 ฉาก) และ render ลงตาข่ายหกเหลี่ยม | `data.py` |
| แบ่งข้อมูลตามตระกูลฉาก พร้อมเทสต์ว่าไม่รั่วข้ามชุด | `splits.py`, `splits.json`, `tests/` |
| Null connectome M2 (degree-preserving) และ M3 (สุ่ม) พร้อมเทสต์ | `nulls.py`, `connectomes/` |
| Baseline M4 (ปรับพารามิเตอร์ให้เท่า M1) และ M5 | `baselines/hex_models.py` |
| Training harness แบบ multi-task (flow + depth) ใช้ร่วมทุกโมเดล | `train.py`, `models.py` |
| สคริปต์ประเมิน test แบบรันครั้งเดียว + metric + ข้อมูลสำหรับ error analysis | `eval.py`, `evalmetrics.py` |
| สคริปต์ย้ายไปรันบนเครื่อง GPU อื่น (A100/SLURM) | `scripts/` |
| เทสต์ทั้งหมด **34 ผ่าน** | `pytest -q tests baselines/test_baselines.py` |

## 📊 ผลเบื้องต้น (smoke test, ยังไม่ใช่ผลจริง)
- ทุกโมเดล train ได้จริงบน RTX 4060 ด้วยความเร็ว 0.1–0.74 วินาทีต่อ iteration (ดู [MODELS.md](MODELS.md))
- **⚠️ ที่ 2,000 iteration ทุกโมเดลยังทำนาย flow ได้แค่ระดับ "ทำนายศูนย์"** (val EPE 5.03–5.08 เทียบกับ zero-flow 5.074) แม้ลอง learning rate ตั้งแต่ 5e-5 ถึง 2e-3
- **depth เริ่มเรียนรู้แล้ว:** val RMSE ลดจาก 1.87 เหลือราว 1.77
- **บริบท:** flyvis ที่ train เต็ม 250k iteration ในเปเปอร์ก็ชนะ zero-flow บนฉากใหม่เพียง ~4% แปลว่า flow บน Sintel ในความละเอียดของตาแมลงเป็นงานที่ยากจริง

## ⏭️ ขั้นต่อไป
1. **Learnability probe บน A100:** train M1 และ M4 ที่ 10k iteration
   - ถ้า EPE ต่ำกว่า zero-flow ชัดเจน ให้ใช้แผนเดิม
   - ถ้าไม่ต่ำกว่า ให้เปลี่ยนงาน flow เป็นวิดีโอสังเคราะห์ของ flyvis (moving edges/dots ที่รู้ flow แน่นอน) และคงงาน depth บน Sintel ไว้ ยังเป็น multi-task video เหมือนเดิม การเปลี่ยนนี้จะบันทึกเป็น amendment ใน `PLAN.md` **ก่อน** train ชุดเต็ม
2. **Train ชุดเต็ม:** 24 รอบ (`scripts/grid.tsv`) บน A100 80 GB
3. **ประเมิน:** ประเมิน test ครั้งเดียว ทำ error analysis และกราฟ
4. **นำเสนอ:** ทำสไลด์ และให้แต่ละคนเตรียมอธิบายส่วนของตัวเอง (ดู [TEAM.md](TEAM.md))

## ❓ เรื่องที่กลุ่มต้องตัดสินใจหรือยืนยัน
- วันส่ง proposal กับวันนำเสนอจริงคือวันไหน (ปรับตารางใน TEAM.md)
- ใครรับผิดชอบบทบาทไหน (ใส่ชื่อใน TEAM.md)
- ถ้าต้องใช้แผนสำรอง (วิดีโอสังเคราะห์) ทุกคนเห็นด้วยไหม
