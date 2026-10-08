# สถานะโปรเจกต์ (อัปเดต 2026-10-08)

**สรุป:** ระบบพร้อมและ grid 39 รอบกำลังรันบนเครื่องเดียว (RTX 4060) ตั้งแต่ 2026-10-08 09:35 ยังไม่มีผลบน test set (test ประเมินครั้งเดียวหลัง grid จบ)

## ✅ เสร็จแล้ว
| งาน | หลักฐาน |
|---|---|
| แผนงานและสมมติฐานที่ลงทะเบียนไว้ล่วงหน้า (H1–H7, amendment A1–A7) | `PLAN.md` |
| ดาวน์โหลดข้อมูล Sintel + depth (23 ฉาก) และ render ลงตาข่ายหกเหลี่ยม | `data.py` |
| แบ่งข้อมูลตามตระกูลฉาก พร้อมเทสต์ว่าไม่รั่วข้ามชุด | `splits.py`, `splits.json`, `tests/` |
| Null connectome M2 (degree-preserving) และ M3 (สุ่ม) พร้อมเทสต์ | `nulls.py`, `connectomes/` |
| Baseline M4 (ปรับพารามิเตอร์ให้เท่า M1) และ M5 | `baselines/hex_models.py` |
| โมเดลใหม่: hybrid M6/M7, frozen-residual M6f/M7f, HexConvGRU-K M8 | `models.py`, PLAN A5/A6 |
| Training harness แบบ multi-task (flow + depth) ใช้ร่วมทุกโมเดล | `train.py`, `models.py` |
| Floors (เส้นฐานไม่ต้อง train) และ claim ladder | `floors.py`, PLAN A6.1 |
| สคริปต์ประเมิน test แบบรันครั้งเดียว + metric + ข้อมูลสำหรับ error analysis | `eval.py`, `evalmetrics.py` |
| เร่งความเร็ว: `fastfly` (Triton fused rollout), fused penalty, `torch.compile` | `fastfly/`, [R2_BENCHMARK.md](R2_BENCHMARK.md) |
| Learnability probe (A4): **ผ่าน** | [PROBE_C014.md](PROBE_C014.md) |
| สคริปต์รัน grid บนเครื่อง local (resume ได้ + watchdog) | `scripts/run_grid_local.sh`, `scripts/grid_v2.tsv` |
| เทสต์ **47 ผ่านบน CPU** (+ `tests/test_fastfly.py` ผ่าน Triton interpreter ~2 นาที) | `pytest -q tests baselines/test_baselines.py --ignore=tests/test_fastfly.py` |

## 📊 ผลที่มีตอนนี้ (val เท่านั้น ยังไม่ใช่ผลสรุป)
**Learnability probe (A4, seed เดียว, 10k iteration, lr 5e-4):** best val EPE ของ M1 = **4.915**, M4 = **4.501** เทียบ zero-flow = 5.074 → ผ่านเกณฑ์ จึงคง flow บน Sintel ไว้ (val มี noise และเป็น seed เดียว ไม่ใช่การทดสอบ H1–H3)

**Floors บน val (A6.1):**
| เส้นฐาน | ค่า |
|---|---|
| flow: zero-flow EPE | 5.074 |
| flow: Lucas–Kanade บนตาข่ายหกเหลี่ยม (แผนที่หน่วย 2×2 fit บน train, window/lambda จูนบน val) | 4.709 |
| flow: train-mean flow | 5.288 |
| flow: oracle constant-velocity (**oracle**, ไม่ใช่คู่แข่ง) | 1.316 |
| depth: train-mean RMSE | 1.863 |
| depth: per-hexal train-mean RMSE | 1.856 |

- LK ชนะ M1 ใน probe; M4 ใน probe ชนะ LK ราว 4% โมเดลจะรายงานเป็น **margin เหนือ floor ที่ดีที่สุดที่ไม่ใช่ oracle**
- **ความเร็ว (R2):** `fastfly` เร่ง M1 จาก 0.177 เป็น 0.074 s/iter (2.40x; 2.7x เทียบ implementation เดิม 0.202) ตรวจ parity แล้ว; `torch.compile` เร่ง M4 5x (0.030 s/iter ใน train.py), M5 3.8x; M8 0.064 s/iter หลัง compile ~25 นาที; bf16/tf32 ไม่ช่วย
- **บริบท:** flyvis ที่ train เต็ม 250k iteration ในเปเปอร์ชนะ zero-flow บนฉากใหม่เพียง ~4% flow บน Sintel ในความละเอียดของตาแมลงจึงเป็นงานที่ยาก

## 🔄 กำลังทำ
- **Grid 39 รอบ** (`scripts/grid_v2.tsv`): lr 5e-4, 30k iteration, 3 seed ต่อโมเดล รันบน RTX 4060 ตั้งแต่ 2026-10-08 09:35 ประมาณการ ~31–35 ชั่วโมง
- **หยุดชั่วคราว:** สร้างไฟล์ `~/flyproj/.orchestra/GRID_PAUSE` (จะหยุดก่อนรอบถัดไป) · **รันต่อ:** ลบไฟล์นั้นแล้วรัน `scripts/run_grid_local.sh` ใหม่ (ข้ามรอบที่เสร็จแล้ว)

## ⏭️ ขั้นต่อไป
1. รอ grid จบ แล้วประเมิน test ครั้งเดียว (`eval.py`) และรัน `floors.py --split test --once`
2. สรุปผลตาม claim ladder L1/L2/L3 พร้อม margin เหนือ floor
3. Diagnostics เชิงสำรวจ (L3): CKA, front-end necessity, เสถียรภาพของ state, ablation-superposition, divergence/curl
4. Error analysis, กราฟ, สไลด์ และให้แต่ละคนเตรียมอธิบายส่วนของตัวเอง (ดู [TEAM.md](TEAM.md))

## ⚠️ ข้อจำกัดที่ต้องรายงาน
- ไม่ได้จูน lr แยกต่อโมเดล (งบเวลา) ใช้ lr 5e-4 เท่ากันทุกรอบ
- ผล probe เป็น seed เดียวบน val; M4 > M1 ใน probe เป็นสัญญาณเตือนว่า H2 อาจไม่ผ่าน (ดู PROBE_C014.md)
- M6/M7 เพิ่มหลังเห็นผล probe (post-probe แต่ก่อน grid) ระบุไว้ใน PLAN A5

## ❓ เรื่องที่กลุ่มต้องตัดสินใจหรือยืนยัน
- วันส่ง proposal กับวันนำเสนอจริงคือวันไหน (ปรับตารางใน TEAM.md)
- ใครรับผิดชอบบทบาทไหน (ใส่ชื่อใน TEAM.md)
