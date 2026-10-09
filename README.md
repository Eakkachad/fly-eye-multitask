# Fly-Eye Multi-Task Video
### ใช้ wiring ของสมองแมลงหวี่เป็น prior ให้ deep learning: ทำนาย optic flow และ depth จากวิดีโอพร้อมกัน

> โปรเจกต์วิชา **Deep Learning for Image Analysis** · ประเภทงาน **Multi-task Learning + Video Analysis**
> สถานะ (2026-10-08): probe ผ่านแล้ว, grid 39 รอบกำลังรันบน RTX 4060 (~31–35 ชม.), ยังไม่มีผลบน test → ดู [docs/STATUS.md](docs/STATUS.md)

## สรุปใน 30 วินาที
ให้โมเดลดูวิดีโอสั้น ๆ ผ่าน "ตาแมลงหวี่" (ตาข่ายหกเหลี่ยม 721 จุด, 19 เฟรม) แล้วทำนาย **2 งานพร้อมกัน** ในทุกจุดภาพ:
1. **Optic flow:** แต่ละจุดเคลื่อนที่ไปทางไหน เร็วแค่ไหน
2. **Depth:** แต่ละจุดอยู่ไกลแค่ไหน

เราเปรียบเทียบโมเดล 5 แบบหลัก (M1–M5) บวกโมเดลเสริม M6/M7/M6f/M7f/M8 ภายใต้เงื่อนไขเดียวกัน เพื่อตอบคำถามว่า **"wiring จริงจากสมองแมลงหวี่ (connectome) ช่วยให้เรียนรู้ได้ดีขึ้นหรือใช้ข้อมูลน้อยลงหรือไม่"**

| ID | โมเดล | พารามิเตอร์ |
|---|---|---|
| M1 | flyvis: wiring จาก connectome จริง | 15,387 |
| M2 | flyvis: สลับสายแบบเก็บ degree (null) | 15,387 |
| M3 | flyvis: สุ่มสายทั้งหมด (null) | 15,387 |
| M4 | HexConvGRU (deep learning มาตรฐาน, ขนาดเท่า M1) | 15,402 |
| M5 | HexConvGRU ขนาดใหญ่ | 604,835 |
| M6 / M7 | hybrid: front-end จาก connectome จริง / rewired + HexConvGRU | 15,423 |
| M6f / M7f | front-end M1/M2 ที่ freeze + trunk residual (zero-init) | 14,689 (train ได้) |
| M8 | HexConvGRU-K (วน GRU K=1..4 ครั้งต่อเฟรม) | 15,402 |

- **ข้อมูล:** MPI Sintel (ภาพยนตร์ CG ที่มี ground truth ของ flow และ depth) **แบ่ง train/val/test ตามตระกูลฉาก** เพื่อไม่ให้ข้อมูลรั่วข้ามชุด
- **ตัวชี้วัด:** EPE และ angular error สำหรับ flow, RMSE และ AbsRel สำหรับ depth
- **Protocol:** 3 seed ต่อโมเดล และประเมิน test ครั้งเดียว

## เอกสาร (อ่านตามลำดับนี้)
| เอกสาร | เนื้อหา | ใครควรอ่าน |
|---|---|---|
| [docs/PROPOSAL.md](docs/PROPOSAL.md) | ร่าง proposal ครบทุกหัวข้อตามเกณฑ์ | ทุกคน, อาจารย์ |
| [docs/BACKGROUND.md](docs/BACKGROUND.md) | connectome, flyvis, null model, คำศัพท์ | ทุกคน (เริ่มที่นี่ถ้ายังไม่คุ้น) |
| [docs/DATA.md](docs/DATA.md) | ข้อมูลส่วนไหนใช้หรือไม่ใช้, การแปลง, การแบ่งชุด | ฝ่ายข้อมูล |
| [docs/MODELS.md](docs/MODELS.md) | รายละเอียดโมเดล M1–M8 และความเป็นธรรมของการเปรียบเทียบ | ฝ่ายโมเดล |
| [docs/EVALUATION.md](docs/EVALUATION.md) | ตัวชี้วัด, protocol, error analysis, ข้อจำกัด | ฝ่ายประเมินผล |
| [docs/TEAM.md](docs/TEAM.md) | บทบาท 8 คน, กำหนดการ, คำถามที่น่าจะโดนถาม | ทุกคน |
| [docs/METHODS_OVERVIEW.md](docs/METHODS_OVERVIEW.md) | วิธีการที่ implement และสถานะโปรเจกต์ (ภาพรวม, แนวโน้ม val, ผลที่คาดหวัง) | ทุกคน, อาจารย์ |
| [docs/STATUS.md](docs/STATUS.md) | อะไรเสร็จแล้ว, ผลเบื้องต้น, ความเสี่ยง, ขั้นต่อไป | ทุกคน |
| [docs/PROBE_C014.md](docs/PROBE_C014.md) | ผล learnability probe (A4) | ทุกคน |
| [docs/R2_BENCHMARK.md](docs/R2_BENCHMARK.md) | เบนช์มาร์กความเร็ว (fastfly, compile) | ฝ่าย engineering |
| [docs/RUNNING.md](docs/RUNNING.md) | วิธีติดตั้ง, รัน, ประเมิน, reproduce (ภาษาอังกฤษ) | ฝ่าย engineering |
| [PLAN.md](PLAN.md) | แผนและสมมติฐานที่ลงทะเบียนล่วงหน้า (ต้นฉบับ, ภาษาอังกฤษ) | อ้างอิง |

## โครงสร้าง repo
```
data.py, splits.py, splits.json   ข้อมูลและการแบ่งชุด (ตามตระกูลฉาก)
nulls.py, connectomes/            null connectome M2/M3
models.py                         สร้างโมเดล M1–M8 (interface เดียวกัน)
baselines/                        HexConv / HexConvGRU (M4, M5) + metrics
train.py                          harness สำหรับ train แบบ multi-task ใช้ร่วมทุกโมเดล
eval.py, evalmetrics.py           ประเมิน test ครั้งเดียว + ข้อมูลสำหรับ error analysis
scripts/                          ติดตั้งบนเครื่อง GPU, รันชุดการทดลอง, SLURM
tests/                            เทสต์ (47 เทสต์ผ่านบน CPU)
docs/                             เอกสารทั้งหมด
```

## เริ่มต้นเร็ว
```bash
bash scripts/setup_remote.sh          # ติดตั้ง env + ดาวน์โหลด Sintel (~7.2 GB) + สร้าง null + รันเทสต์
scripts/run_grid_local.sh             # grid 39 รอบบนเครื่องเดียว (resume ได้)
DRY_RUN=1 scripts/run_grid.sh         # (A100/หลาย GPU) ดูคำสั่งของ scripts/grid.tsv
python eval.py runs/<run_name>        # ประเมิน test (ครั้งเดียว)
```
รายละเอียดใน [docs/RUNNING.md](docs/RUNNING.md)

## อ้างอิงและ license
- **MPI Sintel:** Butler et al., ECCV 2012 ใช้เพื่อการวิจัย ข้อมูลไม่ได้อยู่ใน repo นี้
- **flyvis:** Lappalainen et al., Nature 2024 (MIT license)
- **FlyWire connectome:** Dorkenwald et al., Nature 2024
- **ส่วนที่ AI ช่วยเขียน:** โค้ดส่วนหนึ่งเขียนด้วย AI coding assistant (Claude subagents, Gemini ผ่าน agy) และผ่านการตรวจ เทสต์ และรีวิวโดยกลุ่ม ประวัติทั้งหมดอยู่ใน git log และ `docs/agy/`
