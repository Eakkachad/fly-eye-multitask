# การประเมินผล (Evaluation)

## ตัวชี้วัด (รายงานแยกแต่ละงาน)
| งาน | ตัวชี้วัด | นิยาม | ยิ่งต่ำยิ่งดี |
|---|---|---|---|
| Optic flow | **EPE** (End-Point Error) | ค่าเฉลี่ยของ ‖(u,v)_pred − (u,v)_gt‖₂ ทุก hexal ทุกเฟรม | ✅ |
| Optic flow | **Angular error** (องศา) | มุมระหว่างเวกเตอร์ (u, v, 1) ที่ทำนายกับค่าจริง | ✅ |
| Depth | **RMSE** | √mean((d_pred − d_gt)²) ในหน่วย log-depth ที่ standardize แล้ว | ✅ |
| Depth | **AbsRel** | mean(\|d_pred − d_gt\| / d_gt) คำนวณบน **depth จริง** (แปลงกลับจาก log แล้ว clip ในช่วง 0.19–2,522 หน่วยของ Sintel) | ✅ |

- **ทำไมเลือกชุดนี้:** EPE เป็นมาตรฐานของ optic flow (Sintel benchmark ใช้ตัวนี้) ส่วน RMSE กับ AbsRel เป็นมาตรฐานของงาน depth (ตรงกับตัวอย่าง "Pixel Regression → MAE/RMSE" ในเกณฑ์)
- **หน่วย:**
  - flow: หน่วยเดียวกับ flyvis (ระยะบนตาข่ายต่อเฟรม หลังการ render)
  - depth: RMSE ใช้หน่วย log-depth ที่ normalize ด้วยสถิติของ train ส่วน AbsRel คำนวณบน depth จริง จึงไม่ขึ้นกับการ normalize

## Floors (เส้นฐานที่ไม่ต้อง train, PLAN A6.1)
ใช้ตรวจว่าโมเดลเรียนรู้จริง ลงทะเบียนไว้ก่อน grid และรายงานบน test คู่กับโมเดล (`floors.py --split test --once`) ค่าบน val:
| งาน | floor | ค่า (val) |
|---|---|---|
| flow EPE | zero-flow | 5.074 |
| flow EPE | Lucas–Kanade บนตาข่ายหกเหลี่ยม (ไม่เรียนรู้; fit แผนที่หน่วย 2×2 บน train, จูน window/lambda บน val) | 4.709 |
| flow EPE | train-mean flow | 5.288 |
| flow EPE | oracle constant-velocity (ใช้ GT flow เฟรมก่อนหน้า; **oracle ไม่ใช่คู่แข่ง**) | 1.316 |
| depth RMSE | train-mean | 1.863 |
| depth RMSE | per-hexal train-mean | 1.856 |

- โมเดลจะรายงานเป็น **margin เหนือ floor ที่ดีที่สุดที่ไม่ใช่ oracle** (flow: LK, depth: per-hexal mean)
- ใน probe (val, seed เดียว) LK ชนะ M1 (4.915) แต่ M4 (4.501) ชนะ LK ราว 4% ถ้าโมเดลไม่ชนะ floor ก็รายงานตามจริง

## Claim ladder (PLAN A6.1)
ทุกข้อสรุปในรายงานติดป้ายระดับความน่าเชื่อถือ:
| ระดับ | ความหมาย |
|---|---|
| **L1** | ลงทะเบียนไว้ล่วงหน้า + ยืนยันบน test set |
| **L2** | ลงทะเบียนไว้ล่วงหน้า แต่ val เท่านั้นหรือผลไม่ครบ |
| **L3** | เชิงสำรวจ (exploratory) |

Diagnostics ที่เป็น L3 (ไม่ train ใหม่): CKA ระหว่างโมเดล (พร้อมแถบ seed-vs-seed), front-end necessity, เสถียรภาพของ recurrent state, ablation-superposition (ปิดกลุ่มเซลล์/channel แล้วดูว่าบวกกันได้ไหม), divergence/curl ของ flow เทียบ ground truth

## Protocol (ป้องกันการ "แอบดู" ชุด test)
1. **เลือก hyperparameter บน validation เท่านั้น** เช่น lr และ checkpoint ที่ดีที่สุด (`best.pt` คือจุดที่ val loss ต่ำสุด)
2. **ชุด test ประเมินครั้งเดียว** ด้วย `eval.py` ซึ่งไม่ยอมรันซ้ำบน test ถ้าไม่ใส่ `--force` และการใช้ `--force` ต้องบันทึกเหตุผล
3. **รายงานค่าเฉลี่ย ± SD จาก 3 seed** และเปรียบเทียบแบบจับคู่ seed (paired)
4. **สมมติฐาน H1–H7 ลงทะเบียนไว้ก่อน train** (H4–H7 เพิ่มใน amendment A5/A6 ก่อน grid) (`PLAN.md`) ไม่ว่าผลจะออกมาอย่างไรก็รายงานทั้งหมด

## การวิเคราะห์ข้อผิดพลาด (Error Analysis)
`eval.py` บันทึก error ราย hexal (`per_pixel.npz`) เพื่อวิเคราะห์ต่อ:
- **แยกตามความเร็วการเคลื่อนไหว:** โมเดลพลาดตอนวัตถุเคลื่อนเร็วหรือช้า
- **แยกตาม texture:** วัดจากความแปรปรวนของความสว่างเฉพาะที่ เพราะพื้นผิวเรียบไม่มีข้อมูลให้คำนวณ flow (aperture problem)
- **แยกรายฉาก:** ฉากไหนยาก เช่น ฉากมืด ฉากมีหมอก หรือฉากเคลื่อนที่เร็ว
- **ภาพตัวอย่าง** (`examples.npz`): ลำดับที่ทำนายดีที่สุดและแย่ที่สุด แสดงภาพ input, flow ที่ทำนายเทียบกับ ground truth และ depth ที่ทำนายเทียบกับ ground truth
- **กราฟ data efficiency:** error ที่ข้อมูล 25% เทียบกับ 100% ของ M1, M2 และ M5

## ข้อจำกัดที่จะรายงาน
- ข้อมูลน้อย (674 เฟรมใน train) และชุด test มีแค่ 2 ตระกูลฉาก
- งบ train จำกัด เปเปอร์ต้นฉบับใช้ 250k iteration แต่เราใช้น้อยกว่ามาก
- render บนตาข่าย 721 จุด ความละเอียดต่ำกว่าภาพ Sintel ต้นฉบับมาก จึงเทียบกับ leaderboard ของ Sintel โดยตรงไม่ได้
- depth วัดในหน่วย log-standardized
