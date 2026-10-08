# Project Proposal (ร่าง): Fly-Eye Multi-Task Video
**"ใช้ wiring ของสมองแมลงหวี่เป็น prior ให้ deep learning: ทำนาย optic flow และ depth จากวิดีโอพร้อมกัน"**

> วิชา Project 2: Deep Learning for Image Analysis · กลุ่ม 8 คน · ประเภทงาน: **Multi-task Learning + Video Analysis (ความยาก 4 คะแนน)**
> เอกสารนี้เป็นร่างสำหรับส่งอาจารย์และให้สมาชิกในกลุ่มอ่าน รายละเอียดเชิงลึกอยู่ในเอกสารอื่นในโฟลเดอร์ `docs/` (ลิงก์อยู่ท้ายแต่ละหัวข้อ)

---

## 1. ที่มาและแรงจูงใจ
- โมเดล deep learning ทั่วไปเรียนรู้โครงสร้างทั้งหมดจากข้อมูล จึงต้องใช้ข้อมูลและพารามิเตอร์จำนวนมาก
- สมองสัตว์ได้ "โครงสร้าง" มาจากวิวัฒนาการโดยไม่ต้องเรียนรู้ใหม่ ปี 2024 มีการเผยแพร่แผนที่การเชื่อมต่อของสมองแมลงหวี่ทั้งตัว (**connectome**: FlyWire, Nature 2024)
- Lappalainen et al. (Nature 2024) นำ connectome ของระบบการมองเห็นมาสร้างเป็น neural network ชื่อ **flyvis** แล้ว train ให้ทำนายการเคลื่อนไหว (optic flow) จากวิดีโอ
- กลุ่มเรามีงานวิจัยต่อเนื่องที่ศึกษาว่า "โครงสร้างจาก connectome มีค่าแค่ไหน" (ดู [BACKGROUND.md](BACKGROUND.md))

**คำถามของโปรเจกต์นี้:** ถ้าให้โมเดลที่มี wiring จาก connectome จริง แข่งกับ
- โมเดลเดียวกันที่สลับสายแบบสุ่ม (null model)
- deep learning มาตรฐาน (ConvGRU) ที่มีพารามิเตอร์เท่ากันหรือมากกว่า ~40 เท่า

บนงานวิดีโอแบบ multi-task เดียวกัน **wiring จริงช่วยให้เรียนได้ดีขึ้นหรือใช้ข้อมูลน้อยลงหรือไม่**

## 2. นิยามปัญหา (Problem Definition)
| | รายละเอียด |
|---|---|
| **Input** | วิดีโอขาวดำ 19 เฟรม (50 Hz) ที่แปลงให้อยู่ในรูป "ตาแมลงหวี่" คือตาข่ายหกเหลี่ยม 721 จุด (hexal) |
| **Output งานที่ 1** | **Optic flow** เวกเตอร์ 2 มิติ (u, v) ต่อ hexal ต่อเฟรม → pixel regression |
| **Output งานที่ 2** | **Depth** ค่าความลึก 1 ค่า ต่อ hexal ต่อเฟรม → pixel regression |
| **ทำไมเป็น Video Analysis** | flow นิยามได้จากความต่างระหว่างเฟรมเท่านั้น โมเดลทุกตัวเป็นแบบ recurrent ที่สะสมข้อมูลข้ามเวลา ไม่ได้ทำนายทีละเฟรมแยกกัน |
| **ทำไมเป็น Multi-task** | โมเดลเดียวมี backbone ร่วมกัน และมี 2 head (flow และ depth) ที่ train พร้อมกันด้วย loss รวม แล้วรายงานผลแยกแต่ละงาน |

## 3. ข้อมูล (Dataset)
- **MPI Sintel** (Butler et al., ECCV 2012) เป็นภาพยนตร์ CG แบบ open-source ที่มี ground truth ของ flow และ depth ทุกพิกเซล
- **ส่วนที่ใช้:** ชุด `training` 23 ฉาก ได้แก่ render แบบ `final` (มี motion blur และแสงจริง), `flow` และ `depth` รวม 1,064 เฟรม
- **ส่วนที่ไม่ใช้:** ชุด `test` ทางการ (ไม่มี ground truth สาธารณะ) และ render แบบ `clean`
- **การแปลงข้อมูล:** render ภาพลงตาข่ายหกเหลี่ยมด้วยโค้ดของ flyvis แล้ว normalize depth ด้วย log และค่าสถิติจาก train เท่านั้น
- **การแบ่งข้อมูล (สำคัญ):**
  - **แบ่งตาม "ตระกูลฉาก"** ฉากที่มาจากฉากต้นทางเดียวกัน (เช่น ambush_2 กับ ambush_4) อยู่ชุดเดียวกันเสมอ จึงไม่มีเฟรมคล้ายกันรั่วข้ามชุด
  - train 15 ฉาก (674 เฟรม), validation 4 ฉาก (190 เฟรม), test 4 ฉาก (200 เฟรม)
  - มีเทสต์อัตโนมัติยืนยันว่าไม่มีฉากซ้ำข้ามชุด
- **Augmentation:** ใช้กับ train เท่านั้น ได้แก่ flip, rotate, contrast, brightness และ noise

→ รายละเอียด: [DATA.md](DATA.md)

## 4. แบบจำลองที่เปรียบเทียบ (≥ 3 รูปแบบ)
ทุกโมเดลใช้ input เดียวกัน, head แบบเดียวกัน, loss เดียวกัน, optimizer และจำนวน iteration เท่ากัน

| ID | โมเดล | พารามิเตอร์ที่ train | บทบาท |
|---|---|---|---|
| **M1** | flyvis: wiring จาก **connectome จริง** (65 ชนิดเซลล์, 604 คู่การเชื่อมต่อ) train ใหม่ตั้งแต่ต้น | 15,387 | สมมติฐานหลัก |
| **M2** | flyvis ที่ **สลับสายแบบ degree-preserving** เก็บจำนวนเส้นเข้า/ออกของแต่ละชนิดเซลล์และเครื่องหมาย (กระตุ้น/ยับยั้ง) ไว้ แต่สุ่มว่าใครต่อกับใคร | 15,387 | null เชิงโครงสร้าง |
| **M3** | flyvis ที่ **สุ่มการเชื่อมต่อทั้งหมด** (จำนวนเส้นและสัดส่วน E/I เท่าเดิม) | 15,387 | null แบบอ่อน |
| **M4** | **ConvGRU บนตาข่ายหกเหลี่ยม** ขนาดเล็ก พารามิเตอร์เท่ากับ M1 | 15,402 | deep learning มาตรฐาน (ขนาดเท่ากัน) |
| **M5** | **ConvGRU ขนาดใหญ่** | 604,835 | deep learning มาตรฐาน (ใหญ่กว่า ~40 เท่า) |
| **M6** | **hybrid:** front-end flyvis (connectome จริง) ส่ง activity ของเซลล์ output ทั้ง 721 hexal เข้า trunk แบบ HexConvGRU → head flow/depth | 15,423 | ทดสอบ "connectome + deep learning" |
| **M7** | M6 แต่ front-end ใช้ connectome ที่สลับสายแบบ degree-preserving | 15,423 | control: แยก "wiring จริง" ออกจาก "front-end recurrent เพิ่ม" |
| **M6f** | front-end = M1 ที่ train แล้ว (**frozen**) + trunk แบบ residual ที่ layer สุดท้ายเริ่มที่ศูนย์ (เริ่ม train ที่คำตอบของ M1 พอดี) | 14,689 (trainable) | hybrid แบบ residual |
| **M7f** | เหมือน M6f แต่ใช้ M2 (สลับสาย) เป็น front-end | 14,689 (trainable) | control ของ M6f |
| **M8** | **HexConvGRU-K:** สถาปัตยกรรมแบบ M4 แต่วน GRU cell K รอบต่อเฟรม (weight ร่วม), train ด้วย K ~ U{1..4}, ประเมินที่ K = 1..4 (any-time) | 15,402 | ทดสอบ "คิดนานขึ้น" โดยไม่เพิ่มพารามิเตอร์ |

**การทดลองเสริม (data efficiency):** train M1, M2 และ M5 ด้วยข้อมูลเพียง 25% (4 ฉาก) เพื่อดูว่าโมเดลไหนทนต่อข้อมูลน้อยได้ดีกว่า

ทุกการตั้งค่ารันซ้ำ **3 seed** รวม **39 รอบ**การ train (M1–M5 รวมชุดข้อมูล 25% = 24 รอบ, M6/M7 = 6, M6f/M7f = 6, M8 = 3)

M6–M8 เพิ่มหลังเห็นผล probe (ดู §7) แต่ก่อนรัน grid ไอเดียเชิงวิธีวิจัย (floor, claim ladder, residual hybrid, any-time K, diagnostics) นำมาจากบันทึกวิจัย katgpt-rs ของกลุ่มเราโดยสรุป

→ รายละเอียด: [MODELS.md](MODELS.md)

## 5. การประเมินผล (Evaluation)
| งาน | ตัวชี้วัด |
|---|---|
| Optic flow | **EPE** (End-Point Error: ระยะห่างระหว่างเวกเตอร์ที่ทำนายกับค่าจริง) และ **Angular error** (องศา) |
| Depth | **RMSE** (ในหน่วย log-depth ที่ normalize แล้ว) และ **AbsRel** (ค่าคลาดเคลื่อนสัมพัทธ์) |

- **Protocol:**
  - เลือก hyperparameter บน validation เท่านั้น
  - ชุด test **ประเมินครั้งเดียว**ตอนจบ โค้ดจะไม่ยอมรันซ้ำถ้าไม่ใส่ `--force`
  - รายงานค่าเฉลี่ย ± ส่วนเบี่ยงเบนมาตรฐานจาก 3 seed
- **ค่าอ้างอิง (floors)** ลงทะเบียนไว้ล่วงหน้าและรายงานบน test คู่กับโมเดล:
  - flow: zero-flow, train-mean flow, **Lucas–Kanade บนตาข่ายหกเหลี่ยม** (ไม่เรียนรู้ มีแผนที่เชิงเส้น 4 พารามิเตอร์ที่ fit บน train เท่านั้น) และ **oracle** constant-velocity (ใช้ GT flow เฟรมก่อนหน้า ระบุเป็น oracle ไม่ใช่คู่แข่ง)
  - depth: ค่าเฉลี่ยจาก train, ค่าเฉลี่ยรายตำแหน่ง hexal
  - โมเดลรายงานเป็น **margin เหนือ floor ที่ดีที่สุดที่ไม่ใช่ oracle**
- **Claim ladder:** ทุกข้อสรุปติดป้าย **L1** = ลงทะเบียนล่วงหน้า + ประเมินบน test, **L2** = ลงทะเบียนล่วงหน้าแต่ใช้ val หรือผลบางส่วน, **L3** = เชิงสำรวจ
- **การวิเคราะห์เชิงสำรวจ (L3, ไม่ train ใหม่):** CKA ระหว่างโมเดล (เทียบกับช่วงความต่างระหว่าง seed), front-end necessity, เสถียรภาพของ recurrent state, ablation-superposition (ปิดกลุ่มเซลล์/channel แล้วดูว่าผลรวมกันได้หรือไม่), divergence/curl ของ flow บนตาข่ายหกเหลี่ยม
- **วิเคราะห์ข้อผิดพลาด:**
  - แยก error ตามความเร็วการเคลื่อนไหวและตามความซับซ้อนของ texture
  - แสดงภาพตัวอย่างที่โมเดลทำนายดีที่สุดและแย่ที่สุด เทียบกับ ground truth

→ รายละเอียด: [EVALUATION.md](EVALUATION.md)

## 6. สมมติฐาน (ลงทะเบียนไว้ล่วงหน้าก่อน train)
- **H1:** M1 (connectome จริง) ดีกว่า M2 (สลับสาย) ในทั้งสองงาน ทุก seed ที่จับคู่กัน
- **H2:** M1 ดีกว่า M4 (ConvGRU ขนาดเท่ากัน)
- **H3:** เมื่อข้อมูลเหลือ 25% ช่องว่างระหว่าง M1 กับโมเดลอื่นจะกว้างขึ้น (โครงสร้างช่วยชดเชยข้อมูลที่ขาด)

เพิ่มก่อนรัน grid (PLAN A5/A6) หลังเห็นผล probe:
- **H4:** M6 ดีกว่า M4 ทั้ง flow EPE และ depth RMSE (ค่าเฉลี่ย 3 seed และ ≥ 2/3 seed ที่จับคู่)
- **H5:** M6 ดีกว่า M7 ใน flow EPE ครบ 3/3 seed ที่จับคู่ (ประโยชน์เฉพาะของ connectome) ถ้า H4 จริงแต่ H5 ไม่จริง จะรายงานว่าประโยชน์มาจากสถาปัตยกรรม ไม่ใช่ wiring
- **H6:** M8 (K=4) ดีกว่า M4 ใน test flow EPE อย่างน้อย 2/3 seed ที่จับคู่
- **H7:** M6f ดีกว่า M7f ใน test flow EPE ครบ 3/3 seed ที่จับคู่

ไม่ว่าผลจะยืนยันหรือหักล้างสมมติฐาน เราจะรายงานตามจริง

## 7. ความเสี่ยงและแผนสำรอง (ตามผลทดลองเบื้องต้น)
- **ผล learnability probe (PLAN A4, ลงทะเบียนก่อนรัน): PASS** ที่ 10k iteration, seed เดียว, val เท่านั้น: val EPE ดีที่สุดของ M1 = **4.915**, M4 = **4.501** เทียบกับ zero-flow 5.074 จึงคง flow บน Sintel ไว้ (ไม่ต้องใช้แผนสำรองวิดีโอสังเคราะห์)
- **คำเตือนตามจริง:**
  - Lucas–Kanade floor (val 4.709) ดีกว่าค่า probe ของ M1
  - ใน probe M4 ดีกว่า M1 จึง **H2 มีความเสี่ยง** ที่จะไม่เป็นจริง
  - probe เป็น seed เดียวและ val มี noise จึงไม่ใช่การทดสอบ H1–H3
  - โมเดล hybrid (M6–M8) เพิ่มหลังเห็น probe (เห็นเฉพาะ val) แต่ก่อนรัน grid และเปิดเผยไว้ใน PLAN
  - ไม่ได้จูน learning rate รายโมเดล (ใช้ 5e-4 เท่ากันทุกโมเดล) เป็นข้อจำกัด
- **ทรัพยากร:** ทุกอย่าง train บน **RTX 4060 8 GB** ในเครื่องเอง เพราะวัดการเร่งความเร็วได้จริง (kernel Triton `fastfly` เร่งโมเดล flyvis 2.4×, `torch.compile` เร่ง M4 5×) grid 39 รอบ (30k iteration) ใช้เวลาประมาณ 31–35 ชั่วโมง

→ สถานะล่าสุด: [STATUS.md](STATUS.md)

## 8. การแบ่งงานและกำหนดการ
สมาชิก 8 คน แต่ละคนรับผิดชอบ 1 ส่วนและต้องอธิบายส่วนของตนเองได้ตอนนำเสนอ → [TEAM.md](TEAM.md)

## 9. สิ่งที่จะส่ง
1. **โค้ด:** repository นี้ พร้อมวิธีใช้ แหล่งข้อมูล และการตั้งค่า ([RUNNING.md](RUNNING.md))
2. **สไลด์นำเสนอ:** ครอบคลุมเกณฑ์การให้คะแนน

## 10. เอกสารอ้างอิง
- Butler, D. J., Wulff, J., Stanley, G. B., & Black, M. J. (2012). A naturalistic open source movie for optical flow evaluation. *ECCV*.
- Lappalainen, J. K. et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. *Nature* 634.
- Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634.
- Maslov, S., & Sneppen, K. (2002). Specificity and stability in topology of protein networks. *Science* 296. (วิธีสลับสายแบบ degree-preserving)
- Shi, X. et al. (2015). Convolutional LSTM network. *NeurIPS*. / Ballas, N. et al. (2016). Delving deeper into convolutional networks for learning video representations (ConvGRU). *ICLR*.
