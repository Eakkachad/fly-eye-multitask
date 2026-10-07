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

**การทดลองเสริม (data efficiency):** train M1, M2 และ M5 ด้วยข้อมูลเพียง 25% (4 ฉาก) เพื่อดูว่าโมเดลไหนทนต่อข้อมูลน้อยได้ดีกว่า

ทุกการตั้งค่ารันซ้ำ **3 seed** รวม 24 รอบการ train

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
- **ค่าอ้างอิง:** "ทำนายศูนย์" (zero-flow) และ "ทำนาย depth เฉลี่ย" เพื่อบอกว่าโมเดลเรียนรู้จริงหรือไม่
- **วิเคราะห์ข้อผิดพลาด:**
  - แยก error ตามความเร็วการเคลื่อนไหวและตามความซับซ้อนของ texture
  - แสดงภาพตัวอย่างที่โมเดลทำนายดีที่สุดและแย่ที่สุด เทียบกับ ground truth

→ รายละเอียด: [EVALUATION.md](EVALUATION.md)

## 6. สมมติฐาน (ลงทะเบียนไว้ล่วงหน้าก่อน train)
- **H1:** M1 (connectome จริง) ดีกว่า M2 (สลับสาย) ในทั้งสองงาน ทุก seed ที่จับคู่กัน
- **H2:** M1 ดีกว่า M4 (ConvGRU ขนาดเท่ากัน)
- **H3:** เมื่อข้อมูลเหลือ 25% ช่องว่างระหว่าง M1 กับโมเดลอื่นจะกว้างขึ้น (โครงสร้างช่วยชดเชยข้อมูลที่ขาด)

ไม่ว่าผลจะยืนยันหรือหักล้างสมมติฐาน เราจะรายงานตามจริง

## 7. ความเสี่ยงและแผนสำรอง (ตามผลทดลองเบื้องต้น)
- **ผล smoke test:** ที่ 2,000 iteration ทุกโมเดลยังทำนาย flow ได้ใกล้ระดับ "ทำนายศูนย์" (val EPE 5.03–5.08 เทียบกับ 5.07)
  - แม้แต่ flyvis ที่ train เต็ม 250k iteration ในเปเปอร์ก็ชนะ zero-flow บนฉากที่ไม่เคยเห็นเพียง ~4%
  - flow บน Sintel จึงเป็นงานที่ยาก
- **แผน:** รันทดสอบที่ 10k iteration บน GPU A100 ก่อน
  - ถ้าโมเดลยังไม่หลุดจากระดับ zero-flow จะเปลี่ยนงาน flow ไปใช้**วิดีโอสังเคราะห์** (ขอบภาพและจุดเคลื่อนที่ที่รู้ flow แน่นอน ซึ่ง flyvis มีให้) แต่ยังคงงาน depth บน Sintel ไว้
  - ยังคงเป็น multi-task video เหมือนเดิม การเปลี่ยนแปลงนี้จะบันทึกไว้อย่างเปิดเผย
- **ทรัพยากร:** RTX 4060 8 GB สำหรับพัฒนา และ A100 80 GB สำหรับ train ชุดเต็ม

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
