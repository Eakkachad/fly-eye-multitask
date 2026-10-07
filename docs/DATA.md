# ข้อมูล (Data)

## แหล่งข้อมูล
**MPI Sintel Flow Dataset** — Butler et al., ECCV 2012 — http://sintel.is.tue.mpg.de/
- เป็นภาพยนตร์ 3D แบบ open source ("Sintel", Blender Foundation) ที่ render ใหม่พร้อม ground truth ของ optic flow และ depth ทุกพิกเซล
- **การใช้งาน:** ใช้เพื่อการวิจัยได้ และต้องอ้างอิง Butler et al. 2012
- **ข้อมูลไม่ได้อยู่ใน repo นี้** ต้องดาวน์โหลดเองผ่าน `scripts/setup_remote.sh` ซึ่งโหลดให้อัตโนมัติ
  - `MPI-Sintel-complete.zip`: 5.6 GB (ภาพ + flow)
  - `MPI-Sintel-depth-training-20150305.zip`: 1.6 GB (depth)

## ส่วนที่ใช้ / ไม่ใช้
| ส่วนของ Sintel | ใช้? | เหตุผล |
|---|---|---|
| `training/final` (ภาพ render แบบเต็ม: motion blur, หมอก, แสง) | ✅ input | ใกล้ภาพจริงกว่า `clean` |
| `training/flow` (.flo) | ✅ target งาน flow | ground truth ทุกพิกเซล |
| `training/depth` (.dpt) | ✅ target งาน depth | ground truth ทุกพิกเซล |
| `training/clean`, `albedo`, `occlusions`, `invalid` | ❌ | ไม่จำเป็นต่อคำถามของเรา |
| `test/` (ชุดทดสอบทางการ) | ❌ | ไม่มี ground truth สาธารณะ จึงใช้แบ่งชุด test ของเราเองจาก `training` แทน |

ชุด `training` มี **23 ฉาก รวม 1,064 เฟรม** (ส่วนใหญ่ฉากละ 50 เฟรม)

## การแปลงเป็น "ภาพจากตาแมลงหวี่"
1. **Render ลงตาข่ายหกเหลี่ยม:** ใช้ `RenderedSintel` ของ flyvis แปลงภาพเป็นค่าความสว่างบนตาข่ายหกเหลี่ยมรัศมี 15 รวม **721 hexal** โดยใช้ box filter (extent 15, kernel 13) และตัดภาพเป็น 3 ช่องแนวตั้ง (`vertical_splits=3`) ทำให้ 1 เฟรมกลายเป็น 3 "มุมมอง"
2. **ตัดลำดับเวลา:** ตัดคลิปละ **19 เฟรม** (dt = 0.02 s หรือ 50 Hz) แล้ว resample ให้ตรงกับ time step ของโมเดล โดยใน train เลือกจุดเริ่มคลิปแบบสุ่ม
3. **Target flow:** render flow ลงตาข่ายเดียวกัน เป็น 2 ช่องต่อ hexal
4. **Target depth:** ค่า depth ดิบมีช่วงกว้างมาก (0.15 ถึง 2,550) จึงแปลงเป็น **log(depth)** แล้ว clip และ standardize ด้วยค่าเฉลี่ยและ SD **จากชุด train เท่านั้น** (`depth_norm.json`: mean 1.139, std 1.816 ในหน่วย log)
5. **Cache:** ข้อมูลที่ render แล้วเก็บไว้ราว 36 MB การโหลดใช้ ~0.007 s ต่อ batch จึงไม่ใช่คอขวด

## การแบ่งชุดข้อมูล (ป้องกันข้อมูลรั่ว)
Sintel มีหลายฉากที่ตัดมาจาก**ฉากต้นทางเดียวกัน** (เช่น `ambush_2`, `ambush_4` ... `ambush_7`) ถ้าแบ่งแบบสุ่มรายเฟรมหรือรายฉาก ภาพที่คล้ายกันมากจะรั่วข้ามชุด
**เราจึงแบ่งตาม "ตระกูลฉาก" (scene family)** ตระกูลเดียวกันอยู่ชุดเดียวกันเสมอ และใช้ seed คงที่ 20261005 บาลานซ์ตามจำนวนเฟรม (`splits.py`, `splits.json`)

| ชุด | ตระกูลฉาก | ฉาก | เฟรม |
|---|---|---|---|
| **train** | alley, ambush, bamboo, shaman, sleeping, temple | 15 | 674 |
| **validation** | market, mountain | 4 | 190 |
| **test** | bandage, cave | 4 | 200 |
| train 25% (การทดลองเสริม) | alley_2, ambush_6, bamboo_1, shaman_3 | 4 | 170 |

- `tests/` มีเทสต์ยืนยันว่า**ไม่มีฉากหรือตระกูลฉากซ้ำข้ามชุด**
- ชุด test ไม่ถูกใช้ในการตัดสินใจใด ๆ จนกว่าจะประเมินครั้งสุดท้าย

## Augmentation (เฉพาะ train)
ใช้การตั้งค่าเดิมของ flyvis:
- flip (p = 0.5) และ rotation บนตาข่ายหกเหลี่ยม (p = 0.5)
- contrast (std 0.2) และ brightness (std 0.1)
- Gaussian noise (0.08)
- random temporal crop

**Validation และ test:** ไม่มี augmentation และใช้ทุกเฟรมของทุกฉาก

## ข้อควรระวัง
- โมเดล flyvis ที่ pretrain มาแล้ว (ใช้เป็นแค่ค่าอ้างอิง M0) **เคยเห็น** `bandage_2`, `cave_2`, `market_5`, `market_6` ตอน train ถ้าจะรายงาน M0 ต้องใช้เฉพาะฉากที่มันไม่เคยเห็น
- **ข้อมูลน้อย:** ชุด train มีแค่ 674 เฟรม ข้อนี้เป็นทั้งข้อจำกัด และเป็นเหตุผลที่คำถามเรื่อง "prior แทนข้อมูล" น่าสนใจ
