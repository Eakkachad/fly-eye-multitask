# สถานะโปรเจกต์ (อัปเดต 2026-10-10)

**สรุป:** การทดลองครบทั้งหมดแล้ว (phase 1: 39 runs, phase 2: 15 runs รวม 54 runs). ชุด test ถูกประเมิน **ครั้งเดียวทั้งสองชุด**: Sintel test (phase 1, A1–A8) และ Spring (phase 2, A9). ทั้งสองชุดถือว่าใช้หมดแล้ว ห้ามประเมินซ้ำ.

## เสร็จแล้ว
| งาน | ไฟล์ |
|---|---|
| ผล phase 1 + H1–H7 | [RESULTS.md](RESULTS.md), `results/summary.csv` |
| ผล phase 2 (Spring) + P1–P4 | [RESULTS_P2.md](RESULTS_P2.md), `results/summary_p2.csv` |
| Error analysis Sintel / Spring | [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md), [ERROR_ANALYSIS_SPRING.md](ERROR_ANALYSIS_SPRING.md) |
| Engineering benchmark | [R2_BENCHMARK.md](R2_BENCHMARK.md) |
| เรื่องเล่า/สีกลาง | [NARRATIVE.md](NARRATIVE.md) |
| **รายงานผลรวมฉบับสุดท้าย + รูปสรุป** | [FINAL_REPORT.md](FINAL_REPORT.md), `docs/figs/final_*.png` (`make_summary_figs.py`) |

## ผลหลัก (ดูรายละเอียดใน FINAL_REPORT)
- Sintel test: มีแค่ M5 ชนะ Lucas–Kanade; wiring จริง (M1) ไม่ดีกว่า null; H1, H2, H4–H7 ไม่ผ่าน, H3 ผ่านบางส่วน.
- Spring: ไม่มีโมเดลชนะ zero-flow/LK ด้าน flow; Ours-S ชนะ M4 3/3 seed บน flow แต่ P1–P3 ไม่ผ่าน; M1 มี seed ระเบิด (EPE 254).

## ที่เหลือ
- ทำ slides และซ้อมพรีเซนต์ (ใช้ FINAL_REPORT, NARRATIVE และ `docs/PRESENTER_NOTES.md`).
- ไม่มีการรันโมเดลหรือประเมิน test เพิ่ม.

## ข้อจำกัดที่ต้องรายงาน
test Sintel มี 4 ฉาก, ข้อมูลเทรนน้อย, 3 seed, ไม่ปรับ lr ต่อ arm, error analysis เป็น L3 (ดู FINAL_REPORT ข้อ 10).
