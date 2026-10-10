# Shared narrative + style guide for final deliverables (all agents follow this)

## One-sentence story
เราทำงาน multi-task video (ทาย optic flow + depth ทุกจุดบนตาแมลงหวี่ 721 hexal) เปรียบเทียบอย่างแฟร์ระหว่าง
วิธีคลาสสิก (floors), deep learning มาตรฐาน, **สมองแมลงหวี่ (connectome) + null model** และ **วิธีของเรา**
แล้วทดสอบบน test 2 ชุดที่ใช้ครั้งเดียว (Sintel ฉากที่ไม่เคยเห็น + Spring ต่างโดเมน)

## Key messages (only claim what the numbers say; label L1/L2/L3)
1. งานยากจริง: บน Sintel test มีแค่ M5 (ConvGRU ใหญ่) ที่ชนะ Lucas–Kanade; บน Spring ไม่มีโมเดลไหนชนะ zero-flow/LK ด้าน flow (L1).
2. สมองแมลงหวี่: wiring จริง (M1) ไม่ดีกว่า wiring สลับ/สุ่ม (M2/M3) — H1, H5, H7 ไม่ผ่าน (L1). ข้ามโดเมน (Spring) null M2/M3 กลับดีที่สุดในโมเดลที่เรียนรู้; M1 seed 1 ระเบิด (EPE 254, ยืนยันว่าไม่ใช่บั๊ก kernel).
3. วิธีของเรา (recurrent depth + EPE loss + scale-invariant depth loss + init-normalised task weights): Ours-S ดีที่สุดในกลุ่ม deep learning ขนาดเท่ากันบน Spring flow (0.590 vs M4 0.649, 3/3 seeds) แต่ไม่ชนะ floors; P1–P3 ไม่ผ่านตามเกณฑ์ (L1). Ablation: EPE loss และการวนหลายรอบช่วย (2/3 seeds), scale-invariant loss ไม่ช่วย.
4. Recurrent depth: ภายในโมเดลเดียวกัน K มากขึ้นดีขึ้นเสมอ (Sintel test M8 K1 5.77 → K4 5.58) แต่ overfit เมื่อ train นาน (val ดีสุด ~6k iter).
5. วิธีวิจัยที่ทำให้เชื่อผลได้: pre-registration (A1–A9), floors, null controls, test ใช้ครั้งเดียว ×2 ชุด, val ไม่สะท้อน test (r ภายในโมเดล 0.09) — จับข้อสรุปผิดได้อย่างน้อย 3 ครั้ง.
6. Engineering: fused Triton kernel (flyvis 2.4×), scatter-free HexConv backward + cell-level compile (M4 3.1×, M8 2.1×) → 54 training runs บน RTX 4060 8 GB.
Never name internal private projects; cite published literature instead.

## Colour / naming map (use everywhere, always also say it in text)
- connectome (real wiring): M1, M6, M6f — amber #E3A33B
- null / rewired: M2, M3, M7, M7f — coral #E07A5F
- deep learning baselines: M4, M5, M8 — teal #3BA7A0
- our method (Ours-S, Ours-L, ablations noSI/noEPE/K1): violet #6A5ACD (ablations: same hue, lighter/hatched)
- floors (zero-flow, LK, constant depth): grey #8A8F98; oracle: dashed grey, labelled ORACLE
Figures: 150 dpi PNG, titles + axis labels with units, legend names like "M2 (null)", lower-is-better noted, claim level (L1/L2/L3) in title.

## Sources of numbers
docs/RESULTS.md (Sintel test, phase 1), docs/RESULTS_P2.md (Spring, phase 2), docs/ERROR_ANALYSIS.md, docs/R2_BENCHMARK.md,
results/summary.csv, results/summary_p2.csv, runs/<name>/{test,spring_last,spring_best,...}/metrics.json, per_pixel.npz.
