#!/usr/bin/env python
"""Error analysis of the phase-2 Spring TEST results from ALREADY-SAVED outputs only (claim level L3).

Reads runs/*/spring_last*/{metrics.json,per_pixel.npz}, runs/*/spring_best/metrics.json,
floors_out/spring/metrics.json and the GT cache data/spring_hex (via spring_data.SpringHex, CPU only,
no model is built or run).  Writes docs/figs/sp_*.png, results/error_analysis_spring/*.csv|json and
docs/ERROR_ANALYSIS_SPRING.md.  NOTE: per_pixel.npz stores only per-hexal error maps (epe, angular,
depth_aligned_err, ...) and NOT the predictions, so the predicted speed is *estimated* from
(|GT|, EPE, angular error) -- see est_speed() and its synthetic validation (R['synth']).
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # never touch the GPU
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import hsv_to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
RUNS, FIGS, TABS = ROOT / "runs", ROOT / "docs" / "figs", ROOT / "results" / "error_analysis_spring"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2)
AMBER, TEAL, CORAL, GREY, VIOLET = "#E3A33B", "#3BA7A0", "#E07A5F", "#8A8F98", "#6A5ACD"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})

# arm -> (run-name pattern, colour, hatch, legend label)
ARMS = {
    "M1": ("m1_s{}_f1.0", AMBER, "", "M1 (connectome)"),
    "M6": ("m6_s{}_f1.0", AMBER, "", "M6 (connectome hybrid)"),
    "M6f": ("m6f_s{}_f1.0", AMBER, "", "M6f (connectome frozen)"),
    "M2": ("m2_s{}_f1.0", CORAL, "", "M2 (null)"),
    "M3": ("m3_s{}_f1.0", CORAL, "", "M3 (null)"),
    "M7": ("m7_s{}_f1.0", CORAL, "", "M7 (null hybrid)"),
    "M7f": ("m7f_s{}_f1.0", CORAL, "", "M7f (null frozen)"),
    "M4": ("m4_s{}_f1.0", TEAL, "", "M4 (deep learning)"),
    "M5": ("m5_s{}_f1.0", TEAL, "", "M5 (deep learning)"),
    "M8": ("m8_s{}_f1.0", TEAL, "", "M8 (deep learning)"),
    "Ours-S": ("p2_oursS_s{}", VIOLET, "", "Ours-S (ours)"),
    "Ours-L": ("p2_oursL_s{}", "#483D8B", "", "Ours-L (ours)"),
    "noEPE": ("p2_noEPE_s{}", "#9A8FE0", "//", "Ours-S noEPE (ablation)"),
    "K1": ("p2_K1_s{}", "#B3AAEA", "xx", "Ours-S K1 (ablation)"),
    "noSI": ("p2_noSI_s{}", "#CCC6F2", "..", "Ours-S noSI (ablation)"),
}
RUNOF = {(a, s): p[0].format(s) for a, p in ARMS.items() for s in SEEDS}
R: dict = {}


def save_csv(name, header, rows):
    with open(TABS / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def savefig(fig, name):
    fig.savefig(FIGS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def jl(p):
    return json.loads(Path(p).read_text())


# ---------------------------------------------------------------- speed estimate
def est_speed(G, e, th_deg, n=100):
    """|p| from |g|=G, EPE=e=|p-g| and Barron angular error th between (p,1),(g,1).
    |p| lies in [|e-G|, e+G]; cos th = (X+1)/(sqrt(|p|^2+1) sqrt(G^2+1)) with
    X = p.g = (|p|^2+G^2-e^2)/2.  Grid search over |p| (n points) -> nearest cos."""
    G, e, th = (np.asarray(x, np.float64) for x in (G, e, th_deg))
    lo, hi = np.abs(e - G), e + G
    t = np.linspace(0, 1, n)[None]
    s = lo[:, None] + (hi - lo)[:, None] * t
    X = (s ** 2 + G[:, None] ** 2 - e[:, None] ** 2) / 2
    c = (X + 1) / np.sqrt(s ** 2 + 1) / np.sqrt(G ** 2 + 1)[:, None]
    i = np.argmin(np.abs(c - np.cos(np.radians(th))[:, None]), 1)
    return s[np.arange(len(G)), i]


SUB = 4  # estimated speed uses every SUB-th frame (cost); all other metrics use all frames


def _est_worker(run):
    d = np.load(RUNS / run / "spring_last" / "per_pixel.npz")
    rows = np.arange(0, d["epe"].shape[0], SUB)
    G, e, a, fv = (d[k][rows].astype(np.float64) for k in ("gt_speed", "epe", "angular", "flow_valid"))
    e = np.minimum(np.nan_to_num(e, nan=65504.0, posinf=65504.0), 65504.0)   # float16 overflow -> lower bound
    a = np.nan_to_num(a, nan=90.0, posinf=90.0)
    fv = fv.astype(bool)
    out = np.full(G.shape, np.nan, np.float32)
    idx = np.flatnonzero(fv.ravel())
    flat = out.ravel()
    for a0 in range(0, len(idx), 200000):
        j = idx[a0:a0 + 200000]
        flat[j] = est_speed(G.ravel()[j], e.ravel()[j], a.ravel()[j])
    return run, out


# ---------------------------------------------------------------- GT (cache) + floors
from spring_data import SpringHex  # noqa: E402  (CPU only; reads the cached hex tensors)

ds = SpringHex()
clip_names = list(ds.names)
seq_of_clip = np.array([n.split("_")[1] for n in clip_names])
SEQS = sorted(set(seq_of_clip))
dtf = ds.depth_transform
STD = float(dtf.std)
GT_FLOW, GT_LOGD, LUM = [], [], []
import torch  # noqa: E402

for i in range(len(ds)):
    it = ds.get_item(i)
    GT_FLOW.append(it["flow"].numpy().astype(np.float32))
    GT_LOGD.append(torch.log(dtf.clip(it["depth"]))[:, 0].numpy().astype(np.float32))
    LUM.append(it["lum"][:, 0].numpy().astype(np.float32))
lens = [x.shape[0] for x in GT_FLOW]
OFF = np.concatenate([[0], np.cumsum(lens)])
GT_FLOW, GT_LOGD, LUM = (np.concatenate(x, 0) for x in (GT_FLOW, GT_LOGD, LUM))
T = OFF[-1]
crow = np.repeat(np.arange(len(lens)), lens)           # row -> clip
rseq = np.searchsorted(SEQS, seq_of_clip[crow])        # row -> sequence index
tnorm = np.concatenate([np.linspace(0, 1, n) for n in lens])
R["n_clips"], R["n_rows"], R["seqs"] = len(lens), int(T), SEQS
R["seq_frames_50hz"] = {s: int(sum(l for l, q in zip(lens, seq_of_clip) if q == s)) for s in SEQS}
FLOORS = jl(ROOT / "floors_out/spring/metrics.json")["floors"]
EDGES = np.array(FLOORS["flow_zero"]["epe_by_speed"]["edges"])
LK_BIN = np.array(FLOORS["flow_lucas_kanade_hex"]["epe_by_speed"]["epe"])
ZERO_BIN = np.array(FLOORS["flow_zero"]["epe_by_speed"]["epe"])
R["floor"] = dict(zero=FLOORS["flow_zero"]["epe"], lk=FLOORS["flow_lucas_kanade_hex"]["epe"],
                  depth_const=FLOORS["depth_constant"]["depth_rmse_aligned"])

# ---------------------------------------------------------------- load per-pixel outputs
PX, NONFIN = {}, {}
for (arm, s), run in RUNOF.items():
    d = np.load(RUNS / run / "spring_last" / "per_pixel.npz")
    assert list(d["names"]) == clip_names and int(d["offsets"][-1]) == T
    _e = d["epe"].astype(np.float32)
    NONFIN[f"{arm}_s{s}"] = int((~np.isfinite(_e[d["flow_valid"]])).sum())
    _e = np.minimum(np.nan_to_num(_e, nan=65504.0, posinf=65504.0), 65504.0)  # float16 overflow -> lower bound
    PX[(arm, s)] = dict(epe=_e, ang=d["angular"].astype(np.float32),
                        dal=d["depth_aligned_err"].astype(np.float32), fv=d["flow_valid"], dv=d["depth_valid"])
    if (arm, s) == ("M1", 0):
        gts = d["gt_speed"].astype(np.float32)
        FV, DV = d["flow_valid"], d["depth_valid"]
_gn = np.linalg.norm(GT_FLOW, axis=1)[FV]
R["gt_speed_check_max_rel_diff_vs_cache"] = float((np.abs(gts[FV] - _gn) / np.maximum(_gn, 1e-3)).max())
R["gt_speed_check_mean_abs_diff"] = float(np.abs(gts[FV] - _gn).mean())
R["valid_masks_identical_all_runs"] = bool(all((p["fv"] == FV).all() and (p["dv"] == DV).all() for p in PX.values()))
R["nonfinite_epe_valid_float16_overflow"] = {k: v for k, v in NONFIN.items() if v}
GS = np.where(FV, gts, np.nan)
R["mean_gt_speed"] = float(np.nanmean(GS))
bidx = np.full(GS.shape, -1, np.int16)
bidx[FV] = np.clip(np.searchsorted(EDGES, gts[FV], side="right") - 1, 0, 9)
NB = np.bincount(bidx[FV], minlength=10)
GSUM = np.bincount(bidx[FV], weights=gts[FV].astype(np.float64), minlength=10)
GBIN = GSUM / NB   # mean GT speed per decile
R["bins"] = dict(edges=EDGES.tolist(), n=NB.tolist(), mean_gt=GBIN.tolist(), zero_floor_epe=ZERO_BIN.tolist(),
                 lk_epe=LK_BIN.tolist(), zero_check=float(np.abs(GBIN - ZERO_BIN).max()))

# estimated predicted speeds (parallel, CPU)
EST = {}
_cache = Path(os.environ.get("SP_CACHE", "/tmp/claude-1000/-home-user-eakject/ff095c5b-fd7e-4b32-a2e8-0f76d9503445/scratchpad/est_cache.npz"))
if _cache.exists():
    _z = np.load(_cache)
    EST = {k: _z[k] for k in _z.files}
else:
    with ProcessPoolExecutor(max_workers=12) as ex:
        for run, arr in ex.map(_est_worker, sorted(RUNOF.values())):
            EST[run] = arr
    np.savez(_cache, **EST)
SUBROWS = np.arange(0, T, SUB)

# synthetic validation of est_speed (float16 storage as in per_pixel.npz)
rng = np.random.default_rng(0)
gs = gts[FV][rng.integers(0, FV.sum(), 100000)].astype(np.float64)
ps = np.abs(rng.lognormal(-2, 1, 100000))
ga, pa = rng.uniform(0, 6.2832, 100000), rng.uniform(0, 6.2832, 100000)
gv = np.stack([gs * np.cos(ga), gs * np.sin(ga)], 1)
pv = np.stack([ps * np.cos(pa), ps * np.sin(pa)], 1)
ee = np.linalg.norm(pv - gv, axis=1)
cosv = ((pv * gv).sum(1) + 1) / np.sqrt((pv ** 2).sum(1) + 1) / np.sqrt((gv ** 2).sum(1) + 1)
thv = np.degrees(np.arccos(np.clip(cosv, -1, 1)))
sh = est_speed(gs.astype(np.float16), ee.astype(np.float16), thv.astype(np.float16))
R["synth"] = dict(mean_true=float(ps.mean()), mean_est=float(sh.mean()), ratio=float(sh.mean() / ps.mean()),
                  median_abs_err=float(np.median(np.abs(sh - ps))))
# same, but with a pred speed proportional to GT (calibrated-like) and per-decile bias
for tag, pp in (("lognormal", ps), ("proportional", gs * rng.lognormal(0, 0.5, 100000))):
    pv = np.stack([pp * np.cos(pa), pp * np.sin(pa)], 1)
    ee = np.linalg.norm(pv - gv, axis=1)
    cosv = ((pv * gv).sum(1) + 1) / np.sqrt((pv ** 2).sum(1) + 1) / np.sqrt((gv ** 2).sum(1) + 1)
    thv = np.degrees(np.arccos(np.clip(cosv, -1, 1)))
    sh2 = est_speed(gs.astype(np.float16), ee.astype(np.float16), thv.astype(np.float16))
    bb = np.clip(np.searchsorted(EDGES, gs, side="right") - 1, 0, 9)
    R["synth"][tag] = dict(ratio=float(sh2.mean() / pp.mean()),
                           ratio_by_decile=[float(sh2[bb == i].mean() / pp[bb == i].mean()) for i in range(10)])

# ---------------------------------------------------------------- helpers over runs
SEQ_LAB = {s: s for s in SEQS}


def per_run_stats(p, arm=None, s=None, run=None):
    e = np.where(FV, p["epe"], 0.0)
    rowsum, rowcnt = e.sum(1), FV.sum(1)
    out = dict(
        epe=float(rowsum.sum() / rowcnt.sum()),
        seq_epe=np.array([rowsum[rseq == i].sum() / rowcnt[rseq == i].sum() for i in range(len(SEQS))]),
        clip_epe=np.array([rowsum[crow == c].sum() / rowcnt[crow == c].sum() for c in range(len(lens))]),
        bin_epe=np.bincount(bidx[FV], weights=p["epe"][FV].astype(np.float64), minlength=10) / NB,
        rowmean=rowsum / np.maximum(rowcnt, 1))
    dsq = np.where(DV & np.isfinite(p["dal"]), p["dal"].astype(np.float64) ** 2, 0.0)
    cnt = (DV & np.isfinite(p["dal"])).sum(1)
    out["seq_dep"] = np.array([np.sqrt(dsq[rseq == i].sum() / cnt[rseq == i].sum()) for i in range(len(SEQS))])
    out["dep"] = float(np.sqrt(dsq.sum() / cnt.sum()))
    if run is not None:
        est = EST[run]
        fvs = FV[SUBROWS]
        ev = np.where(fvs, est, 0.0)
        gsub = np.where(fvs, gts[SUBROWS], 0.0)
        b = bidx[SUBROWS]
        out["est_mean"] = float(ev.sum() / fvs.sum())
        out["est_ratio"] = out["est_mean"] / float(gsub.sum() / fvs.sum())
        m = fvs & (b >= 0)
        out["bin_est"] = np.bincount(b[m], weights=est[m].astype(np.float64), minlength=10) / \
            np.maximum(np.bincount(b[m], minlength=10), 1)
        out["bin_gt_sub"] = np.bincount(b[m], weights=gts[SUBROWS][m].astype(np.float64), minlength=10) / \
            np.maximum(np.bincount(b[m], minlength=10), 1)
        out["seq_est_ratio"] = np.array([ev[rseq[SUBROWS] == i].sum() / gsub[rseq[SUBROWS] == i].sum()
                                         for i in range(len(SEQS))])
        r10 = est[m] > 10 * np.maximum(gts[SUBROWS][m], 0.1)
        out["frac_est_gt10x"] = float(r10.mean())
        out["frac_est_lt_gt"] = float((est[m] < gts[SUBROWS][m]).mean())
    out["frac_epe_gt5"] = float((p["epe"][FV] > 5.0).mean())
    _e, _g = p["epe"][FV].astype(np.float64), gts[FV].astype(np.float64)
    out["speed_lo_ratio"] = float(np.abs(_e - _g).mean() / _g.mean())   # rigorous brackets of mean|p|/mean|g|
    out["speed_hi_ratio"] = float((_e + _g).mean() / _g.mean())
    return out


ST = {k: per_run_stats(p, run=RUNOF[k]) for k, p in PX.items()}
# cross-check against metrics.json; then use the exact metrics.json clip EPEs (per_pixel float16 overflows for M1 s1)
mx, mx_dep = 0.0, 0.0
for (a, s), run in RUNOF.items():
    m = jl(RUNS / run / "spring_last" / "metrics.json")
    if (a, s) != ("M1", 1):
        mx = max(mx, abs(ST[(a, s)]["epe"] - m["epe"]) / max(m["epe"], 1e-9))
    mx_dep = max(mx_dep, abs(ST[(a, s)]["dep"] - m["depth_rmse_aligned"]))
    ce = np.array([c["epe"] for c in m["clips"]])
    nv = np.array([c["n_flow_valid"] for c in m["clips"]])
    if (a, s) == ("M1", 1):
        R["m1s1_perpixel_lower_bound_epe"] = ST[(a, s)]["epe"]
    ST[(a, s)]["clip_epe"] = ce
    ST[(a, s)]["epe"] = float(m["epe"])
    ST[(a, s)]["seq_epe"] = np.array([(ce * nv)[seq_of_clip == q].sum() / nv[seq_of_clip == q].sum() for q in SEQS])
R["check_max_rel_diff_epe_vs_metrics_json_excl_m1s1"] = mx
R["check_max_abs_diff_depth_vs_metrics_json"] = mx_dep


def amean(arm, key, seeds=SEEDS):
    return np.mean([ST[(arm, s)][key] for s in seeds], 0)


# entities: M1 split into seeds (0,2) and seed 1 because of the blow-up
ENT = {}
for a in ARMS:
    if a == "M1":
        ENT["M1 (s0,s2)"] = ("M1", (0, 2))
        ENT["M1 s1"] = ("M1", (1,))
    else:
        ENT[a] = (a, SEEDS)
ECOL = {e: ARMS[a][1] for e, (a, _) in ENT.items()}
EHATCH = {e: ARMS[a][3 - 2] for e, (a, _) in ENT.items()}
ELAB = {e: (ARMS[a][3] if e == a else e + " (connectome)") for e, (a, _) in ENT.items()}
ELAB["M1 (s0,s2)"] = "M1 seeds 0,2 (connectome)"
ELAB["M1 s1"] = "M1 seed 1 (connectome)"


def eg(e, key):
    a, ss = ENT[e]
    return amean(a, key, ss)


# ================================================================== Q1 speed
R["q1"] = {}
for arm in ARMS:
    R["q1"][arm] = dict(lo=float(amean(arm, "speed_lo_ratio")), hi=float(amean(arm, "speed_hi_ratio")), epe=float(amean(arm, "epe")), est_ratio=float(amean(arm, "est_ratio")),
                        est_mean=float(amean(arm, "est_mean")),
                        frac_est_lt_gt=float(amean(arm, "frac_est_lt_gt")))
for e in ("M1 (s0,s2)", "M1 s1"):
    R["q1"][e] = dict(lo=float(eg(e, "speed_lo_ratio")), hi=float(eg(e, "speed_hi_ratio")), epe=float(eg(e, "epe")), est_ratio=float(eg(e, "est_ratio")), est_mean=float(eg(e, "est_mean")),
                      frac_est_lt_gt=float(eg(e, "frac_est_lt_gt")))
# excess over zero-flow per decile: (EPE_bin - zero_bin) * n_bin / N_total
EXC = {}
for e in ENT:
    ex_ = (eg(e, "bin_epe") - ZERO_BIN) * NB / NB.sum()
    EXC[e] = ex_
    R["q1"][e].update(excess_total=float(ex_.sum()), excess_share_d123=float(ex_[:3].sum() / ex_.sum()) if ex_.sum() > 0 else None,
                      excess_share_d1234=float(ex_[:4].sum() / ex_.sum()) if ex_.sum() > 0 else None,
                      n_bins_beating_zero=int((eg(e, "bin_epe") < ZERO_BIN).sum()),
                      n_bins_beating_lk=int((eg(e, "bin_epe") < LK_BIN).sum()))
R["q1_bins"] = {e: dict(epe=eg(e, "bin_epe").tolist(), est_over_gt=(eg(e, "bin_est") / eg(e, "bin_gt_sub")).tolist(),
                        excess=EXC[e].tolist()) for e in ENT}
R["q1"]["_lk_excess_share_d123"] = float((((LK_BIN - ZERO_BIN) * NB / NB.sum())[:3]).sum() / (((LK_BIN - ZERO_BIN) * NB / NB.sum())).sum())
R["q1"]["_lk_total_minus_zero"] = float(((LK_BIN - ZERO_BIN) * NB / NB.sum()).sum())
R["q1"]["_zero_share_of_epe_top2_bins"] = float((ZERO_BIN * NB)[8:].sum() / (ZERO_BIN * NB).sum())
rows = []
for e in ENT:
    for b in range(10):
        rows.append([e, b + 1, f"{EDGES[b]:.4f}", f"{EDGES[b+1]:.4f}", GBIN[b], eg(e, "bin_epe")[b], ZERO_BIN[b], LK_BIN[b],
                     eg(e, "bin_est")[b], eg(e, "bin_est")[b] / eg(e, "bin_gt_sub")[b]])
save_csv("q1_by_speed_decile.csv", ["entity", "decile", "lo", "hi", "mean_gt", "epe", "zero_epe", "lk_epe", "est_pred_speed",
                                    "est_pred_over_gt"], rows)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
show = ["M1 (s0,s2)", "M2", "M3", "M4", "M8", "Ours-S"]
LS = {"M1 (s0,s2)": "-", "M2": "-", "M3": "--", "M4": "-", "M8": "--", "Ours-S": "-"}
for e in show:
    axs[0].plot(GBIN, eg(e, "bin_epe"), LS[e], color=ECOL[e], marker="o", ms=3, label=ELAB[e])
    axs[1].plot(GBIN, eg(e, "bin_est") / eg(e, "bin_gt_sub"), LS[e], color=ECOL[e], marker="o", ms=3, label=ELAB[e])
axs[0].plot(GBIN, ZERO_BIN, color=GREY, ls=":", lw=1.8, label="zero-flow floor")
axs[0].plot(GBIN, LK_BIN, color=GREY, ls="-.", lw=1.8, label="Lucas-Kanade floor")
axs[0].set(xscale="log", yscale="log", xlabel="mean GT speed in decile [lattice units / frame]",
           ylabel="EPE [lattice units / frame]  (lower is better)", title="(a) EPE by GT-speed decile (L3)")
axs[1].axhline(1, color=GREY, ls="--", lw=1.2, label="predicted = GT speed")
axs[1].set(xscale="log", yscale="log", xlabel="mean GT speed in decile [lattice units / frame]",
           ylabel="estimated predicted speed / GT speed [ratio]", title="(b) speed calibration by decile (L3, estimated)")
axs[0].legend(fontsize=6.5)
axs[1].legend(fontsize=6.5)
fig.suptitle("Spring test: Q1 over-prediction of motion (seed means; deciles of GT speed; L3 exploratory)", fontsize=9)
fig.tight_layout()
savefig(fig, "sp_q1_speed.png")

# ================================================================== Q2 M1 seed 1 blow-up
q2 = {}
for s in SEEDS:
    st = ST[("M1", s)]
    q2[f"s{s}"] = dict(epe=st["epe"], seq_epe=dict(zip(SEQS, st["seq_epe"].tolist())),
                       worst_clip=clip_names[int(np.argmax(st["clip_epe"]))], worst_clip_epe=float(st["clip_epe"].max()),
                       median_clip_epe=float(np.median(st["clip_epe"])),
                       frac_epe_gt5=st["frac_epe_gt5"], frac_est_gt10x=st["frac_est_gt10x"], est_ratio=st["est_ratio"],
                       est_mean=st["est_mean"])
    pe = np.where(FV, PX[("M1", s)]["epe"], np.nan)
    flat = np.sort(pe[FV].astype(np.float64))[::-1]
    q2[f"s{s}"]["top1pct_share_of_sum"] = float(flat[:len(flat) // 100].sum() / flat.sum())
    q2[f"s{s}"]["max_epe"] = float(flat[0])
    q2[f"s{s}"]["n_clips_epe_gt5"] = int((st["clip_epe"] > 5).sum())
    # growth over time inside the clip: mean EPE in first vs last 10% of each clip
    first = np.mean([st["rowmean"][(crow == c) & (tnorm <= 0.1)].mean() for c in range(len(lens))])
    last = np.mean([st["rowmean"][(crow == c) & (tnorm >= 0.9)].mean() for c in range(len(lens))])
    q2[f"s{s}"]["first10"], q2[f"s{s}"]["last10"] = float(first), float(last)
    rs = []
    for c in range(len(lens)):
        m = crow == c
        x = np.argsort(np.argsort(st["rowmean"][m])), np.arange(m.sum())
        rs.append(np.corrcoef(x[0], x[1])[0, 1])
    q2[f"s{s}"]["median_spearman_epe_vs_time"] = float(np.median(rs))
    # clips where the last 10% is >2x the first 10%
    gr = [st["rowmean"][(crow == c) & (tnorm >= 0.9)].mean() / st["rowmean"][(crow == c) & (tnorm <= 0.1)].mean()
          for c in range(len(lens))]
    q2[f"s{s}"]["n_clips_late_gt_2x_early"] = int(sum(g > 2 for g in gr))
R["q2"] = q2
# same clips, other arms: clip EPE of M1 s1's worst-5 clips
worst5 = np.argsort(-ST[("M1", 1)]["clip_epe"])[:5]
R["q2"]["worst5"] = [dict(clip=clip_names[c], m1_s1=float(ST[("M1", 1)]["clip_epe"][c]),
                          m1_s0=float(ST[("M1", 0)]["clip_epe"][c]), m1_s2=float(ST[("M1", 2)]["clip_epe"][c]),
                          M2=float(amean("M2", "clip_epe")[c]), M3=float(amean("M3", "clip_epe")[c]),
                          M4=float(amean("M4", "clip_epe")[c]), OursS=float(amean("Ours-S", "clip_epe")[c]),
                          zero=float(np.nanmean(GS[OFF[c]:OFF[c + 1]]))) for c in worst5]
# frames > 10x mean GT? other runs with big blow-ups
R["q2"]["other_runs_epe_gt5"] = {f"{a}_s{s}": ST[(a, s)]["frac_epe_gt5"] for (a, s) in ST if ST[(a, s)]["frac_epe_gt5"] > 0.001 and not (a == "M1" and s == 1)}
mb = {s: jl(RUNS / RUNOF[("M1", s)] / "spring_best" / "metrics.json")["epe"] for s in SEEDS}
R["q2"]["m1_best_ckpt_epe"] = mb
R["q2"]["m1_s1_last_vs_best_iter"] = (jl(RUNS / RUNOF[("M1", 1)] / "spring_last" / "metrics.json").get("ckpt_iter"),
                                      jl(RUNS / RUNOF[("M1", 1)] / "spring_best" / "metrics.json").get("ckpt_iter"))
save_csv("q2_m1_clip_epe.csv", ["clip", "M1_s0", "M1_s1", "M1_s2", "M2", "M3", "M4", "OursS", "zero_flow"],
         [[clip_names[c]] + [ST[("M1", s)]["clip_epe"][c] for s in SEEDS] +
          [amean(a, "clip_epe")[c] for a in ("M2", "M3", "M4", "Ours-S")] + [float(np.nanmean(GS[OFF[c]:OFF[c + 1]]))]
          for c in range(len(lens))])

from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel  # noqa: E402

hxu, hxv = get_hex_coords(15)
hx, hy = hex_to_pixel(hxu, hxv)


def hexplot(ax, vals, title, cmap=None, vmin=None, vmax=None, rgb=None, norm=None):
    if rgb is not None:
        ax.scatter(hx, hy, c=rgb, s=9, marker="h", linewidths=0)
        sc = None
    else:
        sc = ax.scatter(hx, hy, c=vals, s=9, marker="h", linewidths=0, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=6.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return sc


def flow_rgb(f, vmax):
    ang = (np.arctan2(f[1], f[0]) + np.pi) / (2 * np.pi)
    sat = np.clip(np.hypot(f[0], f[1]) / vmax, 0, 1)
    return hsv_to_rgb(np.stack([ang, sat, np.ones_like(sat)], -1))


fig, axs = plt.subplots(2, 2, figsize=(11, 8))
tb = np.minimum((tnorm * 10).astype(int), 9)
for e in ("M1 s1", "M1 (s0,s2)", "M2", "M3", "M4", "Ours-S"):
    a, ss = ENT[e]
    ys = []
    for s in ss:
        rm = ST[(a, s)]["rowmean"]
        ys.append([np.mean([rm[(crow == c) & (tb == b)].mean() for c in range(len(lens))]) for b in range(10)])
    axs[0, 0].plot(np.arange(10) * 10 + 5, np.mean(ys, 0), color=ECOL[e], ls="--" if e in ("M3", "M1 (s0,s2)") else "-", marker="o", ms=3,
                   label=ELAB[e])
axs[0, 0].axhline(R["mean_gt_speed"], color=GREY, ls=":", lw=1.5, label="zero-flow EPE (= mean GT speed)")
axs[0, 0].set(yscale="log", xlabel="position within clip [% of clip length]", ylabel="mean EPE [lattice units / frame]",
              title="(a) EPE over time within a clip (mean of 24 clips; lower is better)")
axs[0, 0].legend(fontsize=6)
for c in range(len(lens)):
    m = crow == c
    axs[0, 1].plot(tnorm[m][::5] * 100, ST[("M1", 1)]["rowmean"][m][::5], color=AMBER, lw=0.8, alpha=0.8)
axs[0, 1].set(yscale="log", xlabel="position within clip [% of clip length]", ylabel="frame-mean EPE [lattice units / frame]",
              title="(b) M1 seed 1 (connectome, amber): each line = 1 of 24 clips")
x = np.arange(len(SEQS))
for k, s in enumerate(SEEDS):
    axs[1, 0].bar(x + (k - 1) * 0.27, ST[("M1", s)]["seq_epe"], 0.27, color=AMBER, alpha=[0.45, 1, 0.7][k],
                  label=f"M1 seed {s} (connectome)")
axs[1, 0].plot(x, [np.nanmean(GS[rseq == i]) for i in range(len(SEQS))], "o", color=GREY, label="zero-flow floor")
axs[1, 0].set(yscale="log", xticks=x, xticklabels=SEQS, xlabel="Spring sequence", ylabel="EPE [lattice units / frame]",
              title="(c) M1 EPE per sequence")
axs[1, 0].legend(fontsize=6)
_pe = PX[("M1", 1)]["epe"]
R["q2"]["m1s1_nonfinite_valid"] = int((~np.isfinite(_pe[FV])).sum())
mapv = np.array([np.mean(_pe[:, j][FV[:, j] & np.isfinite(_pe[:, j])]) for j in range(721)])
print("mapv", np.nanmin(mapv), np.nanmax(mapv), np.isnan(mapv).sum(), R["q2"]["m1s1_nonfinite_valid"])
from matplotlib.colors import LogNorm  # noqa: E402

sc = hexplot(axs[1, 1], mapv, "(d) M1 seed 1: time-mean EPE per hexal [lattice units / frame]", cmap="magma",
             norm=LogNorm(vmin=max(mapv.min(), 0.1), vmax=mapv.max()))
fig.colorbar(sc, ax=axs[1, 1], shrink=0.8, label="EPE [lattice units / frame], log scale")
R["q2"]["hexal_map"] = dict(min=float(mapv.min()), median=float(np.median(mapv)), max=float(mapv.max()),
                            frac_hexals_gt5=float((mapv > 5).mean()))
fig.suptitle("Spring test: Q2 M1 seed 1 blow-up (EPE 254) from saved per-pixel errors (L3 exploratory)", fontsize=9)
fig.tight_layout()
savefig(fig, "sp_q2_blowup.png")

# ================================================================== Q3 per sequence EPE
ZSEQ = np.array([np.nanmean(GS[rseq == i]) for i in range(len(SEQS))])
R["zero_seq"] = dict(zip(SEQS, ZSEQ.tolist()))
LEARN = list(ENT.keys())
SEQ_EPE = {e: eg(e, "seq_epe") for e in ENT}
R["q3"] = dict(seq_epe={e: dict(zip(SEQS, v.tolist())) for e, v in SEQ_EPE.items()})
cmp_ = ["M2", "M3", "M1 (s0,s2)", "M4", "Ours-S"]
R["q3"]["beats_zero_count"] = {e: int((SEQ_EPE[e] < ZSEQ).sum()) for e in ENT}
R["q3"]["best_arm_per_seq"] = {s: min(ENT, key=lambda e: SEQ_EPE[e][i]) for i, s in enumerate(SEQS)}
R["q3"]["best_among_main5"] = {s: min(cmp_, key=lambda e: SEQ_EPE[e][i]) for i, s in enumerate(SEQS)}
R["q3"]["calib"] = {e: dict(est_ratio=float(eg(e, "est_ratio")), seq_ratio=dict(zip(SEQS, eg(e, "seq_est_ratio").tolist())),
                            frac_est_lt_gt=float(eg(e, "frac_est_lt_gt")), frac_est_gt10x=float(eg(e, "frac_est_gt10x")))
                    for e in ENT}
R["q3"]["seed_sd_epe"] = {e: float(np.std([ST[(ENT[e][0], s)]["epe"] for s in ENT[e][1]], ddof=1)) if len(ENT[e][1]) > 1 else None
                          for e in ENT}
# M2/M3 vs Ours-S paired by seed per sequence
R["q3"]["m2m3_better_than_oursS_seqseed_pairs"] = {
    a: int(sum(ST[(a, s)]["seq_epe"][i] < ST[("Ours-S", s)]["seq_epe"][i] for s in SEEDS for i in range(len(SEQS)))) for a in ("M2", "M3")}
R["q3"]["m2m3_better_than_m4_pairs"] = {
    a: int(sum(ST[(a, s)]["seq_epe"][i] < ST[("M4", s)]["seq_epe"][i] for s in SEEDS for i in range(len(SEQS)))) for a in ("M2", "M3")}
R["q3"]["mean_gap_to_zero_per_seq"] = {e: (SEQ_EPE[e] - ZSEQ).tolist() for e in cmp_}
save_csv("q3_epe_per_sequence.csv", ["entity"] + SEQS, [[e] + SEQ_EPE[e].tolist() for e in ENT] + [["zero-flow"] + ZSEQ.tolist()])
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
w = 0.14
for k, e in enumerate(cmp_):
    axs[0].bar(np.arange(len(SEQS)) + (k - 2) * w, SEQ_EPE[e] / ZSEQ, w, color=ECOL[e], hatch=EHATCH[e] if e == "x" else "",
               label=ELAB[e])
axs[0].axhline(1, color=GREY, ls=":", lw=1.8, label="zero-flow floor (ratio 1)")
axs[0].set(xticks=np.arange(len(SEQS)), xticklabels=SEQS, xlabel="Spring sequence",
           ylabel="EPE / zero-flow EPE [ratio] (lower is better)", title="(a) per-sequence EPE relative to zero-flow (L3)")
axs[0].legend(fontsize=6, ncol=2)
for e in cmp_:
    axs[1].plot(range(len(SEQS)), eg(e, "seq_est_ratio"), "o-", color=ECOL[e], ms=3, label=ELAB[e])
axs[1].axhline(1, color=GREY, ls="--", label="predicted = GT speed")
axs[1].set(xticks=range(len(SEQS)), xticklabels=SEQS, xlabel="Spring sequence",
           ylabel="est. predicted speed / GT speed [ratio]", title="(b) speed calibration per sequence (estimated)")
axs[1].legend(fontsize=6)
fig.suptitle("Spring test: Q3 null models (coral) vs connectome (amber), deep learning (teal), ours (violet); seed means; L3", fontsize=9)
fig.tight_layout()
savefig(fig, "sp_q3_perseq.png")

# ================================================================== Q4 ablations / K curve / Ours-L
q4 = {}
for X in ("noEPE", "K1", "noSI", "M4"):
    d_epe = np.array([[ST[(X, s)]["seq_epe"][i] - ST[("Ours-S", s)]["seq_epe"][i] for i in range(len(SEQS))] for s in SEEDS])
    d_dep = np.array([[ST[(X, s)]["seq_dep"][i] - ST[("Ours-S", s)]["seq_dep"][i] for i in range(len(SEQS))] for s in SEEDS])
    q4[X] = dict(d_epe_mean_per_seq=d_epe.mean(0).tolist(), d_dep_mean_per_seq=d_dep.mean(0).tolist(),
                 X_worse_epe_pairs=int((d_epe > 0).sum()), X_worse_dep_pairs=int((d_dep > 0).sum()),
                 X_worse_epe_seeds_pooled=int(sum(ST[(X, s)]["epe"] > ST[("Ours-S", s)]["epe"] for s in SEEDS)),
                 X_worse_dep_seeds_pooled=int(sum(ST[(X, s)]["dep"] > ST[("Ours-S", s)]["dep"] for s in SEEDS)),
                 X_worse_epe_seqs_mean=int((d_epe.mean(0) > 0).sum()), X_worse_dep_seqs_mean=int((d_dep.mean(0) > 0).sum()),
                 mean_d_epe=float(d_epe.mean()), mean_d_dep=float(d_dep.mean()))
# Ours-L vs Ours-S
dL = np.array([[ST[("Ours-L", s)]["seq_epe"][i] - ST[("Ours-S", s)]["seq_epe"][i] for i in range(len(SEQS))] for s in SEEDS])
dLd = np.array([[ST[("Ours-L", s)]["seq_dep"][i] - ST[("Ours-S", s)]["seq_dep"][i] for i in range(len(SEQS))] for s in SEEDS])
q4["Ours-L"] = dict(d_epe_mean_per_seq=dL.mean(0).tolist(), d_dep_mean_per_seq=dLd.mean(0).tolist(),
                    X_worse_epe_pairs=int((dL > 0).sum()), X_worse_dep_pairs=int((dLd > 0).sum()),
                    X_worse_epe_seeds_pooled=int(sum(ST[("Ours-L", s)]["epe"] > ST[("Ours-S", s)]["epe"] for s in SEEDS)),
                    X_worse_dep_seeds_pooled=int(sum(ST[("Ours-L", s)]["dep"] > ST[("Ours-S", s)]["dep"] for s in SEEDS)),
                    X_worse_epe_seqs_mean=int((dL.mean(0) > 0).sum()), X_worse_dep_seqs_mean=int((dLd.mean(0) > 0).sum()),
                    mean_d_epe=float(dL.mean()), mean_d_dep=float(dLd.mean()),
                    est_ratio_L=float(amean("Ours-L", "est_ratio")), est_ratio_S=float(amean("Ours-S", "est_ratio")),
                    bin_epe_L=amean("Ours-L", "bin_epe").tolist(), bin_epe_S=amean("Ours-S", "bin_epe").tolist())
q4["M4_vs_oursS_wins_epe_seeds"] = None
# any-time K curve (spring_last_k*): EPE pooled & per sequence
KC = {}
for arm in ("Ours-S", "K1", "Ours-L"):
    per_k = {}
    for k in (1, 2, 3, 4):
        seq_sum, seq_n, tot, totn = np.zeros((3, len(SEQS))), np.zeros((3, len(SEQS))), [], []
        for si, s in enumerate(SEEDS):
            m = jl(RUNS / RUNOF[(arm, s)] / f"spring_last_k{k}" / "metrics.json")
            for c in m["clips"]:
                i = SEQS.index(c["name"].split("_")[1])
                seq_sum[si, i] += c["epe"] * c["n_flow_valid"]
                seq_n[si, i] += c["n_flow_valid"]
            tot.append(m["epe"])
        per_k[k] = dict(epe=float(np.mean(tot)), seed_epe=tot, seq=(seq_sum / seq_n).mean(0).tolist())
    KC[arm] = per_k
    per_k["k4_minus_k1_per_seq"] = (np.array(per_k[4]["seq"]) - np.array(per_k[1]["seq"])).tolist()
    per_k["n_seq_k4_better_than_k1"] = int((np.array(per_k["k4_minus_k1_per_seq"]) < 0).sum())
    per_k["seeds_k4_better_than_k1"] = int(sum(per_k[4]["seed_epe"][i] < per_k[1]["seed_epe"][i] for i in range(3)))
R["q4"], R["q4"]["kcurve"] = q4, KC
save_csv("q4_paired_diffs.csv", ["X_minus_OursS", "metric"] + SEQS,
         [[X, m] + q4[X][k] for X in ("noEPE", "K1", "noSI", "M4", "Ours-L") for m, k in (("epe", "d_epe_mean_per_seq"), ("dep_aligned", "d_dep_mean_per_seq"))])

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
xs = np.arange(len(SEQS))
for k, X in enumerate(("noEPE", "K1", "noSI", "M4")):
    axs[0].bar(xs + (k - 1.5) * 0.2, q4[X]["d_epe_mean_per_seq"], 0.2, color=ARMS[X][1], hatch=ARMS[X][2], edgecolor="white",
               label=ARMS[X][3] if X != "M4" else "M4 (deep learning)")
axs[0].axhline(0, color="k", lw=0.8)
axs[0].set(xticks=xs, xticklabels=SEQS, xlabel="Spring sequence", ylabel="EPE(X) - EPE(Ours-S) [lattice units / frame]\n(> 0: Ours-S better)",
           title="(a) paired-by-seed difference vs Ours-S, per sequence (L3)")
axs[0].legend(fontsize=6)
for arm, ls in (("Ours-S", "-"), ("K1", "--"), ("Ours-L", "-")):
    axs[1].plot([1, 2, 3, 4], [KC[arm][k]["epe"] for k in (1, 2, 3, 4)], ls, marker="o", ms=4, color=ARMS[arm][1], label=ARMS[arm][3])
axs[1].axhline(R["floor"]["zero"], color=GREY, ls=":", lw=1.8, label="zero-flow floor")
axs[1].axhline(R["floor"]["lk"], color=GREY, ls="-.", lw=1.8, label="Lucas-Kanade floor")
axs[1].set(xticks=[1, 2, 3, 4], xlabel="inner recurrent steps K at evaluation [count]", ylabel="Spring EPE [lattice units / frame]",
           title="(b) any-time K curve (seed mean)")
axs[1].legend(fontsize=6)
axs[2].bar(xs - 0.2, SEQ_EPE["Ours-S"], 0.4, color=VIOLET, label="Ours-S (ours)")
axs[2].bar(xs + 0.2, SEQ_EPE["Ours-L"], 0.4, color="#483D8B", hatch="\\\\", edgecolor="white", label="Ours-L (ours, 274k params)")
axs[2].plot(xs, ZSEQ, "o", color=GREY, label="zero-flow floor")
axs[2].set(xticks=xs, xticklabels=SEQS, xlabel="Spring sequence", ylabel="EPE [lattice units / frame]  (lower is better)",
           title="(c) Ours-L vs Ours-S per sequence")
axs[2].legend(fontsize=6)
fig.suptitle("Spring test: Q4 our method - ablations, K curve, size (violet = ours; lighter/hatched = ablations; L3)", fontsize=9)
fig.tight_layout()
savefig(fig, "sp_q4_ablation.png")

# ================================================================== Q5 depth
DCONST = np.full(len(SEQS), np.nan)
sq = np.zeros(len(SEQS))
cn = np.zeros(len(SEQS))
GMED = {}
for c in range(len(lens)):
    sl = slice(OFF[c], OFF[c + 1])
    g = GT_LOGD[sl][DV[sl]]
    err = g - np.median(g)
    i = SEQS.index(seq_of_clip[c])
    sq[i] += (err.astype(np.float64) ** 2).sum()
    cn[i] += err.size
    GMED.setdefault(i, []).append(float(np.median(g)))
DCONST = np.sqrt(sq / cn)
R["q5"] = dict(const_floor_seq=dict(zip(SEQS, DCONST.tolist())),
               const_floor_pooled=float(np.sqrt(sq.sum() / cn.sum())), const_floor_metrics_json=R["floor"]["depth_const"],
               gt_median_logdepth_seq={SEQS[i]: float(np.mean(v)) for i, v in GMED.items()})
DEP = {e: eg(e, "seq_dep") for e in ENT}
R["q5"]["dep_seq"] = {e: dict(zip(SEQS, DEP[e].tolist())) for e in ENT}
R["q5"]["beats_floor_seqs"] = {e: int((DEP[e] < DCONST).sum()) for e in ENT}
R["q5"]["beats_floor_per_seq_arms"] = {s: [e for e in ENT if DEP[e][i] < DCONST[i]] for i, s in enumerate(SEQS)}
R["q5"]["pooled"] = {e: float(eg(e, "dep")) for e in ENT}
R["q5"]["best_arm_per_seq"] = {s: min(ENT, key=lambda e: DEP[e][i]) for i, s in enumerate(SEQS)}
# raw vs aligned: per clip shift and raw rmse (std units) from metrics.json
SH, RAW = {}, {}
for e, (a, ss) in ENT.items():
    sh_, rw_ = np.zeros(len(lens)), np.zeros(len(lens))
    for s in ss:
        m = jl(RUNS / RUNOF[(a, s)] / "spring_last" / "metrics.json")
        sh_ += np.array([c["align_shift_nat_log"] for c in m["clips"]]) / len(ss)
        rw_ += np.array([c["depth_rmse"] for c in m["clips"]]) / len(ss)
    SH[e], RAW[e] = sh_, rw_
sel = ["M2", "M4", "Ours-S"]
R["q5"]["shift_per_seq"] = {e: {s: float(np.mean([SH[e][c] for c in range(len(lens)) if seq_of_clip[c] == s])) for s in SEQS} for e in sel}
R["q5"]["raw_nat_per_seq"] = {e: {s: float(np.mean([RAW[e][c] for c in range(len(lens)) if seq_of_clip[c] == s]) * STD) for s in SEQS} for e in sel}
R["q5"]["shift_mean_all_arms"] = float(np.mean([SH[e].mean() for e in ENT if e != "M1 s1"]))
R["q5"]["shift_range_all_arms"] = [float(min(SH[e].min() for e in ENT)), float(max(SH[e].max() for e in ENT))]
R["q5"]["shift_sd_across_clips_OursS"] = float(SH["Ours-S"].std())
R["q5"]["std_nat_log"] = STD
R["q5"]["corr_shift_gtmedian_OursS"] = float(np.corrcoef(SH["Ours-S"], [np.mean([m for m in GMED[SEQS.index(seq_of_clip[c])]]) for c in range(len(lens))])[0, 1])
R["q5"]["raw_pooled_nat_OursS"] = float(np.sqrt(np.mean(RAW["Ours-S"] ** 2)) * STD)
save_csv("q5_depth_aligned_per_sequence.csv", ["entity"] + SEQS + ["pooled"],
         [[e] + DEP[e].tolist() + [R["q5"]["pooled"][e]] for e in ENT] + [["constant floor"] + DCONST.tolist() + [R["q5"]["const_floor_pooled"]]])
show5 = ["M1 (s0,s2)", "M1 s1", "M2", "M3", "M4", "M5", "M8", "Ours-S", "noEPE", "K1", "noSI", "Ours-L"]
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
mat = np.array([DCONST - DEP[e] for e in show5])
im = axs[0].imshow(mat, cmap="RdBu", vmin=-0.8, vmax=0.8, aspect="auto")
axs[0].set(xticks=range(len(SEQS)), xticklabels=SEQS, yticks=range(len(show5)), xlabel="Spring sequence",
           title="(a) constant-floor RMSE - model RMSE [nat-log]; blue (> 0) = model beats floor (L3)")
axs[0].set_yticklabels([ELAB[e].split(" (")[0] if e in ARMS else e for e in show5])
for lab, e in zip(axs[0].get_yticklabels(), show5):
    lab.set_color(ECOL[e])
for i in range(len(show5)):
    for j in range(len(SEQS)):
        axs[0].text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=axs[0], shrink=0.8, label="floor - model [nat-log depth RMSE]")
for e in ("M2", "M4", "Ours-S"):
    axs[1].plot(SH[e], RAW[e] * STD, "o", ms=4, color=ECOL[e], label=ELAB[e])
axs[1].set(xlabel="per-clip median alignment shift [nat-log]", ylabel="raw RMSE (std units x std) [nat-log]",
           title="(b) raw RMSE vs alignment shift (24 clips)")
axs[1].legend(fontsize=6)
fig.suptitle("Spring test: Q5 depth, aligned RMSE vs constant-depth floor (grey floor; teal/coral/violet arms; lower is better; L3)", fontsize=9)
fig.tight_layout()
savefig(fig, "sp_q5_depth.png")

# ================================================================== Q6 qualitative
S0 = ST[("Ours-S", 0)]
zero_clip = np.array([np.nanmean(GS[OFF[c]:OFF[c + 1]]) for c in range(len(lens))])
ratio_clip = S0["clip_epe"] / zero_clip
cg, cf = int(np.argmin(ratio_clip)), int(np.argmax(ratio_clip))


def pick_frame(c, good):
    rows_ = np.arange(OFF[c], OFF[c + 1])
    sp = np.nanmean(GS[rows_], 1)
    ok = sp >= np.median(sp)
    r = S0["rowmean"][rows_] / np.maximum(sp, 1e-6)
    r = np.where(ok, r, np.nan)
    return int(rows_[np.nanargmin(r) if good else np.nanargmax(r)])


QUAL = []
fig, axs = plt.subplots(2, 8, figsize=(22, 6.8))
for r_, (c, good) in enumerate(((cg, True), (cf, False))):
    row = pick_frame(c, good)
    g = GT_FLOW[row]
    vmax = float(np.percentile(np.hypot(g[0], g[1]), 99)) or 1.0
    e_s = np.where(FV[row], PX[("Ours-S", 0)]["epe"][row], np.nan)
    e_m = np.where(FV[row], PX[("M1", 0)]["epe"][row], np.nan)
    gsp = np.where(FV[row], gts[row], np.nan)
    sub_i = row // SUB if row % SUB == 0 else None
    es = EST[RUNOF[("Ours-S", 0)]]
    est_row = es[min(row // SUB, es.shape[0] - 1)]  # nearest estimated frame (every 4th)
    dg = GT_LOGD[row]
    dp = np.where(DV[row], dg + PX[("Ours-S", 0)]["dal"][row], np.nan)
    dgm = np.where(DV[row], dg, np.nan)
    lo, hi = np.nanpercentile(dgm, [2, 98])
    lab = f"{'GOOD' if good else 'FAIL'}: {clip_names[c][7:]}, frame {row - OFF[c]}/{lens[c]}"
    hexplot(axs[r_, 0], LUM[row], f"{lab}\nluminance [a.u., 2-98 pct]", cmap="gray", vmin=np.percentile(LUM[row], 2), vmax=np.percentile(LUM[row], 98))
    hexplot(axs[r_, 1], None, f"GT flow, colour wheel\n(hue = direction, sat. = speed/{vmax:.2f})", rgb=flow_rgb(g, vmax))
    vm = float(np.nanpercentile(np.concatenate([gsp, e_s, e_m]), 98))
    s1 = hexplot(axs[r_, 2], gsp, f"GT speed (= zero-flow EPE {np.nanmean(gsp):.2f})", cmap="viridis", vmin=0, vmax=vm)
    hexplot(axs[r_, 3], est_row, f"Ours-S est. predicted speed\n(frame {min(row // SUB, es.shape[0]-1) * SUB - OFF[c]}, mean {np.nanmean(est_row):.2f})",
            cmap="viridis", vmin=0, vmax=vm)
    hexplot(axs[r_, 4], e_s, f"Ours-S EPE (mean {np.nanmean(e_s):.2f})", cmap="magma", vmin=0, vmax=vm)
    hexplot(axs[r_, 5], e_m, f"M1 seed 0 EPE (mean {np.nanmean(e_m):.2f})", cmap="magma", vmin=0, vmax=vm)
    drm = float(np.sqrt(np.nanmean(np.where(DV[OFF[c]:OFF[c+1]], PX[("Ours-S", 0)]["dal"][OFF[c]:OFF[c+1]], np.nan) ** 2)))
    s2 = hexplot(axs[r_, 6], dgm, "GT log-depth [nat-log]", cmap="cividis", vmin=lo, vmax=hi)
    hexplot(axs[r_, 7], dp, f"Ours-S depth, median-aligned [nat-log]\n(clip aligned RMSE {drm:.2f})", cmap="cividis", vmin=lo, vmax=hi)
    QUAL.append(dict(kind="good" if good else "fail", clip=clip_names[c], row=int(row - OFF[c]), clip_len=int(lens[c]),
                     clip_ratio_epe_over_zero=float(ratio_clip[c]), clip_epe=float(S0["clip_epe"][c]), clip_zero=float(zero_clip[c]),
                     frame_epe_oursS=float(np.nanmean(e_s)), frame_epe_m1=float(np.nanmean(e_m)), frame_gt_speed=float(np.nanmean(gsp)),
                     frame_est_speed=float(np.nanmean(est_row)),
                     depth_rmse_clip_oursS=float(np.sqrt(np.nanmean(np.where(DV[OFF[c]:OFF[c+1]], PX[("Ours-S", 0)]["dal"][OFF[c]:OFF[c+1]], np.nan) ** 2))),
                     m1_s0_clip_epe=float(ST[("M1", 0)]["clip_epe"][c])))
    fig.colorbar(s1, ax=axs[r_, 2:6].tolist(), shrink=0.8, pad=0.01, label="speed / EPE [lattice units / frame]")
    fig.colorbar(s2, ax=axs[r_, 6:8].tolist(), shrink=0.8, pad=0.01, label="log depth [nat-log, 2-98 pct]")
R["q6"] = QUAL
save_csv("q6_qualitative_frames.csv", list(QUAL[0].keys()), [list(q.values()) for q in QUAL])
fig.suptitle("Spring test: Q6 Ours-S (violet method, seed 0) good vs failing clip - all L3, selected post hoc by clip-level EPE / zero-flow EPE. "
             "Predicted flow VECTORS are not stored; speed is estimated from (|GT|, EPE, angular error). Lower EPE = better.", fontsize=8)
wax = fig.add_axes([0.905, 0.93, 0.07, 0.07])
X_, Y_ = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60))
wim = hsv_to_rgb(np.stack([(np.arctan2(Y_, X_) + np.pi) / (2 * np.pi), np.clip(np.hypot(X_, Y_), 0, 1), np.ones_like(X_)], -1))
wax.imshow(np.where((np.hypot(X_, Y_) <= 1)[..., None], wim, 1.0), origin="lower", extent=[-1, 1, -1, 1])
wax.set_xticks([-1, 0, 1]); wax.set_yticks([-1, 0, 1]); wax.tick_params(labelsize=5)
wax.set_title("wheel: hue = direction\nsat. = speed / vmax", fontsize=5)
savefig(fig, "sp_q6_qualitative.png")

# ---------------------------------------------------------------- extras: onset of M1 s1 blow-up, sequence shares
on = []
rm1 = ST[("M1", 1)]["rowmean"]
for c in range(len(lens)):
    m = np.flatnonzero(crow == c)
    cross = np.flatnonzero(rm1[m] > 5.0)
    on.append(dict(clip=clip_names[c], len=int(lens[c]), first_step_gt5=int(cross[0]) if len(cross) else None,
                   clip_epe=float(ST[("M1", 1)]["clip_epe"][c]),
                   epe_by_time_decile=[float(np.mean(rm1[m][(tnorm[m] >= b / 10) & (tnorm[m] <= (b + 1) / 10)])) for b in range(10)]))
R["q2"]["onset"] = [o for o in on if o["first_step_gt5"] is not None]
R["q2"]["max_len_nonblowup_clips"] = int(max(o["len"] for o in on if o["first_step_gt5"] is None))
R["q2"]["len_blowup_clips"] = [o["len"] for o in on if o["first_step_gt5"] is not None]
R["q2"]["n_clips"] = len(on)
R["q2"]["other_seeds_M1_s1_vs_seq0001_peak_rowmean"] = {f"M1_s{s_}": float(ST[("M1", s_)]["rowmean"][rseq == 0].max()) for s_ in SEEDS}
nv_seq = np.array([FV[rseq == i].sum() for i in range(len(SEQS))])
R["zero_share_by_seq"] = dict(zip(SEQS, (ZSEQ * nv_seq / (ZSEQ * nv_seq).sum()).tolist()))
R["valid_share_by_seq"] = dict(zip(SEQS, (nv_seq / nv_seq.sum()).tolist()))
R["q2"]["hexal_map"]["note"] = "time-mean over all 24 clips"
# ---------------------------------------------------------------- dump
(TABS / "error_analysis_spring_numbers.json").write_text(json.dumps(R, indent=1, default=float))
print("numbers ->", TABS / "error_analysis_spring_numbers.json")



# ================================================================== markdown (Thai); every number comes from R
def f(x, d=3):
    return f"{x:.{d}f}"


def tbl(h, rows):
    return "\n".join(["| " + " | ".join(h) + " |", "|" + "---|" * len(h)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


q1, q2, q3, q4, q5, q6 = (R[k] for k in ("q1", "q2", "q3", "q4", "q5", "q6"))
B = R["q1_bins"]
S0 = "Ours-S"
ents = ["M2", "M3", "M1 (s0,s2)", "M1 s1", "M4", "M8", "Ours-S", "Ours-L"]
t1 = tbl(["entity", "EPE", "est. |pred|/|GT| (bracket)", "EPE decile 1 / 5 / 10", "pred/GT decile 1 / 5 / 10", "ส่วนเกินเหนือ zero ใน decile 1-3", "decile ที่ชนะ zero / LK"],
         [[e, f(q1[e]["epe"]), f"{q1[e]['est_ratio']:.2f} [{q1[e]['lo']:.2f}-{q1[e]['hi']:.2f}]",
           " / ".join(f(B[e]["epe"][i]) for i in (0, 4, 9)), " / ".join(f(B[e]["est_over_gt"][i], 2) for i in (0, 4, 9)),
           (f"{100*q1[e]['excess_share_d123']:.0f}%" if q1[e]["excess_share_d123"] is not None else "-"),
           f"{q1[e]['n_bins_beating_zero']} / {q1[e]['n_bins_beating_lk']}"] for e in ents])
b = R["bins"]
sy = R["synth"]
w = q2["s1"]
on = [o for o in q2["onset"] if o["clip_epe"] > 100]
gr = [np.mean([o["epe_by_time_decile"][i + 1] / o["epe_by_time_decile"][i] for o in on]) for i in (6, 7, 8)]
w5 = q2["worst5"]
t2 = tbl(["clip", "M1 s1", "M1 s0", "M1 s2", "M2", "M3", "M4", "Ours-S", "zero-flow"],
         [[x["clip"][7:], f(x["m1_s1"], 1), f(x["m1_s0"]), f(x["m1_s2"]), f(x["M2"]), f(x["M3"]), f(x["M4"]), f(x["OursS"]), f(x["zero"])] for x in w5[:3]])
seqs = SEQS
ms = ["zero-flow", "M2", "M3", "M1 (s0,s2)", "M4", "M8", "Ours-S"]
t3 = tbl(["entity"] + seqs + ["ชนะ zero (ตัวอย่าง seq)"],
         [["zero-flow"] + [f(R["zero_seq"][s_]) for s_ in seqs] + ["-"]] +
         [[e] + [f(q3["seq_epe"][e][s_]) for s_ in seqs] + [f"{q3['beats_zero_count'][e]}/8"] for e in ms[1:]])
t3b = tbl(["entity"] + seqs, [[e] + [f(q3["calib"][e]["seq_ratio"][s_], 2) for s_ in seqs] for e in ("M2", "M3", "M4", "Ours-S")])
kc = q4["kcurve"]
t4 = tbl(["X (เทียบ Ours-S)", "EPE: X แย่กว่า (seed x seq จาก 24)", "EPE: X แย่กว่า (seed รวม /3)", "depth aligned: X แย่กว่า (จาก 24)", "depth: seed รวม /3", "mean dEPE", "mean dDepth"],
         [[X, q4[X]["X_worse_epe_pairs"], q4[X]["X_worse_epe_seeds_pooled"], q4[X]["X_worse_dep_pairs"], q4[X]["X_worse_dep_seeds_pooled"],
           f(q4[X]["mean_d_epe"]), f(q4[X]["mean_d_dep"])] for X in ("noEPE", "K1", "noSI", "M4", "Ours-L")])
t4k = tbl(["arm", "K=1", "K=2", "K=3", "K=4", "seed ที่ K=4 ดีกว่า K=1", "seq ที่ K=4 ดีกว่า K=1"],
          [[a, *[f(kc[a][k]["epe"]) for k in (1, 2, 3, 4)], f"{kc[a]['seeds_k4_better_than_k1']}/3", f"{kc[a]['n_seq_k4_better_than_k1']}/8"]
           for a in ("Ours-S", "K1", "Ours-L")])
ds_ = q5["dep_seq"]
t5 = tbl(["entity"] + seqs + ["pooled"],
         [["constant floor"] + [f(q5["const_floor_seq"][s_], 2) for s_ in seqs] + [f(q5["const_floor_pooled"], 3)]] +
         [[e] + [("**" + f(ds_[e][s_], 2) + "**") if ds_[e][s_] < q5["const_floor_seq"][s_] else f(ds_[e][s_], 2) for s_ in seqs] + [f(q5["pooled"][e], 3)]
          for e in ("M2", "M3", "M4", "M1 (s0,s2)", "Ours-S", "noEPE", "K1", "noSI", "Ours-L")])
g_, f_ = q6
sh = q5["shift_per_seq"]["Ours-S"]
beat = q5["beats_floor_per_seq_arms"]
md = f"""# Error analysis: Spring test (phase 2) - L3

> ตัวเลขทุกตัวสร้างโดย `error_analysis_spring.py` จาก output ที่บันทึกไว้เท่านั้น (`runs/*/spring_last*/per_pixel.npz`, `metrics.json`, `floors_out/spring`, cache GT `data/spring_hex`); ไม่มีการรันโมเดลบน Spring/Sintel อีก. ทุกข้อสรุปเป็น **L3 (exploratory)** ยกเว้นตัวชี้วัด L1 ที่ทวนซ้ำ (EPE รวม, depth aligned RMSE). ตารางอยู่ที่ `results/error_analysis_spring/`, ภาพ `docs/figs/sp_*.png` (สี: connectome=amber, null=coral, deep learning=teal, วิธีของเรา=violet, ablation=violet อ่อน/ลาย, floors=เทา).
> ข้อจำกัดของข้อมูล: `per_pixel.npz` เก็บเฉพาะ error รายเฮกซัล ไม่เก็บค่าทำนาย จึง **ประมาณ** ความเร็วที่ทำนายจาก (|GT|, EPE, angular error) ด้วยการแก้สมการต่อเฮกซัล (ใช้ทุก {SUB} เฟรม). ทดสอบกับข้อมูลสังเคราะห์ (float16 เหมือนไฟล์จริง): อัตราส่วนเฉลี่ยประมาณ/จริง = {f(sy['proportional']['ratio'], 2)} (ความเร็วทำนายสัดส่วนกับ GT) และ {f(sy['lognormal']['ratio'], 2)} (lognormal; decile บนสุดเอนเอียง {f(sy['lognormal']['ratio_by_decile'][9], 1)}x) จึงให้ช่วงขอบเขตตรงๆ ควบคู่ (|EPE-|GT|| ถึง EPE+|GT|). ตรวจแล้ว: EPE/depth จาก per-pixel ตรง metrics.json (ต่างสูงสุด {R['check_max_rel_diff_epe_vs_metrics_json_excl_m1s1']:.1e}), GT speed ตรง cache, valid mask เหมือนกันทุก run. M1 seed 1 มี {R['nonfinite_epe_valid_float16_overflow']['M1_s1']} เฮกซัลที่ EPE เกิน float16 (ตั้งเป็น 65504 = ขอบล่าง) จึงใช้ EPE ระดับ clip จาก metrics.json แทนสำหรับตารางที่เป็น EPE รวม/ต่อ sequence.

บริบท: Spring มี 8 sequence (24 clip, {R['n_rows']} ก้าว 50 Hz), GT ช้า: zero-flow EPE = {f(R['floor']['zero'], 4)} (= ความเร็ว GT เฉลี่ย {f(R['mean_gt_speed'], 4)}), LK = {f(R['floor']['lk'], 4)}. ความเร็ว GT กระจุกที่ sequence 0033 ({100*R['valid_share_by_seq']['0033']:.0f}% ของเฮกซัล แต่ {100*R['zero_share_by_seq']['0033']:.0f}% ของ zero-flow EPE รวม) และ 0005 ({100*R['zero_share_by_seq']['0005']:.0f}%). แยก M1 เป็น "M1 seeds 0,2" กับ "M1 seed 1" เสมอ.

## 1. การทำนายการเคลื่อนไหวเกินจริง? (sp_q1_speed.png)

{t1}

(EPE เป็นค่าเฉลี่ย seed; decile = 10 ช่วงของ |GT| ตามที่ floors ใช้ เฉลี่ย GT ใน decile 1 = {f(b['mean_gt'][0], 4)}, decile 10 = {f(b['mean_gt'][9], 2)}; LK ใน decile 1 = {f(b['lk_epe'][0], 3)} ส่วน zero = {f(b['zero_floor_epe'][0], 4)}.)

- **ไม่ใช่โมเดลทุกตัว over-predict**: ความเร็วที่ประมาณเทียบ GT รวมของ M2/M3/Ours-S = {f(q1['M2']['est_ratio'],2)}/{f(q1['M3']['est_ratio'],2)}/{f(q1['Ours-S']['est_ratio'],2)} (under; ขอบเขตกว้าง {f(q1['Ours-S']['lo'],2)}-{f(q1['Ours-S']['hi'],2)}), M4 {f(q1['M4']['est_ratio'],2)}, Ours-L {f(q1['Ours-L']['est_ratio'],2)}, M8 {f(q1['M8']['est_ratio'],2)} (ขอบล่าง {f(q1['M8']['lo'],2)} > 1 จึง over-predict แน่นอน). ค่าเฉลี่ยรวมถูกถ่วงด้วย sequence เร็ว (0033) ที่ทุกโมเดลทายต่ำเกิน.
- **รูปแบบร่วมกัน (hexal เกือบนิ่ง over, เฮกซัลเร็ว under)**: ทุก arm ใน (b) ทายเร็วเกิน GT หลายสิบเท่าใน decile 1 (Ours-S {f(B['Ours-S']['est_over_gt'][0],0)}x, M2 {f(B['M2']['est_over_gt'][0],0)}x) และต่ำกว่า GT ใน decile 10 (Ours-S {f(B['Ours-S']['est_over_gt'][9],2)}x). EPE ใน decile 1 ของทุกโมเดลจึงสูงกว่า zero ({f(B['Ours-S']['epe'][0],3)} vs {f(b['zero_floor_epe'][0],4)}) ขณะที่ใน decile 10 เกือบเท่า zero ({f(B['Ours-S']['epe'][9],2)} vs {f(b['zero_floor_epe'][9],2)}) -> เป็นลักษณะ regression ไปหาความเร็วกลาง.
- **การแพ้ zero-flow กระจุกที่เฮกซัลเกือบนิ่งหรือไม่?** ส่วนเกินของ EPE เหนือ zero (ถ่วงน้ำหนักด้วยจำนวนเฮกซัล) ตกอยู่ใน decile 1-3 (30% ของเฮกซัล) {100*q1['Ours-S']['excess_share_d123']:.0f}% สำหรับ Ours-S, {100*q1['M2']['excess_share_d123']:.0f}% M2, {100*q1['M4']['excess_share_d123']:.0f}% M4, {100*q1['M1 (s0,s2)']['excess_share_d123']:.0f}% M1 (s0,s2) และ {100*q1['M1 (s0,s2)']['excess_share_d1234']:.0f}% ใน decile 1-4; ใช่ "ส่วนใหญ่" สำหรับโมเดลที่ calibrate พอใช้ แต่ไม่ใช่สำหรับ over-predictor (M8 {100*q1['M8']['excess_share_d123']:.0f}%, Ours-L {100*q1['Ours-L']['excess_share_d123']:.0f}%) ที่เสียทุกช่วงความเร็ว. M1 seed 1: {100*q1['M1 s1']['excess_share_d123']:.0f}% (ระเบิดนอกเฮกซัลนิ่ง, ดู 2). LK เอง *ชนะ* zero ใน decile 5-10 แต่แพ้ใน decile 1-4 (ส่วนต่างรวม LK-zero = {f(q1['_lk_total_minus_zero'],3)}); โมเดลที่เรียนรู้ชนะ zero ได้เพียง {max(q1[e]['n_bins_beating_zero'] for e in ents)} ใน 10 decile และชนะ LK ได้ {max(q1[e]['n_bins_beating_lk'] for e in ents)}.
- {100*q1['_zero_share_of_epe_top2_bins']:.0f}% ของ zero-flow EPE มาจาก decile 9-10 จึงเป็นบริเวณที่ตัดสิน EPE รวม.

## 2. M1 seed 1 ระเบิด (EPE {f(w['epe'],1)}) (sp_q2_blowup.png)

- **ที่ไหน**: ระเบิดเฉพาะ sequence 0001 (EPE {f(q2['s1']['seq_epe']['0001'],0)}; sequence อื่น {f(min(v for k,v in q2['s1']['seq_epe'].items() if k!='0001'),2)}-{f(max(v for k,v in q2['s1']['seq_epe'].items() if k!='0001'),2)}) ใน 3 clip ของ 0001 ({', '.join(f"{o['clip'][7:]}: {f(o['clip_epe'],0)}" for o in on)}); clip อื่น {q2['n_clips']-3} clip ปกติ (median clip EPE {f(w['median_clip_epe'],2)}). seed 0/2 บน 0001 อยู่ที่ {f(q2['s0']['seq_epe']['0001'],2)} / {f(q2['s2']['seq_epe']['0001'],2)}.
{t2}
- **โตตามเวลา (ลักษณะ ODE/recurrent ไม่เสถียร)**: EPE เฉลี่ย 10% แรกของ clip = {f(w['first10'],2)}, 10% ท้าย = {f(w['last10'],0)}; ใน 3 clip ที่ระเบิด EPE ข้ามค่า 5 ที่ก้าว {', '.join(str(o['first_step_gt5']) for o in on)} (จาก {on[0]['len']} ก้าว) แล้วโตแบบ exponential ~{f(gr[0],1)}x/{f(gr[1],1)}x/{f(gr[2],1)}x ต่อ 10% ของ clip (decile 7->8->9->10); max frame-mean = {f(q2['other_seeds_M1_s1_vs_seq0001_peak_rowmean']['M1_s1'],0)} (seed 0/2: {f(q2['other_seeds_M1_s1_vs_seq0001_peak_rowmean']['M1_s0'],2)}/{f(q2['other_seeds_M1_s1_vs_seq0001_peak_rowmean']['M1_s2'],2)}). clip ยาวที่สุดที่ไม่ระเบิดมี {q2['max_len_nonblowup_clips']} ก้าว ขณะที่ 0001 ยาว {on[0]['len']} ก้าว และเริ่มระเบิดราวก้าว ~{min(o['first_step_gt5'] for o in on)} -> สอดคล้องกับสมมติฐานว่าความไม่เสถียรต้องการ clip ยาว (ไม่ยืนยัน; seed 0/2 บน 0001 ที่ยาวเท่ากันไม่ระเบิด). 
- **ขนาด**: ระดับเฮกซัล-เฟรม {100*w['frac_epe_gt5']:.1f}% มี EPE > 5 (= 10x ความเร็ว GT เฉลี่ย {f(R['mean_gt_speed'],2)}) เทียบ seed 0/2 {100*q2['s0']['frac_epe_gt5']:.1f}%/{100*q2['s2']['frac_epe_gt5']:.1f}% และ M2/M3/M4/Ours-S ราว 2.2% (เป็นเฮกซัลของ 0033 ที่ GT เร็วเอง); ความเร็วประมาณ > 10x ของ max(|GT|, 0.1) = {100*w['frac_est_gt10x']:.1f}% (seed 0/2: {100*q2['s0']['frac_est_gt10x']:.1f}%/{100*q2['s2']['frac_est_gt10x']:.1f}%). 1% ของเฮกซัล-เฟรมที่แย่สุดให้ {100*w['top1pct_share_of_sum']:.0f}% ของผลรวม EPE (ขอบล่าง). แผนที่ (d): เฮกซัลบางกลุ่มของตาเกิน EPE 100 (เวลาเฉลี่ย median {f(q2['hexal_map']['median'],2)}, max {f(q2['hexal_map']['max'],0)}) -> เป็นการระเบิดแบบ localised ในบางส่วนของตา ไม่ใช่ทั้งภาพ.
- checkpoint: seed 1 `best.pt` (iter {q2['m1_s1_last_vs_best_iter'][1]}) มี Spring EPE {f(q2['m1_best_ckpt_epe'][1],3)} เทียบ last (iter {q2['m1_s1_last_vs_best_iter'][0]}) {f(w['epe'],1)} -> เกิดตอนท้ายของการฝึก. ยืนยันแล้วว่าเหมือนกันโดยไม่ใช้ fused kernel.

## 3. ทำไม null (M2/M3) ทั่วไปได้ดีที่สุด? (sp_q3_perseq.png; เชิงพรรณนา ไม่อ้างเหตุผล)

{t3}

ความเร็วทำนาย / GT ต่อ sequence (ประมาณ):
{t3b}

- M2/M3 ชนะรายseq ในกลุ่ม M2/M3/M1/M4/Ours-S: best ต่อ seq = {', '.join(f"{k}:{v}" for k, v in q3['best_among_main5'].items())}. ใน 24 คู่ (seed x seq) M2 ดีกว่า Ours-S {q3['m2m3_better_than_oursS_seqseed_pairs']['M2']} คู่, M3 {q3['m2m3_better_than_oursS_seqseed_pairs']['M3']}; ดีกว่า M4 {q3['m2m3_better_than_m4_pairs']['M2']}/{q3['m2m3_better_than_m4_pairs']['M3']} คู่. ช่องว่างจริงเล็ก (EPE รวม M2 {f(q1['M2']['epe'])}, M3 {f(q1['M3']['epe'])}, Ours-S {f(q1['Ours-S']['epe'])}, M4 {f(q1['M4']['epe'])}) และ ไม่มี arm ใดชนะ zero ได้เกิน {max(q3['beats_zero_count'].values())}/8 sequence.
- สิ่งที่เห็น: M2/M3 ทายความเร็วต่ำกว่า GT ({f(q1['M2']['est_ratio'],2)}x) ใน sequence เร็ว (0033: {f(q3['calib']['M2']['seq_ratio']['0033'],2)}x) และทายเกินใน sequence เกือบนิ่ง (0007: {f(q3['calib']['M2']['seq_ratio']['0007'],1)}x) น้อยกว่า M1/M4 ({f(q3['calib']['M1 (s0,s2)']['seq_ratio']['0007'],1)}x/{f(q3['calib']['M4']['seq_ratio']['0007'],1)}x) -> EPE ต่อ seq ที่นิ่ง (0007, 0001) ต่ำกว่า. ความแปรปรวนข้าม seed ของ M2/M3 เล็ก (sd {f(q3['seed_sd_epe']['M2'],3)}/{f(q3['seed_sd_epe']['M3'],3)}).
- **สมมติฐาน (ยังไม่ทดสอบ)**: (H-a) การ regularise ของ wiring ที่ถูกสลับทำให้เอาต์พุตหดเข้าหาศูนย์ ซึ่งได้เปรียบเมื่อ GT ช้า; (H-b) wiring จริงของ M1 ให้ dynamics ที่ไวต่อ seed/ความยาว clip (ดู 2) ; (H-c) ฝึกบน Sintel (เร็วกว่า) ทำให้ prior ความเร็วของ M4/Ours-L สูงเกินสำหรับ Spring ขณะที่ null ไม่ได้ใช้โครงสร้าง; (H-d) ผลต่างเล็กใกล้ noise ของ 3 seed / 8 sequence. ข้อมูลนี้แยกสมมติฐานไม่ได้.

## 4. วิธีของเรา: ablation / K / ขนาด (sp_q4_ablation.png)

{t4}

(ต่อ sequence จัดคู่ด้วย seed; "X แย่กว่า" = EPE/depth สูงกว่า Ours-S.)
{t4k}

- EPE: noEPE แย่กว่าใน {q4['noEPE']['X_worse_epe_seqs_mean']}/8 sequence (ค่าเฉลี่ย seed), K1 {q4['K1']['X_worse_epe_seqs_mean']}/8, noSI {q4['noSI']['X_worse_epe_seqs_mean']}/8, M4 {q4['M4']['X_worse_epe_seqs_mean']}/8. ผลของ EPE-loss/multi-step ขึ้นกับ sequence (K1 ดีกว่า Ours-S ที่ 0010/0044) และขนาดผลเล็ก (mean dEPE {f(q4['K1']['mean_d_epe'],3)}-{f(q4['noSI']['mean_d_epe'],3)}). depth: ablation ไม่ต่างอย่างสม่ำเสมอ (noSI/noEPE mean dDepth {f(q4['noSI']['mean_d_dep'],3)}/{f(q4['noEPE']['mean_d_dep'],3)}).
- **K curve บน Spring**: Ours-S EPE *เพิ่ม* เล็กน้อยเมื่อ K สูงขึ้น ({f(kc['Ours-S'][1]['epe'])} -> {f(kc['Ours-S'][4]['epe'])}; K=4 ไม่ดีกว่า K=1 ในทุก seed) ซึ่งต่างจากข้อความ "K มากดีกว่า" บน Sintel; K1 (ฝึก K=1) แย่ลงเมื่อ eval K สูง (extrapolation) {f(kc['K1'][1]['epe'])} -> {f(kc['K1'][4]['epe'])}; Ours-L ดีขึ้นตาม K ({f(kc['Ours-L'][1]['epe'])} -> {f(kc['Ours-L'][4]['epe'])}, 3/3 seed) แต่ยังแพ้ Ours-S ทุก K.
- **ใหญ่กว่าแย่กว่า**: Ours-L แย่กว่า Ours-S ใน {q4['Ours-L']['X_worse_epe_pairs']}/24 คู่ EPE และ {q4['Ours-L']['X_worse_dep_pairs']}/24 คู่ depth (ทั้ง 3 seed รวม). Ours-L ทายเร็วเกิน ({f(q4['Ours-L']['est_ratio_L'],2)}x vs {f(q4['Ours-L']['est_ratio_S'],2)}x ของ Ours-S) และ EPE สูงกว่าใน decile ช้า (decile 1: {f(q4['Ours-L']['bin_epe_L'][0],2)} vs {f(q4['Ours-L']['bin_epe_S'][0],2)}); ต่างมากที่สุดที่ 0033 ({f(q4['Ours-L']['d_epe_mean_per_seq'][6],2)}).

## 5. Depth บน Spring (sp_q5_depth.png)

ตัวหนา = ชนะ constant floor (aligned RMSE nat-log, ต่ำกว่าดี):
{t5}

- floor ต่อ sequence คำนวณใหม่จาก GT (ค่าคงที่ + alignment ด้วย median ต่อ clip) pooled = {f(q5['const_floor_pooled'],4)} ตรง metrics {f(q5['const_floor_metrics_json'],4)}. floor ต่ำมากใน 0011/0010/0018 ({f(q5['const_floor_seq']['0011'],2)}/{f(q5['const_floor_seq']['0010'],2)}/{f(q5['const_floor_seq']['0018'],2)}): ไม่มี arm ใดชนะที่นั่น. ที่ชนะได้คือ 0044 ({len(beat['0044'])}/15 entity), 0005 ({len(beat['0005'])}), 0007 ({len(beat['0007'])}), 0001 ({len(beat['0001'])}); ชนะ floor ได้ {q5['beats_floor_seqs']['Ours-S']}/8 seq สำหรับ Ours-S, {q5['beats_floor_seqs']['M2']} M2, {q5['beats_floor_seqs']['M3']} M3, {q5['beats_floor_seqs']['M4']} M4, {q5['beats_floor_seqs']['Ours-L']} Ours-L (margin ส่วนใหญ่ < 0.2). depth ที่ดีที่สุดต่อ seq แตกต่างกัน ({', '.join(f'{k}:{v}' for k,v in q5['best_arm_per_seq'].items())}).
- **raw vs aligned**: ค่า raw (หน่วย std x {f(q5['std_nat_log'],3)} = nat-log) ของ Ours-S pooled = {f(q5['raw_pooled_nat_OursS'],2)} เทียบ aligned {f(q5['pooled']['Ours-S'],2)}; shift ต่อ clip เฉลี่ย {f(q5['shift_mean_all_arms'],2)} nat-log (ช่วง {f(q5['shift_range_all_arms'][0],1)}-{f(q5['shift_range_all_arms'][1],1)}; ~{np.exp(q5['shift_mean_all_arms']):.0f}x ใน depth) และสัมพันธ์กับ median log-depth ของ GT (r = {f(q5['corr_shift_gtmedian_OursS'],2)}; Ours-S shift 0007 = {f(sh['0007'],1)}, 0011 = {f(sh['0011'],1)}) -> โมเดลทายสเกลคงที่แบบ Sintel ไม่ปรับตามสเกล Spring; aligned depth วัดเฉพาะโครงสร้างสัมพัทธ์ ซึ่งใกล้ค่าคงที่ (ดู floor). สเกลสัมบูรณ์ของ Spring ขึ้นกับสมมติฐานหน่วย disparity (ต่าง 2x = ต่าง ln2 = 0.69 nat-log เท่านั้น) จึงไม่เปลี่ยน aligned metric.

## 6. ภาพเชิงคุณภาพ (sp_q6_qualitative.png; seed 0, เลือก post hoc)

- **ดี**: {g_['clip'][7:]} เฟรม {g_['row']}/{g_['clip_len']}: Ours-S EPE clip {f(g_['clip_epe'])} เทียบ zero {f(g_['clip_zero'])} (อัตราส่วน {f(g_['clip_ratio_epe_over_zero'],3)} = ต่ำสุดของ 24 clip, คือ "เสมอ" zero ไม่ได้ชนะชัด), เฟรมที่เลือก EPE {f(g_['frame_epe_oursS'],2)} vs M1 s0 {f(g_['frame_epe_m1'],2)}; depth aligned RMSE ของ clip {f(g_['depth_rmse_clip_oursS'],2)} (ต่ำกว่า floor ของ 0011 ไม่ได้ แต่ภาพแสดงโครงสร้างวัตถุกลางภาพ).
- **ล้มเหลว**: {f_['clip'][7:]} เฟรม {f_['row']}/{f_['clip_len']}: ฉากเกือบนิ่ง GT {f(f_['frame_gt_speed'],3)} แต่ Ours-S ทายเร็ว {f(f_['frame_est_speed'],2)} (EPE clip {f(f_['clip_epe'])} = {f(f_['clip_ratio_epe_over_zero'],1)}x zero), error กระจุกที่ขอบบนของตา; M1 s0 ทำได้ {f(f_['frame_epe_m1'],2)} บนเฟรมนี้. depth aligned RMSE {f(f_['depth_rmse_clip_oursS'],2)} (ภาพ depth ทำนายแทบราบ ขณะ GT มีโครงสร้าง).
- เวกเตอร์ flow ที่ทำนายไม่ได้บันทึก จึงแสดงความเร็วที่ประมาณและแผนที่ EPE แทน flow สีของโมเดล; GT flow แสดงด้วย colour wheel.

## ข้อจำกัด

1. มีเพียง **8 sequence** (24 clip ที่ไม่อิสระ: 3 vertical split ต่อ sequence) และ 3 seed; sequence 0033 และ 0005 ครอบงำ EPE รวม; ผลต่อ sequence จึงอาจเป็น noise.
2. **อัตราเฟรม 24 fps** เป็นสมมติฐาน (ไม่อยู่ใน zip); GT flow ไม่ถูก rescale ตามการ resample 50 Hz เหมือน Sintel pipeline, ความเร็วในหน่วย "lattice unit / frame เดิม".
3. **หน่วย disparity** (2x-grid, baseline 0.065 m) เป็นสมมติฐานที่ตรวจแบบ empirical; กระทบสเกลสัมบูรณ์ของ depth (raw RMSE) แต่ไม่กระทบ aligned metric.
4. **ประเมินครั้งเดียว** (test ใช้แล้ว): การเลือก clip/เฟรมใน Q6, decile และ threshold (EPE>5, 10x) กำหนดหลังเห็นผล = L3; ความเร็วทำนายเป็น *ค่าประมาณ* จาก error map (เอนเอียงขึ้นที่ decile เร็ว/ช้าสุด), ไม่มี LK ต่อ sequence (floors เก็บเฉพาะรวมและราย decile).
5. ไม่ทำ causal claim: ข้อ 3 เป็นรายการสมมติฐาน; ไม่มี multiple-comparison correction.
"""
(ROOT / "docs" / "ERROR_ANALYSIS_SPRING.md").write_text(md)
print("md lines:", md.count("\n") + 1)
