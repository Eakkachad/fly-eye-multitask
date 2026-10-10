"""Phase-2 analysis (PLAN.md amendment A9): Spring test, decisions P1-P4. Pure stdlib (no torch).

    python analyze_p2.py            # -> docs/RESULTS_P2.md, results/summary_p2.csv
    python analyze_p2.py --dry      # print only

Reads runs/<name>/spring_last/metrics.json (PRIMARY, L1), spring_best (secondary, L2), spring_last_k<K>
(any-time curves), runs/<name>/test_last/metrics.json (old Sintel test, second look, L2) and
floors_out/spring/metrics.json.  Written by eval_spring.py / floors_spring.py / eval.py.

Decision rules (strict comparisons on the raw floats; rounding is display only; paired by seed;
a decision with any missing seed of an involved arm is INCOMPLETE and capped at L2):
  P1  Ours-L mean Spring flow EPE < every phase-1 arm's mean AND < the LK floor.
        "Phase-1 arm" = the 10 full-data (f=1.0) arms M1..M8 incl. M6f/M7f (M8 at K=4); the f=0.25 data-efficiency
        variants are checked as a sensitivity row only.
  P2  Ours-S < M4 on Spring flow EPE and on aligned depth RMSE, each in >= 2/3 paired seeds.
  P3  ablations hurt, each in >= 2/3 paired seeds: noSI worse aligned-depth RMSE than Ours-S; noEPE worse flow
        EPE; K1 worse flow EPE  (K1 is evaluated at its training K=1; K=4 shown as sensitivity).
  P4  descriptive: Ours-S vs M1 on both tasks.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ("epe", "angular_deg", "depth_rmse_aligned", "depth_rmse", "depth_absrel")
LABEL = {"epe": "flow EPE", "angular_deg": "angular (deg)", "depth_rmse_aligned": "depth RMSE aligned (nat-log) [PRIMARY]",
         "depth_rmse": "depth RMSE raw (std units)", "depth_absrel": "depth AbsRel"}
L2_METRICS = ("epe", "angular_deg", "depth_rmse", "depth_absrel")   # eval.py keys (no alignment on old test)
SEEDS = (0, 1, 2)
P1_ARMS = ("m1", "m2", "m3", "m4", "m5", "m6", "m7", "m6f", "m7f", "m8")
P2_ARMS = ("oursS", "noSI", "noEPE", "K1", "oursL")
NAMES = {"m1": "M1 connectome", "m2": "M2 rewired", "m3": "M3 random ER", "m4": "M4 ConvGRU small",
         "m5": "M5 ConvGRU large", "m6": "M6 hybrid (real)", "m7": "M7 hybrid (rewired)",
         "m6f": "M6f frozen front-end + residual", "m7f": "M7f frozen front-end + residual",
         "m8": "M8 HexConvGRU-K (K=4)", "oursS": "Ours-S (m8, 15.4k)", "noSI": "Ours-S noSI",
         "noEPE": "Ours-S noEPE", "K1": "Ours-S K1 (K_max=1)", "oursL": "Ours-L (m9m, 274k)"}


# ---------------------------------------------------------------- loading
def grid_names(grid):
    out = []
    p = Path(grid)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            out.append(line.split("\t")[0])
    return out


def arm_seed(name):
    """'m1_s0_f0.25' -> ('m1_f0.25', 0); 'm4_s1_f1.0' -> ('m4', 1); 'p2_oursS_s2' -> ('oursS', 2)."""
    if name.startswith("p2_"):
        _, arm, s = name.split("_")
        return arm, int(s[1:])
    m, s, f = name.split("_")
    return (m if float(f[1:]) == 1.0 else f"{m}_{f}"), int(s[1:])


def load(runs, grids, sub, keys=None):
    """-> {(arm, seed): metrics dict} for runs/<name>/<sub>/metrics.json."""
    R = {}
    for g in grids:
        for name in grid_names(g):
            p = Path(runs) / name / sub / "metrics.json"
            if p.exists():
                R[arm_seed(name)] = json.loads(p.read_text())
    return R


def load_floors(floors_dir):
    p = Path(floors_dir) / "spring" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- stats
def vals(R, arm, metric, seeds=SEEDS):
    return [R[(arm, s)].get(metric) if (arm, s) in R else None for s in seeds]


def complete(*lists):
    return all(v is not None for l in lists for v in l)


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), 0
    import numpy as _np  # numpy: NaN propagates (reported as nan, never silently dropped)
    a = _np.asarray(xs, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else float("nan")), len(a)


def n_less(a, b):
    return sum(x < y for x, y in zip(a, b))


def fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def raw(xs):
    return "[" + ", ".join("NA" if x is None else repr(float(x)) for x in xs) + "]"


def floor_values(F):
    """Best floor per metric: flow = min over zero/LK; depth = the constant floor."""
    if not F:
        return {}
    fl = F["floors"]
    flow = [k for k in fl if k.startswith("flow_")]
    dep = [k for k in fl if k.startswith("depth_")]
    out = {}
    if flow:
        bf = min(flow, key=lambda k: fl[k]["epe"])
        out["epe"] = (fl[bf]["epe"], bf)
        out["angular_deg"] = (min(fl[k]["angular_deg"] for k in flow), "min over flow floors")
    if dep:
        bd = dep[0]
        for m in ("depth_rmse_aligned", "depth_rmse", "depth_absrel"):
            out[m] = (fl[bd][m], bd)
    return out


# ---------------------------------------------------------------- decisions
def _level(ok):
    return "L1" if ok else "L2 (incomplete)"


def _dec(i, wording, status, ok, detail, rawtxt, sub=None):
    return dict(id=i, wording=wording, status=status, level=_level(ok), detail=detail, raw=rawtxt, sub=sub or [])


def decide_all(R, F, Rk4=None):
    """R: Spring last-ckpt metrics {(arm, seed): m}. F: Spring floors. Rk4: {(arm,seed): m} at K=4 (K1 sensitivity)."""
    D = []
    # ---- P1
    w = "Ours-L mean Spring flow EPE < every phase-1 arm's mean AND < the LK floor"
    L = vals(R, "oursL", "epe")
    ok = complete(L) and bool(F)
    rows, rawl = [], [f"oursL EPE {raw(L)}"]
    for a in P1_ARMS:
        v = vals(R, a, "epe")
        ok = ok and complete(v)
        rawl.append(f"{a} EPE {raw(v)}")
        rows.append((a, mean_sd(v)[0] if complete(v) else None))
    lk = F["floors"]["flow_lucas_kanade_hex"]["epe"] if F else None
    if ok:
        mL = statistics.fmean(L)
        beat = {a: mL < m for a, m in rows}
        st = "SUPPORTED" if (all(beat.values()) and mL < lk) else "NOT SUPPORTED"
        det = (f"Ours-L mean {mL:.4f}; LK floor {lk:.4f} ({'beaten' if mL < lk else 'NOT beaten'}); "
               + ", ".join(f"{a} {m:.4f} ({'beaten' if beat[a] else 'NOT beaten'})" for a, m in rows))
        sens = []
        for a in sorted({k[0] for k in R if k[0].endswith("_f0.25")}):
            v = vals(R, a, "epe")
            if complete(v):
                sens.append(f"{a} {statistics.fmean(v):.4f} ({'beaten' if mL < statistics.fmean(v) else 'NOT beaten'})")
        if sens:
            det += "; sensitivity f=0.25 arms: " + ", ".join(sens)
    else:
        st, det = "INCOMPLETE", "missing seeds / floors"
    D.append(_dec("P1", w, st, ok, det, " | ".join(rawl) + f" | LK floor {lk}"))
    # ---- P2
    w = "Ours-S < M4 on Spring flow EPE and on aligned depth RMSE, each in >= 2/3 paired seeds"
    ex, ey = vals(R, "oursS", "epe"), vals(R, "m4", "epe")
    dx, dy = vals(R, "oursS", "depth_rmse_aligned"), vals(R, "m4", "depth_rmse_aligned")
    ok = complete(ex, ey, dx, dy)
    if ok:
        we, wd = n_less(ex, ey), n_less(dx, dy)
        st = "SUPPORTED" if (we >= 2 and wd >= 2) else "NOT SUPPORTED"
        det = f"EPE seed wins {we}/3, aligned-depth seed wins {wd}/3"
    else:
        st, det = "INCOMPLETE", "missing seeds"
    D.append(_dec("P2", w, st, ok, det, f"oursS EPE {raw(ex)} M4 {raw(ey)}; oursS aligned {raw(dx)} M4 {raw(dy)}"))
    # ---- P3
    w = ("ablations hurt, each in >= 2/3 paired seeds: noSI worse aligned-depth RMSE than Ours-S; noEPE worse flow EPE; "
         "K1 worse flow EPE")
    subs, oks, rawl = [], [], []
    for abl, met, name in (("noSI", "depth_rmse_aligned", "noSI worse aligned depth"),
                           ("noEPE", "epe", "noEPE worse flow EPE"), ("K1", "epe", "K1 worse flow EPE")):
        base, ab = vals(R, "oursS", met), vals(R, abl, met)
        c = complete(base, ab)
        oks.append(c)
        rawl.append(f"{abl} {met} {raw(ab)} vs oursS {raw(base)}")
        if c:
            wins = n_less(base, ab)      # ablation strictly worse = ours strictly lower
            subs.append((name, "SUPPORTED" if wins >= 2 else "NOT SUPPORTED", f"{wins}/3 seeds worse"))
        else:
            subs.append((name, "INCOMPLETE", "missing seeds"))
    k4 = vals(Rk4 or {}, "K1", "epe")
    if complete(k4, vals(R, "oursS", "epe")):
        subs.append(("sensitivity: K1 evaluated at K=4 worse flow EPE", "(not a decision)",
                     f"{n_less(vals(R, 'oursS', 'epe'), k4)}/3 seeds worse; K1@K=4 {raw(k4)}"))
    ok = all(oks)
    if ok:
        st = "SUPPORTED" if all(s[1] == "SUPPORTED" for s in subs[:3]) else "NOT SUPPORTED"
    else:
        st = "INCOMPLETE"
    D.append(_dec("P3", w, st, ok, "; ".join(f"{n}: {s} ({d})" for n, s, d in subs), " | ".join(rawl), sub=subs))
    # ---- P4
    w = "(descriptive) Ours-S vs M1 (fly brain, same budget) on both tasks"
    ex, ey = vals(R, "oursS", "epe"), vals(R, "m1", "epe")
    dx, dy = vals(R, "oursS", "depth_rmse_aligned"), vals(R, "m1", "depth_rmse_aligned")
    ok = complete(ex, ey, dx, dy)
    if ok:
        st = "DESCRIPTIVE"
        det = (f"EPE mean Ours-S {statistics.fmean(ex):.4f} vs M1 {statistics.fmean(ey):.4f} (Ours-S lower in {n_less(ex, ey)}/3 seeds); "
               f"aligned depth mean Ours-S {statistics.fmean(dx):.4f} vs M1 {statistics.fmean(dy):.4f} (lower in {n_less(dx, dy)}/3)")
    else:
        st, det = "INCOMPLETE", "missing seeds"
    D.append(_dec("P4", w, st, ok, det, f"oursS EPE {raw(ex)} M1 {raw(ey)}; oursS aligned {raw(dx)} M1 {raw(dy)}"))
    return D


# ---------------------------------------------------------------- tables
def arm_list(R):
    arms = sorted({k[0] for k in R}, key=lambda a: (a.endswith("_f0.25"), a in P2_ARMS, a))
    return arms


def arm_rows(R, F, metrics=METRICS):
    fv = floor_values(F)
    rows = []
    for a in arm_list(R):
        row = dict(arm=a, n_seeds=0)
        for met in metrics:
            xs = vals(R, a, met)
            mu, sd, n = mean_sd(xs)
            row.update({f"{met}_mean": mu, f"{met}_sd": sd, f"{met}_seeds": " ".join("NA" if x is None else repr(float(x)) for x in xs)})
            row["n_seeds"] = max(row["n_seeds"], n)
            if met in fv and n:
                row[f"{met}_margin_vs_floor"] = fv[met][0] - mu
                row[f"{met}_margin_pct"] = 100 * (fv[met][0] - mu) / fv[met][0]
        rows.append(row)
    return rows, fv


def _table(P, rows, metrics, margins=False):
    if not margins:
        P("| arm | n | " + " | ".join(LABEL[m] for m in metrics) + " |\n|---|---|" + "---|" * len(metrics))
        for r in rows:
            P(f"| {NAMES.get(r['arm'], r['arm'])} | {r['n_seeds']} | " + " | ".join(
                f"{fmt(r[m + '_mean'])} +- {fmt(r[m + '_sd'])}" for m in metrics) + " |")
    else:
        P("| arm | " + " | ".join(f"{LABEL[m]} margin (floor - model; %)" for m in metrics) + " |\n|---|" + "---|" * len(metrics))
        for r in rows:
            P(f"| {NAMES.get(r['arm'], r['arm'])} | " + " | ".join(
                f"{fmt(r.get(m + '_margin_vs_floor'))} ({fmt(r.get(m + '_margin_pct'), 2)}%)" for m in metrics) + " |")


def build_report(R, Rbest, Rk, Rl2, Rl2ref, F):
    D = decide_all(R, F, Rk.get(4))
    rows, fv = arm_rows(R, F)
    L = []
    P = L.append
    P("# Results phase 2: Spring test (PLAN.md A9)\n")
    P("Generated by `analyze_p2.py`. Primary = last (30k) checkpoint of every run on the 8 pre-selected Spring sequences, "
      "evaluated once (**L1** when all seeds are present). mean +- sd over seeds (sample sd). Metrics are pooled over valid "
      "hexals and all frames; depth PRIMARY = log-depth RMSE (natural log) after per-clip median alignment "
      "(clip = Spring sequence x vertical split); raw standardised RMSE and AbsRel are secondary. Decisions use raw floats "
      "and strict inequalities, paired by seed.\n")
    P("## Decisions (pre-registered, A9)\n")
    P("| id | decision | claim level | pre-registered condition | detail |\n|---|---|---|---|---|")
    for d in D:
        P(f"| {d['id']} | **{d['status']}** | {d['level']} | {d['wording']} | {d['detail']} |")
    P("\nRaw per-seed values behind every decision:\n")
    for d in D:
        P(f"- {d['id']}: {d['raw']}")
    P("\n## Per-arm Spring results, last checkpoint (L1 if 3/3 seeds)\n")
    _table(P, rows, METRICS)
    P("\n## Floors on Spring and margins\n")
    if not F:
        P("floors_out/spring/metrics.json not found; margins omitted.\n")
    else:
        for k, v in F["floors"].items():
            P(f"- {k}: " + ", ".join(f"{a}={b:.4f}" for a, b in v.items() if isinstance(b, float)))
        P("\nFloors used: " + "; ".join(f"{LABEL[m]} = {fv[m][0]:.4f} ({fv[m][1]})" for m in METRICS if m in fv) + "\n")
        _table(P, rows, METRICS, margins=True)
        P("\nPositive margin = better than the floor. The constant-depth floor under per-clip alignment is the within-clip spread of the GT log-depth.\n")
    P("## Secondary: best.pt (val-selected) on Spring (L2)\n")
    rb, _ = arm_rows(Rbest, F)
    _table(P, rb, METRICS)
    P("\n## Any-time K curve (last ckpt; K = inner GRU steps at evaluation; L2)\n")
    for arm in ("m8", "oursS", "K1", "oursL"):
        P(f"\n**{NAMES[arm]}**\n")
        P("| K | " + " | ".join(LABEL[m] for m in ("epe", "depth_rmse_aligned")) + " |\n|---|---|---|")
        for k in (1, 2, 3, 4):
            cells = []
            for met in ("epe", "depth_rmse_aligned"):
                mu, sd, n = mean_sd(vals(Rk.get(k, {}), arm, met))
                cells.append(f"{fmt(mu)} +- {fmt(sd)} (n={n})")
            P(f"| {k} | " + " | ".join(cells) + " |")
    P("\n## Old Sintel test, second look (L2): phase-2 arms, last ckpt (`test_last/`)\n")
    P("The Sintel test was already spent in phase 1 (A1-A8); these numbers are a second look, not a confirmation. "
      "Reference rows are the phase-1 `test/` results (best.pt, val-selected), so they differ in checkpoint rule. "
      "No depth alignment on this split: raw standardised RMSE and AbsRel.\n")
    r2, _ = arm_rows(Rl2, None, L2_METRICS)
    r2ref, _ = arm_rows({k: v for k, v in Rl2ref.items() if k[0] in ("m1", "m4", "m8")}, None, L2_METRICS)
    for r in r2ref:
        r["arm"] = r["arm"]
    _table(P, r2 + r2ref, L2_METRICS)
    return "\n".join(L) + "\n", rows, D


def write_csv(rows, fh):
    cols = ["arm", "n_seeds"]
    for m in METRICS:
        cols += [f"{m}_mean", f"{m}_sd", f"{m}_seeds", f"{m}_margin_vs_floor", f"{m}_margin_pct"]
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: (repr(r[c]) if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(HERE / "runs"))
    p.add_argument("--grids", nargs="+", default=[str(HERE / "scripts" / "grid_v2.tsv"), str(HERE / "scripts" / "grid_p2.tsv")])
    p.add_argument("--floors", default=str(HERE / "floors_out"))
    p.add_argument("--out-dir", default=str(HERE))
    p.add_argument("--dry", action="store_true")
    a = p.parse_args(argv)
    R = load(a.runs, a.grids, "spring_last")
    Rbest = load(a.runs, a.grids, "spring_best")
    Rk = {k: load(a.runs, a.grids, f"spring_last_k{k}") for k in (1, 2, 3, 4)}
    Rl2 = load(a.runs, a.grids[1:], "test_last")
    Rl2ref = load(a.runs, a.grids[:1], "test")
    F = load_floors(a.floors)
    text, rows, D = build_report(R, Rbest, Rk, Rl2, Rl2ref, F)
    if a.dry:
        print(text)
        return D
    out = Path(a.out_dir)
    (out / "docs").mkdir(exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    (out / "docs" / "RESULTS_P2.md").write_text(text)
    with open(out / "results" / "summary_p2.csv", "w", newline="") as fh:
        write_csv(rows, fh)
    print("wrote docs/RESULTS_P2.md and results/summary_p2.csv; " + ", ".join(f"{d['id']}={d['status']}" for d in D))
    return D


if __name__ == "__main__":
    main()
