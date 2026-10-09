"""Pre-registered final analysis (PLAN.md H1-H7, amendments A5-A8). Reads metrics JSONs written by
eval.py / floors.py and writes docs/RESULTS.md + results/summary.csv. Pure stdlib (no torch).

    python analyze.py                  # split=test -> docs/RESULTS.md, results/summary.csv
    python analyze.py --split val      # debugging -> docs/RESULTS_val.md, results/summary_val.csv (all labelled L2/val)
    python analyze.py --dry            # print to stdout, write nothing

Decision rules (strict '<' on the raw floats; rounding is only for display, raw values are also printed):
  H1  M1 < M2 in 3/3 paired seeds on flow EPE AND 3/3 on depth RMSE            (f=1.0)
  H2  mean over seeds M1 < M4 on flow EPE AND on depth RMSE
  H3  descriptive: relative advantage of M1 over M2 and over M5, adv=(X-M1)/X (X = the other model, paired by seed,
      averaged over seeds), is larger at f=0.25 than at f=1.0 (flow EPE and depth RMSE; 4 comparisons)
  H4  M6 < M4 on flow EPE and on depth RMSE, each: mean over seeds AND >=2/3 paired seeds
  H5  M6 < M7 flow EPE in 3/3 paired seeds
  H6  M8 (K=4) < M4 flow EPE in >=2/3 paired seeds
  H7  M6f < M7f flow EPE (final = frozen front-end + residual) in 3/3 paired seeds
A decision needs all 3 seeds of every involved arm; otherwise it is INCOMPLETE and labelled L2.
Claim level: L1 = pre-registered + test + complete; L2 = pre-registered but val-only/incomplete.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ("epe", "angular_deg", "depth_rmse", "depth_absrel")
LABEL = {"epe": "flow EPE", "angular_deg": "angular err (deg)", "depth_rmse": "depth RMSE",
         "depth_absrel": "depth AbsRel"}
SEEDS = (0, 1, 2)
NAMES = {"m1": "M1 connectome", "m2": "M2 rewired", "m3": "M3 random ER", "m4": "M4 ConvGRU small",
         "m5": "M5 ConvGRU large", "m6": "M6 hybrid (real)", "m7": "M7 hybrid (rewired)",
         "m6f": "M6f frozen front-end + residual", "m7f": "M7f frozen front-end + residual",
         "m8": "M8 HexConvGRU-K (K=4)"}


# ---------------------------------------------------------------- loading
def parse_name(name):
    m, s, f = name.split("_")
    return m, float(f[1:]), int(s[1:])


def grid_names(grid):
    out = []
    for line in Path(grid).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            out.append(line.split("\t")[0])
    return out


def load_all(split, runs_dir, grid):
    """-> {(model, frac, seed, variant): metrics dict}; variant in '', 'k1'..'k4', 'frontend'."""
    res = {}
    for name in grid_names(grid):
        m, f, s = parse_name(name)
        for var in ("", "k1", "k2", "k3", "k4", "frontend"):
            p = Path(runs_dir) / name / (f"{split}_{var}" if var else split) / "metrics.json"
            if p.exists():
                res[(m, f, s, var)] = json.loads(p.read_text())
    return res


def load_floors(split, floors_dir):
    p = Path(floors_dir) / split / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- statistics
def vals(R, model, frac, metric, var="", seeds=SEEDS):
    """Per-seed values (None where missing)."""
    return [R[(model, frac, s, var)][metric] if (model, frac, s, var) in R else None for s in seeds]


def complete(*lists):
    return all(v is not None for l in lists for v in l)


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), 0
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else float("nan")), len(xs)


def n_wins(a, b):
    """# paired seeds where a < b (strict)."""
    return sum(x < y for x, y in zip(a, b))


def fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def raw(xs):
    return "[" + ", ".join("NA" if x is None else repr(float(x)) for x in xs) + "]"


def floor_values(F):
    """best non-oracle floors: flow EPE (+ that floor's angular), depth RMSE / AbsRel (each its own best)."""
    if not F:
        return {}
    fl = F["floors"]
    flow = [k for k in fl if k.startswith("flow_")]
    bf = min(flow, key=lambda k: fl[k]["epe"])
    dep = [k for k in fl if k.startswith("depth_")]
    bd = min(dep, key=lambda k: fl[k]["depth_rmse"])
    out = {"epe": (fl[bf]["epe"], bf), "angular_deg": (min(fl[k]["angular_deg"] for k in flow), "min over non-oracle flow floors"),
           "depth_rmse": (fl[bd]["depth_rmse"], bd),
           "depth_absrel": (min(fl[k]["depth_absrel"] for k in dep), "min over depth floors")}
    return out


# ---------------------------------------------------------------- hypotheses
def _dec(hid, wording, status, level, detail, rawtxt):
    return dict(id=hid, wording=wording, status=status, level=level, detail=detail, raw=rawtxt)


def _level(split, ok):
    return "L1" if (split == "test" and ok) else ("L2 (val-only, debug)" if split != "test" else "L2 (incomplete)")


def paired_cmp(R, a, b, metric, fa=1.0, fb=1.0, va="", vb=""):
    x, y = vals(R, a, fa, metric, va), vals(R, b, fb, metric, vb)
    return x, y, complete(x, y)


def decide_all(R, split):
    D = []
    # H1
    w = "M1 test EPE < M2 test EPE in 3/3 paired seeds, and same for depth RMSE"
    ex, ey, c1 = paired_cmp(R, "m1", "m2", "epe")
    dx, dy, c2 = paired_cmp(R, "m1", "m2", "depth_rmse")
    ok = c1 and c2
    if ok:
        we, wd = n_wins(ex, ey), n_wins(dx, dy)
        st = "SUPPORTED" if (we == 3 and wd == 3) else "NOT SUPPORTED"
        det = f"EPE wins {we}/3, depth RMSE wins {wd}/3"
    else:
        st, det = "INCOMPLETE", "missing seeds"
    D.append(_dec("H1", w, st, _level(split, ok), det,
                  f"M1 EPE {raw(ex)} vs M2 {raw(ey)}; M1 RMSE {raw(dx)} vs M2 {raw(dy)}"))
    # H2
    w = "M1 beats M4 on both tasks (mean over seeds)"
    ex, ey, c1 = paired_cmp(R, "m1", "m4", "epe")
    dx, dy, c2 = paired_cmp(R, "m1", "m4", "depth_rmse")
    ok = c1 and c2
    if ok:
        me, md = statistics.fmean(ex), statistics.fmean(dx)
        me4, md4 = statistics.fmean(ey), statistics.fmean(dy)
        st = "SUPPORTED" if (me < me4 and md < md4) else "NOT SUPPORTED"
        det = f"mean EPE M1 {me:.4f} vs M4 {me4:.4f}; mean RMSE M1 {md:.4f} vs M4 {md4:.4f}"
    else:
        st, det = "INCOMPLETE", "missing seeds"
    D.append(_dec("H2", w, st, _level(split, ok), det,
                  f"M1 EPE {raw(ex)} M4 {raw(ey)}; M1 RMSE {raw(dx)} M4 {raw(dy)}"))
    # H3
    w = ("M1-M2 and M1-M5 gaps (relative) at 25% data are larger than at 100% data "
         "(descriptive; reported either way)")
    rows, okall, nlarger, rawl = [], True, 0, []
    for other in ("m2", "m5"):
        for met in ("epe", "depth_rmse"):
            adv = {}
            for fr in (1.0, 0.25):
                a, b = vals(R, "m1", fr, met), vals(R, other, fr, met)
                if not complete(a, b):
                    okall = False
                    adv[fr] = None
                else:
                    adv[fr] = statistics.fmean((y - x) / y for x, y in zip(a, b))
                rawl.append(f"{other}/{met}/f{fr:g}: M1 {raw(a)} vs {raw(b)}")
            larger = adv[0.25] is not None and adv[1.0] is not None and adv[0.25] > adv[1.0]
            nlarger += bool(larger)
            rows.append((other, met, adv[1.0], adv[0.25], larger))
    if okall:
        st = ("SUPPORTED (descriptive)" if nlarger == 4 else
              "NOT SUPPORTED" if nlarger == 0 else f"PARTIAL ({nlarger}/4 comparisons)")
    else:
        st = "INCOMPLETE"
    det = "; ".join(f"M1 vs {o.upper()} {m}: adv f1.0={fmt(a1)}, f0.25={fmt(a2)}, larger@0.25={l}"
                    for o, m, a1, a2, l in rows)
    D.append(_dec("H3", w, st, _level(split, okall), det, " | ".join(rawl)))
    D[-1]["rows"] = rows
    # H4
    w = ("M6 beats M4 on both val-selected test metrics (flow EPE, depth RMSE), "
         "mean over 3 seeds and >= 2/3 paired seeds")
    ex, ey, c1 = paired_cmp(R, "m6", "m4", "epe")
    dx, dy, c2 = paired_cmp(R, "m6", "m4", "depth_rmse")
    ok = c1 and c2
    if ok:
        mo = statistics.fmean(ex) < statistics.fmean(ey) and statistics.fmean(dx) < statistics.fmean(dy)
        we, wd = n_wins(ex, ey), n_wins(dx, dy)
        st = "SUPPORTED" if (mo and we >= 2 and wd >= 2) else "NOT SUPPORTED"
        det = (f"mean EPE {statistics.fmean(ex):.4f} vs {statistics.fmean(ey):.4f}, mean RMSE "
               f"{statistics.fmean(dx):.4f} vs {statistics.fmean(dy):.4f}; seed wins EPE {we}/3, RMSE {wd}/3")
    else:
        st, det = "INCOMPLETE", "missing seeds"
    D.append(_dec("H4", w, st, _level(split, ok), det,
                  f"M6 EPE {raw(ex)} M4 {raw(ey)}; M6 RMSE {raw(dx)} M4 {raw(dy)}"))
    h4 = st
    # H5
    w = "M6 beats M7 on flow EPE in 3/3 paired seeds (connectome-specific benefit)"
    ex, ey, ok = paired_cmp(R, "m6", "m7", "epe")
    st = "INCOMPLETE" if not ok else ("SUPPORTED" if n_wins(ex, ey) == 3 else "NOT SUPPORTED")
    det = "missing seeds" if not ok else f"EPE wins {n_wins(ex, ey)}/3"
    if ok and h4 == "SUPPORTED" and st == "NOT SUPPORTED":
        det += "; H4 holds but H5 fails: gain comes from the architecture, not the wiring"
    D.append(_dec("H5", w, st, _level(split, ok), det, f"M6 EPE {raw(ex)} vs M7 {raw(ey)}"))
    # H6
    w = "M8 (K=4) beats M4 on test flow EPE in >= 2/3 paired seeds"
    var = "k4" if all(("m8", 1.0, s, "k4") in R for s in SEEDS) else ""
    ex, ey, ok = paired_cmp(R, "m8", "m4", "epe", va=var)
    st = "INCOMPLETE" if not ok else ("SUPPORTED" if n_wins(ex, ey) >= 2 else "NOT SUPPORTED")
    det = "missing seeds" if not ok else f"EPE wins {n_wins(ex, ey)}/3 (K=4 from '{split}{'_' + var if var else ''}')"
    D.append(_dec("H6", w, st, _level(split, ok), det, f"M8(K=4) EPE {raw(ex)} vs M4 {raw(ey)}"))
    # H7
    w = "M6f beats M7f on test flow EPE in 3/3 paired seeds"
    ex, ey, ok = paired_cmp(R, "m6f", "m7f", "epe")
    st = "INCOMPLETE" if not ok else ("SUPPORTED" if n_wins(ex, ey) == 3 else "NOT SUPPORTED")
    det = "missing seeds" if not ok else f"EPE wins {n_wins(ex, ey)}/3"
    D.append(_dec("H7", w, st, _level(split, ok), det, f"M6f EPE {raw(ex)} vs M7f {raw(ey)}"))
    return D


# ---------------------------------------------------------------- tables
def arm_rows(R, F):
    fv = floor_values(F)
    rows = []
    arms = sorted({(m, f, v) for (m, f, s, v) in R if v in ("", "frontend")}, key=lambda t: (t[1] != 1.0, t[0], t[1], t[2]))
    arms += sorted({(m, f, v) for (m, f, s, v) in R if v.startswith("k")})
    for m, f, v in arms:
        row = dict(arm=m + (f"[{v}]" if v else ""), frac=f, n_seeds=0)
        for met in METRICS:
            xs = vals(R, m, f, met, v)
            mu, sd, n = mean_sd(xs)
            row.update({f"{met}_mean": mu, f"{met}_sd": sd, f"{met}_seeds": " ".join(
                "NA" if x is None else repr(float(x)) for x in xs)})
            row["n_seeds"] = n
            if met in fv:
                row[f"{met}_margin_vs_floor"] = fv[met][0] - mu       # >0: better than floor
                row[f"{met}_margin_pct"] = 100 * (fv[met][0] - mu) / fv[met][0]
        rows.append(row)
    return rows, fv


def build_report(R, F, split, runs_dir):
    D = decide_all(R, split)
    rows, fv = arm_rows(R, F)
    L = []
    P = L.append
    P(f"# Results ({split} split)\n")
    if split != "test":
        P("> **DEBUG OUTPUT on the val split. Not a result. Every claim below is capped at L2.**\n")
    P("Generated by `analyze.py` from `runs/<name>/%s*/metrics.json` and `floors_out/%s/metrics.json`. " % (split, split)
      + "mean +- sd over seeds (sample sd, n-1); metrics pooled over all pixels and frames of the split's sequences. "
        "Decisions use raw floats; tables are rounded to 4 decimals for display only.\n")
    P("## Hypothesis decisions (pre-registered, PLAN.md)\n")
    P("| id | decision | claim level | pre-registered condition | detail |\n|---|---|---|---|---|")
    for d in D:
        P(f"| {d['id']} | **{d['status']}** | {d['level']} | {d['wording']} | {d['detail']} |")
    P("\nRaw per-seed values behind every decision:\n")
    for d in D:
        P(f"- {d['id']}: {d['raw']}")
    h3 = next(d for d in D if d["id"] == "H3")
    P("\n### H3 data-efficiency table (relative advantage of M1 over X: (X - M1)/X, mean over paired seeds; positive = M1 better)\n")
    P("| comparison | metric | adv @ f=1.0 | adv @ f=0.25 | larger @ 0.25? |\n|---|---|---|---|---|")
    for o, m, a1, a2, l in h3["rows"]:
        P(f"| M1 vs {o.upper()} | {LABEL[m]} | {fmt(a1, 6)} | {fmt(a2, 6)} | {l} |")
    P("\n### Absolute values for the data-efficiency plot (mean +- sd)\n")
    P("| arm | f | flow EPE | depth RMSE |\n|---|---|---|---|")
    for m in ("m1", "m2", "m5"):
        for f in (1.0, 0.25):
            cells = []
            for met in ("epe", "depth_rmse"):
                mu, sd, n = mean_sd(vals(R, m, f, met))
                cells.append(f"{fmt(mu)} +- {fmt(sd)} (n={n})")
            P(f"| {m.upper()} | {f:g} | {cells[0]} | {cells[1]} |")
    P("\n## Per-arm results (mean +- sd over seeds)\n")
    P("| arm | f | n | " + " | ".join(LABEL[m] for m in METRICS) + " |\n|---|---|---|" + "---|" * len(METRICS))
    for r in rows:
        P(f"| {r['arm']} | {r['frac']:g} | {r['n_seeds']} | " + " | ".join(
            f"{fmt(r[m + '_mean'])} +- {fmt(r[m + '_sd'])}" for m in METRICS) + " |")
    P("\nVariants: `[frontend]` = frozen front-end only (no residual trunk); `[kN]` = M8 with N inner GRU steps.\n")
    P("## Floors and margin over the best non-oracle floor\n")
    if not F:
        P(f"floors_out/{split}/metrics.json not found; margins omitted.\n")
    else:
        for k, v in F["floors"].items():
            P(f"- {k}: " + ", ".join(f"{a}={b:.4f}" for a, b in v.items() if isinstance(b, float))
              + ("  (ORACLE, not a competitor)" if "ORACLE" in v else ""))
        P("\nBest non-oracle floors used: " + "; ".join(f"{LABEL[m]} = {fv[m][0]:.4f} ({fv[m][1]})" for m in METRICS) + "\n")
        P("| arm | f | " + " | ".join(f"{LABEL[m]} margin (floor - model; %)" for m in METRICS) + " |\n|---|---|" + "---|" * len(METRICS))
        for r in rows:
            if r["arm"].count("[") and "frontend" not in r["arm"]:
                continue
            P(f"| {r['arm']} | {r['frac']:g} | " + " | ".join(
                f"{fmt(r.get(m + '_margin_vs_floor'))} ({fmt(r.get(m + '_margin_pct'), 2)}%)" for m in METRICS) + " |")
        P("\nPositive margin = better than the floor. A negative margin means the model does not beat the classical/no-learning floor; reported as is.\n")
    P("## M8 any-time curve (K inner GRU steps at evaluation; trained with K ~ U{1..4}; primary K=4) -- L3 curve, H6 uses K=4\n")
    P("| K | " + " | ".join(LABEL[m] for m in METRICS) + " |\n|---|" + "---|" * len(METRICS))
    for k in (1, 2, 3, 4):
        cells = []
        for met in METRICS:
            mu, sd, n = mean_sd(vals(R, "m8", 1.0, met, f"k{k}"))
            cells.append(f"{fmt(mu)} +- {fmt(sd)} (n={n})")
        P(f"| {k} | " + " | ".join(cells) + " |")
    cells = []
    for met in METRICS:
        mu, sd, n = mean_sd(vals(R, "m4", 1.0, met))
        cells.append(f"{fmt(mu)} +- {fmt(sd)} (n={n})")
    P("| M4 ref | " + " | ".join(cells) + " |")
    P("\n## M6f / M7f: front-end only vs final (frozen front-end + residual) -- A6.2 pre-registered reporting\n")
    P("| arm | variant | " + " | ".join(LABEL[m] for m in METRICS) + " |\n|---|---|" + "---|" * len(METRICS))
    for m in ("m6f", "m7f"):
        for var, lab in (("frontend", "front-end only"), ("", "final")):
            cells = []
            for met in METRICS:
                mu, sd, n = mean_sd(vals(R, m, 1.0, met, var))
                cells.append(f"{fmt(mu)} +- {fmt(sd)} (n={n})")
            P(f"| {m.upper()} | {lab} | " + " | ".join(cells) + " |")
    P("\nPer-seed final - front-end-only (negative = residual helps): ")
    for m in ("m6f", "m7f"):
        for met in ("epe", "depth_rmse"):
            a, b = vals(R, m, 1.0, met, ""), vals(R, m, 1.0, met, "frontend")
            d = [None if x is None or y is None else x - y for x, y in zip(a, b)]
            P(f"- {m} {met}: {raw(d)}")
    P("\nSanity: front-end-only M6f/M7f should equal the M1/M2 predictions of the same seed (checked on val for seed 0).\n")
    return "\n".join(L) + "\n", rows, D


def write_csv(rows, fh):
    cols = ["arm", "frac", "n_seeds"]
    for m in METRICS:
        cols += [f"{m}_mean", f"{m}_sd", f"{m}_seeds", f"{m}_margin_vs_floor", f"{m}_margin_pct"]
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: (repr(r[c]) if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--runs", default=str(HERE / "runs"))
    p.add_argument("--grid", default=str(HERE / "scripts" / "grid_v2.tsv"))
    p.add_argument("--floors", default=str(HERE / "floors_out"))
    p.add_argument("--out-dir", default=str(HERE))
    p.add_argument("--dry", action="store_true", help="print the report, write nothing")
    a = p.parse_args(argv)
    R = load_all(a.split, a.runs, a.grid)
    F = load_floors(a.split, a.floors)
    text, rows, D = build_report(R, F, a.split, a.runs)
    if a.dry:
        print(text)
        buf = io.StringIO()
        write_csv(rows, buf)
        print("--- summary.csv (preview) ---\n" + buf.getvalue()[:1500])
        return D
    suf = "" if a.split == "test" else "_val"
    out = Path(a.out_dir)
    (out / "docs").mkdir(exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    (out / "docs" / f"RESULTS{suf}.md").write_text(text)
    with open(out / "results" / f"summary{suf}.csv", "w", newline="") as fh:
        write_csv(rows, fh)
    print(f"wrote docs/RESULTS{suf}.md and results/summary{suf}.csv; " + ", ".join(f"{d['id']}={d['status']}" for d in D))
    return D


if __name__ == "__main__":
    main()
