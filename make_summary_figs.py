"""Final summary figures -> docs/figs/final_*.png (150 dpi).
Reads ONLY results/summary.csv and results/summary_p2.csv (already-produced numbers); runs no model.
Floors are copied from docs/RESULTS.md and docs/RESULTS_P2.md. Colour map: docs/NARRATIVE.md."""
import csv
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parent
OUT = ROOT / "docs" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 150
AMBER, CORAL, TEAL, VIOLET, GREY = "#E3A33B", "#E07A5F", "#3BA7A0", "#6A5ACD", "#8A8F98"
VIOLET_L = "#A79BE3"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

def rows(f):
    out = {}
    for r in csv.DictReader(open(ROOT / "results" / f)):
        out[(r["arm"], r.get("frac", "1.0"))] = r
    return out
P1, P2 = rows("summary.csv"), rows("summary_p2.csv")
def seeds(r, k): return [float(x) for x in r[k].split()]

# (key, label, colour, hollow)
S_ARMS = [("m1", "M1 (connectome)", AMBER), ("m6", "M6 (connectome hybrid)", AMBER), ("m6f", "M6f (connectome frozen+res)", AMBER),
          ("m2", "M2 (null)", CORAL), ("m3", "M3 (null, random)", CORAL), ("m7", "M7 (null hybrid)", CORAL), ("m7f", "M7f (null frozen+res)", CORAL),
          ("m4", "M4 (ConvGRU small)", TEAL), ("m5", "M5 (ConvGRU large)", TEAL), ("m8", "M8 (HexConvGRU-K)", TEAL)]
P_ARMS = S_ARMS + [("oursS", "Ours-S", VIOLET), ("oursL", "Ours-L", VIOLET),
                   ("noEPE", "Ours-S noEPE (ablation)", VIOLET_L), ("K1", "Ours-S K1 (ablation)", VIOLET_L), ("noSI", "Ours-S noSI (ablation)", VIOLET_L)]

def legend_groups(ax, floors=True, loc="lower right"):
    h = [Line2D([], [], marker="o", ls="", color=c, label=n) for n, c in
         [("connectome (real wiring)", AMBER), ("null / rewired", CORAL), ("deep learning baseline", TEAL), ("our method / ablations", VIOLET)]]
    h.append(Line2D([], [], marker="D", ls="", color="k", mfc="none", label="mean of 3 seeds"))
    ax.legend(handles=h, loc=loc, fontsize=7.5, frameon=False)

def dots(ax, data, arms, key, floors, xlabel):
    n = len(arms)
    for i, (k, lab, c) in enumerate(arms):
        y = n - 1 - i
        v = seeds(data[(k, "1.0")], key)
        ax.scatter(v, [y + d for d in (-0.12, 0, 0.12)][:len(v)], s=22, color=c, alpha=.85, zorder=3, edgecolor="none")
        ax.scatter([sum(v) / len(v)], [y], marker="D", s=46, facecolor="none", edgecolor="k", lw=1.1, zorder=4)
    ax.set_yticks(range(n)); ax.set_yticklabels([a[1] for a in arms][::-1])
    for name, val, ls in floors:
        ax.axvline(val, color=GREY, ls=ls, lw=1.3, zorder=1)
    ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.25); ax.set_ylim(-.7, n - .3)

# (a) Sintel test flow --------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
dots(ax, P1, S_ARMS, "epe_seeds", [("zero", 5.620, "-"), ("lk", 5.337, "--")], "flow EPE on Sintel test (hexal/frame; lower is better)")
ax.text(5.625, len(S_ARMS) - .45, "zero-flow 5.620", color=GREY, fontsize=7.5, ha="left", va="bottom")
ax.text(5.337, len(S_ARMS) - .45, "LK floor 5.337", color=GREY, fontsize=7.5, ha="right", va="bottom")
ax.set_title("Sintel test flow EPE per arm, 3 seeds + mean (L1): only M5 beats LK", fontsize=10)
legend_groups(ax, loc="upper left"); ax.set_xlim(5.05, 5.72); fig.tight_layout(); fig.savefig(OUT / "final_scoreboard_sintel.png", dpi=DPI); plt.close(fig)

# (b) Spring flow, broken axis -------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, sharey=True, figsize=(8.6, 5.6), gridspec_kw={"width_ratios": [6, 1], "wspace": .04})
n = len(P_ARMS)
for ax in (a1, a2):
    for i, (k, lab, c) in enumerate(P_ARMS):
        y = n - 1 - i
        v = seeds(P2[(k, "1.0")], "epe_seeds")
        ax.scatter(v, [y - .12, y, y + .12], s=22, color=c, alpha=.85, zorder=3, edgecolor="none")
        if k != "m1":
            ax.scatter([sum(v) / 3], [y], marker="D", s=46, facecolor="none", edgecolor="k", lw=1.1, zorder=4)
    ax.grid(axis="x", alpha=.25)
a1.set_xlim(0.4, 3.0); a2.set_xlim(252, 257); a2.set_xticks([254.5])
a1.axvline(0.4997, color=GREY, lw=1.3); a1.axvline(0.4655, color=GREY, ls="--", lw=1.3)
a1.set_yticks(range(n)); a1.set_yticklabels([a[1] for a in P_ARMS][::-1]); a1.set_ylim(-.7, n - .3)
a2.spines["left"].set_visible(False); a2.tick_params(left=False)
a1.spines["right"].set_visible(False)
d = .012
a1.plot([1 - d, 1 + d], [-d * 6, d * 6], transform=a1.transAxes, color="k", clip_on=False, lw=1)
a2.plot([-d * 6, d * 6], [-d * 6, d * 6], transform=a2.transAxes, color="k", clip_on=False, lw=1)
yM1 = n - 1
a2.annotate("M1 seed index 1:\nEPE 254.46\n(diverged; kept,\nnot dropped)", xy=(254.46, yM1), xytext=(252.3, yM1 - 3.2), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="k"), ha="left")
a1.text(0.99, 0.985, "M1 mean 85.24 is off-scale\n(driven by that one seed); M1's\nother seeds: 0.527, 0.718", transform=a1.transAxes,
        ha="right", va="top", fontsize=7.5, bbox=dict(fc="w", ec=GREY, lw=.6))
a1.text(0.4997, n - .35, "zero 0.4997", color=GREY, fontsize=7.5, ha="left", va="bottom")
a1.text(0.4655, n - .35, "LK 0.4655", color=GREY, fontsize=7.5, ha="right", va="bottom")
fig.suptitle("Spring flow EPE, last checkpoint, 3 seeds + mean (L1): no model beats zero-flow/LK; broken x axis", fontsize=10)
fig.supxlabel("flow EPE on Spring test (hexal/frame; lower is better); left panel 0.4-3.0, right panel 252-257", fontsize=8.5)
legend_groups(a1, loc="lower right"); fig.subplots_adjust(left=.27, right=.98, top=.93, bottom=.09)
fig.savefig(OUT / "final_scoreboard_spring.png", dpi=DPI); plt.close(fig)

# (c) depth ----------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"wspace": .6}, sharey=False)
dots(a, P1, S_ARMS, "depth_rmse_seeds", [("f", 0.7657, "--")], "depth RMSE, Sintel test (standardised log-depth; lower is better)")
a.text(0.7657, len(S_ARMS) - .45, "train-mean per-hexal floor 0.766", color=GREY, fontsize=7.5, ha="left", va="bottom")
a.set_title("Sintel test depth RMSE (L1): 0/10 arms below floor", fontsize=9.5)
dots(b, P2, P_ARMS, "depth_rmse_aligned_seeds", [("f", 0.9362, "--")], "aligned log-depth RMSE, Spring (nat-log, per-clip median; lower is better)")
b.set_yticklabels([a_[1] for a_ in P_ARMS][::-1])
b.text(0.9362, len(P_ARMS) - .45, "constant-depth floor 0.936", color=GREY, fontsize=7.5, ha="left", va="bottom")
b.set_title("Spring aligned depth RMSE, last ckpt (L1): best margin only 3.8%", fontsize=9.5)
legend_groups(b, loc="lower right")
fig.suptitle("Depth versus floors: dashed grey line = best non-learning floor", fontsize=10.5)
fig.subplots_adjust(left=.14,right=.99,top=.88,bottom=.13); fig.savefig(OUT / "final_depth.png", dpi=DPI); plt.close(fig)

# (d) hypothesis board -----------------------------------------------------
H = [("H1", "M1 beats M2 (real vs rewired wiring), 3/3 seeds, both tasks", "NOT SUPPORTED", "L1", "EPE wins 1/3, depth wins 1/3"),
     ("H2", "M1 beats M4 (param-matched ConvGRU) on both tasks", "NOT SUPPORTED", "L1", "EPE 5.590 vs 5.411; RMSE 0.894 vs 0.976"),
     ("H3", "M1 advantage larger at 25% data than 100%", "PARTIAL", "L1", "2 of 4 comparisons (M1-M5 yes, M1-M2 no)"),
     ("H4", "M6 hybrid beats M4 on both tasks", "NOT SUPPORTED", "L1", "EPE wins 0/3, RMSE wins 3/3"),
     ("H5", "M6 beats M7 on EPE, 3/3 seeds (wiring-specific gain)", "NOT SUPPORTED", "L1", "EPE wins 0/3"),
     ("H6", "M8 (K=4) beats M4 on EPE, >=2/3 seeds", "NOT SUPPORTED", "L1", "EPE wins 0/3"),
     ("H7", "M6f beats M7f on EPE, 3/3 seeds", "NOT SUPPORTED", "L1", "EPE wins 1/3"),
     ("P1", "Ours-L beats every phase-1 arm and LK on Spring EPE", "NOT SUPPORTED", "L1", "Ours-L 0.809 vs LK 0.466; LK + 5 of 10 phase-1 arms not beaten"),
     ("P2", "Ours-S beats M4 on Spring EPE and aligned depth, >=2/3 seeds", "NOT SUPPORTED", "L1", "EPE 3/3 seeds yes; depth 1/3 seeds"),
     ("P3", "Ablations (noSI, noEPE, K1) each hurt, >=2/3 seeds", "NOT SUPPORTED", "L1", "noEPE 2/3, K1 2/3 yes; noSI 1/3 no"),
     ("P4", "Ours-S vs M1 (same budget), descriptive", "DESCRIPTIVE", "L1", "EPE 0.590 vs 85.2; depth 0.939 vs 1.183")]
COL = {"NOT SUPPORTED": ("#C9CDD4", "#2B2F36"), "PARTIAL": ("#F3D9A0", "#5A3E00"), "DESCRIPTIVE": ("#DCD7F5", "#2E2670")}
fig, ax = plt.subplots(figsize=(10, 5.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, len(H) + 1.2)
for x, t in [(1, "id"), (7, "pre-registered claim"), (52, "decision"), (68, "level"), (75, "numbers (Sintel test for H, Spring for P)")]:
    ax.text(x, len(H) + .55, t, fontweight="bold", fontsize=9, va="center")
for i, (h, claim, dec, lvl, num) in enumerate(H):
    y = len(H) - i - .35
    if h == "P1": ax.plot([0, 100], [y + .6] * 2, color=GREY, lw=.8, ls=":")
    ax.text(1, y, h, fontweight="bold", fontsize=9.5, va="center", color=VIOLET if h[0] == "P" else "k")
    ax.text(7, y, textwrap.fill(claim, 52), fontsize=8.2, va="center", linespacing=1.05)
    bg, fg = COL[dec]
    ax.add_patch(Rectangle((51.5, y - .3), 15, .6, fc=bg, ec="none"))
    ax.text(59, y, dec, fontsize=8, va="center", ha="center", color=fg, fontweight="bold")
    ax.text(70.5, y, lvl, fontsize=8.5, va="center", ha="center")
    ax.text(75, y, num, fontsize=7.8, va="center")
ax.set_title("Hypothesis board: 0 of 11 pre-registered claims fully supported (H3 partial, P4 descriptive)", fontsize=10.5, loc="left")
fig.tight_layout(); fig.savefig(OUT / "final_hypotheses.png", dpi=DPI); plt.close(fig)

# (e) speed ----------------------------------------------------------------
S = [("M1 / flyvis rollout\nfused Triton (fastfly)", 0.202, 0.074, "2.7x", AMBER),
     ("hybrid M6/M7 trunk\ncompile (A8)", 0.188, 0.069, "2.7x", AMBER),
     ("M4 ConvGRU\ncompile reduce-overhead (R2)", 0.138, 0.027, "5.0x", TEAL),
     ("M4 ConvGRU (phase 2)\ncompile cells", 0.131, 0.042, "3.1x", TEAL),
     ("M8 K=4 (phase 2)\ncompile cells + fast-gather", 0.339, 0.165, "2.1x", TEAL)]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for i, (lab, b, a_, sp, c) in enumerate(S):
    y = len(S) - 1 - i
    ax.barh(y + .19, b, .36, color=c, alpha=.35, hatch="//", edgecolor=c)
    ax.barh(y - .19, a_, .36, color=c)
    ax.text(b + .004, y + .19, f"{b:.3f}", va="center", fontsize=8)
    ax.text(a_ + .004, y - .19, f"{a_:.3f}  ({sp} faster)", va="center", fontsize=8, fontweight="bold")
ax.set_yticks(range(len(S))); ax.set_yticklabels([s[0] for s in S][::-1], fontsize=8)
ax.set_xlabel("training time per iteration (s/iter, batch 4, RTX 4060 8 GB; lower is better)"); ax.set_xlim(0, .4)
ax.legend(handles=[Rectangle((0, 0), 1, 1, fc="w", ec="k", hatch="//", label="before"), Rectangle((0, 0), 1, 1, fc="k", label="after")], frameon=False, loc="lower right")
ax.set_title("Engineering: s/iter before vs after each step (hatched = before; colour = group)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "final_speed.png", dpi=DPI); plt.close(fig)

# (f) ablation on Spring -----------------------------------------------------
V = [("oursS", "Ours-S", VIOLET), ("noEPE", "noEPE", VIOLET_L), ("K1", "K1", VIOLET_L), ("noSI", "noSI", VIOLET_L), ("m4", "M4\n(ConvGRU)", TEAL)]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, key, yl, floors, ttl in [(axs[0], "epe_seeds", "flow EPE, Spring (lower is better)", [(0.4997, "zero-flow 0.4997", "-"), (0.4655, "LK 0.4655", "--")], "Flow EPE"),
                                 (axs[1], "depth_rmse_aligned_seeds", "aligned log-depth RMSE, Spring (lower is better)", [(0.9362, "constant-depth 0.936", "--")], "Aligned depth RMSE")]:
    vals = {k: seeds(P2[(k, "1.0")], key) for k, _, _ in V}
    for s in range(3):
        ax.plot(range(len(V)), [vals[k][s] for k, _, _ in V], color=GREY, lw=.8, alpha=.7, zorder=1)
    for j, (k, lab, c) in enumerate(V):
        ax.scatter([j] * 3, vals[k], color=c, s=34, zorder=3, edgecolor="k", lw=.4)
        m = sum(vals[k]) / 3
        ax.hlines(m, j - .25, j + .25, color="k", lw=1.5, zorder=4)
    for v, t, ls in floors:
        ax.axhline(v, color=GREY, ls=ls, lw=1.2); ax.text(len(V) - .45, v, t, color=GREY, fontsize=7.5, va="bottom", ha="right")
    ax.set_xticks(range(len(V))); ax.set_xticklabels([v[1] for v in V]); ax.set_ylabel(yl); ax.set_title(ttl + " (L1; grey lines join the same seed)", fontsize=9.5)
    ax.grid(axis="y", alpha=.25)
axs[0].set_ylim(0.43, 0.76)
fig.suptitle("Ablations on Spring (last ckpt, paired by seed): EPE loss and recurrence help flow in 2/3 seeds; scale-invariant loss does not help depth", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "final_ablation.png", dpi=DPI); plt.close(fig)
print("ok")
