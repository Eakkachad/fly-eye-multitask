#!/usr/bin/env python
"""Pre-registered error analysis (docs/EVALUATION.md) from ALREADY-SAVED outputs only.

Reads runs/<name>/test*/{metrics.json,per_pixel.npz,examples.npz}, runs/<name>/val_log.jsonl,
floors_out/*/metrics.json, depth_norm.json and the val per_pixel files that exist.
Never evaluates a model, never touches test inputs, never imports torch.
Writes figures to docs/figs/, tables to results/error_analysis/ and docs/ERROR_ANALYSIS.md.

Claim level: everything here is L3 (exploratory) unless marked as a pre-registered metric.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
FIGS = ROOT / "docs" / "figs"
TABS = ROOT / "results" / "error_analysis"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

# colours: connectome arms amber, deep learning teal, null/rewired coral, floors grey
VALC = "#4A4E69"  # val = dark slate (distinct from arm colours)
AMBER, TEAL, CORAL, GREY = "#E3A33B", "#3BA7A0", "#E07A5F", "#8A8A8A"
# ONE mapping used by every figure: group -> colour, arm -> group (coordinator-specified)
GROUP_COLOR = {"connectome": AMBER, "null/rewired": CORAL, "deep learning": TEAL}
ARM_GROUP = {"m1": "connectome", "m6": "connectome", "m6f": "connectome",
             "m2": "null/rewired", "m3": "null/rewired", "m7": "null/rewired", "m7f": "null/rewired",
             "m4": "deep learning", "m5": "deep learning", "m8": "deep learning"}
ARM_LS = {"m1": "-", "m6": ":", "m2": "-", "m7": "--", "m4": "-", "m5": ":", "m8": "--"}
ARM_COLOR = {a: GROUP_COLOR[g] for a, g in ARM_GROUP.items()}
ARM_NAME = {"m1": "M1", "m2": "M2", "m3": "M3", "m4": "M4", "m5": "M5", "m6": "M6",
            "m6f": "M6f", "m7": "M7", "m7f": "M7f", "m8": "M8"}
FLOW_ARMS = ["m1", "m2", "m4", "m5", "m6", "m7", "m8"]
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

R: dict = {}  # every number used in the markdown lives here


def jl(p):
    return json.loads(Path(p).read_text())


def save_csv(name, header, rows):
    with open(TABS / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def savefig(fig, name):
    fig.savefig(FIGS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ load
run_names = sorted(p.name for p in RUNS.iterdir()
                   if p.is_dir() and re.fullmatch(r"m\d+f?_s\d_f[\d.]+", p.name))
info = {}
for n in run_names:
    m = re.fullmatch(r"(m\d+f?)_s(\d)_f([\d.]+)", n)
    info[n] = dict(arm=m.group(1), seed=int(m.group(2)), frac=float(m.group(3)))
R["n_runs"] = len(run_names)
full = [n for n in run_names if info[n]["frac"] == 1.0]
R["n_runs_f1"] = len(full)

PP = {}  # per-pixel test arrays per run (float32 to keep sums honest)


def load_pp(path):
    d = np.load(path, allow_pickle=True)
    return {k: (d[k].astype(np.float32) if d[k].dtype == np.float16 else d[k]) for k in d.files}


for n in run_names:
    PP[n] = load_pp(RUNS / n / "test" / "per_pixel.npz")
ref = PP[full[0]]
names = [str(x) for x in ref["names"]]
offs = ref["offsets"]
seq_slices = [slice(offs[i], offs[i + 1]) for i in range(len(names))]
scene_of_seq = [re.fullmatch(r"sequence_\d+_(.+)_split_\d+", s).group(1) for s in names]
short = [f"{sc}/{s[-2:]}" for sc, s in zip(scene_of_seq, names)]
gt_speed, gt_depth, texture = ref["gt_speed"], ref["gt_depth"], ref["texture"]
for n in run_names:  # the bin variables must be identical across runs
    assert np.array_equal(PP[n]["gt_speed"], gt_speed)

# sanity: per-pixel pooled means reproduce the saved pre-registered metrics
maxdev = 0.0
for n in run_names:
    mj = jl(RUNS / n / "test" / "metrics.json")
    maxdev = max(maxdev, abs(PP[n]["epe"].mean() - mj["epe"]),
                 abs(np.sqrt(PP[n]["depth_sqerr"].mean()) - mj["depth_rmse"]))
R["max_dev_pp_vs_metrics"] = float(maxdev)

test_epe = {n: float(jl(RUNS / n / "test" / "metrics.json")["epe"]) for n in run_names}
test_rmse = {n: float(jl(RUNS / n / "test" / "metrics.json")["depth_rmse"]) for n in run_names}

floors_t = jl(ROOT / "floors_out/test/metrics.json")["floors"]
floors_v = jl(ROOT / "floors_out/val/metrics.json")
R["floor_test_depth_rmse"] = floors_t["depth_train_mean"]["depth_rmse"]
R["floor_test_zero_epe"] = floors_t["flow_zero"]["epe"]
R["floor_test_lk_epe"] = floors_t["flow_lucas_kanade_hex"]["epe"]
vtxt = json.dumps(floors_v)
R["floor_val_depth_rmse"] = float(re.findall(r'"depth_rmse": ([0-9.]+)', vtxt)[0])  # first = depth_train_mean
R["floor_val_zero_epe"] = float(re.findall(r'"epe": ([0-9.]+)', vtxt)[0])
c_floor = jl(ROOT / "floors_out/test/metrics.json")["train_means"]["depth_global_target"]
dn = jl(ROOT / "depth_norm.json")


def to_target(d):
    return (np.log(np.clip(d, dn["lo"], dn["hi"])) - dn["mean"]) / dn["std"]


# ================================================================== Q1 depth
T_test = to_target(gt_depth.astype(np.float64))
floor_sq = (T_test - c_floor) ** 2
R["q1_floor_rmse_recomputed"] = float(np.sqrt(floor_sq.mean()))
R["q1_train_logd_mean"], R["q1_train_logd_sd"] = dn["mean"], dn["std"]
R["q1_train_target_mean"] = c_floor
ld_test = np.log(np.clip(gt_depth.astype(np.float64), dn["lo"], dn["hi"]))
R["q1_test_logd_mean"], R["q1_test_logd_sd"] = float(ld_test.mean()), float(ld_test.std())
R["q1_test_target_mean"], R["q1_test_target_sd"] = float(T_test.mean()), float(T_test.std())
R["q1_test_depth_median"] = float(np.median(gt_depth))
# val: only m4_s0 / m1_s0 / m6f_s0 have a saved val per_pixel (same val frames for all)
vpp = load_pp(RUNS / "m4_s0_f1.0" / "val" / "per_pixel.npz")
T_val = to_target(vpp["gt_depth"].astype(np.float64))
ld_val = np.log(np.clip(vpp["gt_depth"].astype(np.float64), dn["lo"], dn["hi"]))
R["q1_val_logd_mean"], R["q1_val_logd_sd"] = float(ld_val.mean()), float(ld_val.std())
R["q1_val_target_mean"], R["q1_val_target_sd"] = float(T_val.mean()), float(T_val.std())
R["q1_val_floor_rmse_recomputed"] = float(np.sqrt(((T_val - c_floor) ** 2).mean()))
R["q1_val_depth_median"] = float(np.median(vpp["gt_depth"]))
R["q1_val_model_rmse_m4s0"] = float(jl(RUNS / "m4_s0_f1.0/val/metrics.json")["depth_rmse"])
# floor RMSE^2 = var + (mean - c)^2
R["q1_test_floor_var"], R["q1_test_floor_bias2"] = float(T_test.var()), float((T_test.mean() - c_floor) ** 2)
R["q1_val_floor_var"], R["q1_val_floor_bias2"] = float(T_val.var()), float((T_val.mean() - c_floor) ** 2)

vnames = [str(x) for x in vpp["names"]]
voffs = vpp["offsets"]
vsc = [re.fullmatch(r"sequence_\d+_(.+)_split_\d+", s).group(1) for s in vnames]
scene_stats = {}  # per scene (test and val): mean / sd of target
for tag, T, sl, sc in (("test", T_test, seq_slices, scene_of_seq),
                       ("val", T_val, [slice(voffs[i], voffs[i + 1]) for i in range(len(vnames))], vsc)):
    for s in dict.fromkeys(sc):
        idx = [i for i, x in enumerate(sc) if x == s]
        v = np.concatenate([T[sl[i]].ravel() for i in idx])
        scene_stats[f"{tag}:{s}"] = (float(v.mean()), float(v.std()),
                                     float(np.sqrt(((v - c_floor) ** 2).mean())))
save_csv("q1_scene_depth_stats.csv", ["split:scene", "target_mean", "target_sd", "floor_rmse"],
         [[k, *v] for k, v in scene_stats.items()])
R["q1_scene_stats"] = scene_stats

# per-sequence RMSE floor vs arms (f=1.0)
seq_rmse = {n: np.array([np.sqrt(PP[n]["depth_sqerr"][s].mean()) for s in seq_slices]) for n in full}
floor_seq = np.array([np.sqrt(floor_sq[s].mean()) for s in seq_slices])
arm_seq_rmse = {}
for a in ARM_NAME:
    rs = [n for n in full if info[n]["arm"] == a]
    arm_seq_rmse[a] = np.mean([seq_rmse[n] for n in rs], axis=0)
save_csv("q1_depth_rmse_per_sequence.csv", ["sequence", "floor", *ARM_NAME],
         [[short[i], floor_seq[i], *[arm_seq_rmse[a][i] for a in ARM_NAME]] for i in range(12)])
R["q1_floor_seq"] = floor_seq.tolist()
R["q1_arm_seq_rmse"] = {a: v.tolist() for a, v in arm_seq_rmse.items()}
R["q1_runs_beating_floor"] = int(sum(test_rmse[n] < R["floor_test_depth_rmse"] for n in run_names))
R["q1_val_runs_beating_floor"] = int(sum(1 for n in full if jl(RUNS / n / "val_log.jsonl".replace("val_log.jsonl", "summary.json")) is not None and min([json.loads(l) for l in open(RUNS / n / "val_log.jsonl") if l.strip()], key=lambda r: r["loss"])["depth_rmse"] < R["floor_val_depth_rmse"]))
R["q1_min_model_rmse"] = float(min(test_rmse[n] for n in run_names))
R["q1_max_model_rmse"] = float(max(test_rmse[n] for n in run_names))
R["q1_seqs_runs_beat_floor"] = int(sum((seq_rmse[n] < floor_seq).sum() for n in full))
R["q1_seqs_runs_total"] = len(full) * 12
beat_by_seq = [int(sum(seq_rmse[n][i] < floor_seq[i] for n in full)) for i in range(12)]
R["q1_beat_by_seq"] = beat_by_seq

# signed-error decomposition: only examples.npz carries predictions (7 of 12 test seqs; 3 val seqs? see below)
NONFINITE: dict = {}


def ex_stats(path, c):
    e = np.load(path)
    seqs = sorted({int(k.split("_")[0]) for k in e.files})
    out = {}
    for s in seqs:
        g = e[f"{s}_depth_gt"].astype(np.float64).ravel()  # metric depth (inverse-transformed)
        p = e[f"{s}_depth_pred"].astype(np.float64).ravel()
        ok = np.isfinite(g) & np.isfinite(p) & (g > 0) & (p > 0)
        NONFINITE[str(path)] = NONFINITE.get(str(path), 0) + int((~ok).sum())
        g, p = to_target(g[ok]), to_target(p[ok])  # back to the standardised log target used by RMSE
        err = p - g
        fl = c - g
        slope, icpt = np.polyfit(g, p, 1)
        out[s] = dict(mse=float((err ** 2).mean()), bias=float(err.mean()), var=float(err.var()),
                      floor_mse=float((fl ** 2).mean()), floor_bias=float(fl.mean()),
                      floor_var=float(fl.var()), slope=float(slope), icpt=float(icpt),
                      corr=float(np.corrcoef(g, p)[0, 1]), gmean=float(g.mean()),
                      pmean=float(p.mean()), gsd=float(g.std()), psd=float(p.std()))
    return out


dec_test = {n: ex_stats(RUNS / n / "test" / "examples.npz", c_floor) for n in full}
test_ex_seqs = sorted(dec_test[full[0]].keys())
R["q1_test_example_seqs"] = [short[i] for i in test_ex_seqs]
# example sequences are the same 7 for every run?
assert all(sorted(v.keys()) == test_ex_seqs for v in dec_test.values())


def pool(dec, rs):
    """pool decomposition over runs/sequences (mean of per-seq quantities)."""
    ks = ["mse", "bias", "var", "floor_mse", "floor_bias", "floor_var", "slope", "icpt", "corr", "gsd", "psd"]
    return {k: float(np.mean([dec[n][s][k] for n in rs for s in dec[n]])) for k in ks}


bias_all = {n: pool(dec_test, [n]) for n in full}
pooled_models = pool(dec_test, full)
R["q1_dec_models"] = pooled_models
R["q1_dec_by_arm"] = {a: pool(dec_test, [n for n in full if info[n]["arm"] == a]) for a in ARM_NAME}
# bias^2 share (mean over sequences of per-seq bias^2 / mse)
R["q1_bias2_share_models"] = float(np.mean([dec_test[n][s]["bias"] ** 2 / dec_test[n][s]["mse"]
                                            for n in full for s in test_ex_seqs]))
R["q1_bias2_mean_models"] = float(np.mean([dec_test[n][s]["bias"] ** 2 for n in full for s in test_ex_seqs]))
R["q1_var_mean_models"] = float(np.mean([dec_test[n][s]["var"] for n in full for s in test_ex_seqs]))
R["q1_bias2_mean_floor"] = float(np.mean([dec_test[full[0]][s]["floor_bias"] ** 2 for s in test_ex_seqs]))
R["q1_var_mean_floor"] = float(np.mean([dec_test[full[0]][s]["floor_var"] for s in test_ex_seqs]))
R["q1_mse_mean_floor_ex"] = float(np.mean([dec_test[full[0]][s]["floor_mse"] for s in test_ex_seqs]))
R["q1_mse_mean_models_ex"] = float(np.mean([dec_test[n][s]["mse"] for n in full for s in test_ex_seqs]))
# val counterpart (3 runs with saved val examples)
val_ex_runs = [n for n in full if (RUNS / n / "val" / "examples.npz").exists()]
dec_val = {n: ex_stats(RUNS / n / "val" / "examples.npz", c_floor) for n in val_ex_runs}
R["q1_val_ex_runs"] = val_ex_runs
R["q1_nonfinite_example_px"] = int(sum(NONFINITE.values()))
R["q1_dec_val"] = pool(dec_val, val_ex_runs) if val_ex_runs else None
R["q1_dec_test_same_runs"] = pool(dec_test, val_ex_runs) if val_ex_runs else None
R["q1_val_bias2_mean"] = float(np.mean([dec_val[n][s]["bias"] ** 2 for n in val_ex_runs for s in dec_val[n]]))
R["q1_val_var_mean"] = float(np.mean([dec_val[n][s]["var"] for n in val_ex_runs for s in dec_val[n]]))
R["q1_val_floor_bias2_ex"] = float(np.mean([dec_val[val_ex_runs[0]][s]["floor_bias"] ** 2 for s in dec_val[val_ex_runs[0]]]))
R["q1_val_floor_var_ex"] = float(np.mean([dec_val[val_ex_runs[0]][s]["floor_var"] for s in dec_val[val_ex_runs[0]]]))
R["q1_val_mse_models_ex"] = float(np.mean([dec_val[n][s]["mse"] for n in val_ex_runs for s in dec_val[n]]))
R["q1_val_mse_floor_ex"] = float(np.mean([dec_val[val_ex_runs[0]][s]["floor_mse"] for s in dec_val[val_ex_runs[0]]]))
R["q1_val_slope"] = float(np.mean([dec_val[n][s]["slope"] for n in val_ex_runs for s in dec_val[n]]))
R["q1_val_corr"] = float(np.mean([dec_val[n][s]["corr"] for n in val_ex_runs for s in dec_val[n]]))
R["q1_val_bias_mean"] = float(np.mean([dec_val[n][s]["bias"] for n in val_ex_runs for s in dec_val[n]]))
R["q1_test_bias_mean"] = float(np.mean([dec_test[n][s]["bias"] for n in full for s in test_ex_seqs]))
R["q1_gsd_test_ex"] = float(np.mean([dec_test[full[0]][s]["gsd"] for s in test_ex_seqs]))
R["q1_psd_test_ex"] = float(np.mean([dec_test[n][s]["psd"] for n in full for s in test_ex_seqs]))
# lower bound for the achievable RMSE of a constant predictor = within-sequence sd of GT
R["q1_within_seq_sd_rmse"] = float(np.sqrt(np.mean([T_test[s].var() for s in seq_slices])))

save_csv("q1_depth_bias_variance_test_examples.csv",
         ["run", "sequence", "mse", "bias", "var", "floor_mse", "floor_bias", "slope", "corr"],
         [[n, short[s], d["mse"], d["bias"], d["var"], d["floor_mse"], d["floor_bias"], d["slope"], d["corr"]]
          for n in full for s, d in dec_test[n].items()])

# figure 1: depth
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
bins = np.linspace(-3.5, 3.5, 71)
ax[0].hist(T_val.ravel(), bins=bins, density=True, histtype="step", color=VALC, lw=1.6, label=f"val (n={len(vnames)} seq)")
ax[0].hist(T_test.ravel(), bins=bins, density=True, histtype="step", color="#1B1B1B", lw=1.6, label=f"test (n={len(names)} seq)")
ax[0].axvline(0, color=GREY, ls="--", lw=1, label="train mean (0, by construction)")
ax[0].axvline(c_floor, color=GREY, ls=":", lw=1)
ax[0].set(xlabel="standardised log-depth target [unitless, train mean 0 / sd 1]", ylabel="density [1/unit]",
          title="(a) GT depth distributions (L3)")
ax[0].legend(fontsize=7)
x = np.arange(12)
ax[1].bar(x, floor_seq, color=GREY, width=0.7, label="train-mean floor")
for a, mk in [("m1", "o"), ("m4", "o"), ("m6", "s"), ("m7", "^")]:
    ax[1].plot(x, arm_seq_rmse[a], mk, ms=4, color=ARM_COLOR[a], mfc="none" if a in ("m6", "m7") else ARM_COLOR[a],
               label=f"{ARM_NAME[a]} (mean of 3 seeds)")
ax[1].set_xticks(x, short, rotation=70, ha="right", fontsize=6)
ax[1].set(ylabel="depth RMSE [std. log-depth units]", title="(b) test RMSE per sequence (L3)")
ax[1].legend(fontsize=6.5, ncol=1)
b2 = [np.mean([dec_test[n][s]["bias"] ** 2 for s in test_ex_seqs]) for n in full]
vr = [np.mean([dec_test[n][s]["var"] for s in test_ex_seqs]) for n in full]
ax[2].scatter(b2, vr, c=[ARM_COLOR[info[n]["arm"]] for n in full], s=28, edgecolor="k", lw=0.4)
ax[2].scatter([R["q1_bias2_mean_floor"]], [R["q1_var_mean_floor"]], marker="s", s=60, color=GREY, edgecolor="k", label="train-mean floor")
ax[2].set(xlabel="mean bias$^2$ per sequence [(std. units)$^2$]", ylabel="mean error variance [(std. units)$^2$]",
          title="(c) MSE = bias$^2$ + variance, 7 test seqs (L3)")
ax[2].legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in [("connectome (M1,M6,M6f)", AMBER), ("null/rewired (M2,M3,M7,M7f)", CORAL), ("deep learning (M4,M5,M8)", TEAL)]] + [Line2D([], [], marker="s", ls="", color=GREY, mec="k", label="train-mean floor")], fontsize=6.5)
fig.tight_layout()
savefig(fig, "ea_q1_depth.png")

# ================================================================== Q2 flow
spd_edges = [0, 1, 2, 4, 8, 16, np.inf]
spd_lab = ["0-1", "1-2", "2-4", "4-8", "8-16", ">16"]
tex_q = np.quantile(texture, [0.2, 0.4, 0.6, 0.8])
tex_edges = [-np.inf, *tex_q.tolist(), np.inf]
tex_lab = ["Q1 (flattest)", "Q2", "Q3", "Q4", "Q5 (richest)"]
R["q2_tex_edges"] = tex_q.tolist()
spd_bin = np.digitize(gt_speed, spd_edges[1:-1])
tex_bin = np.digitize(texture, tex_q)
R["q2_spd_frac"] = [float((spd_bin == b).mean()) for b in range(6)]
R["q2_zero_by_spd"] = [float(gt_speed[spd_bin == b].mean()) for b in range(6)]
R["q2_zero_by_tex"] = [float(gt_speed[tex_bin == b].mean()) for b in range(5)]
R["q2_zero_overall"] = float(gt_speed.mean())
arm_runs = {a: [n for n in full if info[n]["arm"] == a] for a in ARM_NAME}


q2 = {}
for a in FLOW_ARMS:
    rs = arm_runs[a]
    sp = np.array([[PP[n]["epe"][spd_bin == b].mean() for b in range(6)] for n in rs])
    tx = np.array([[PP[n]["epe"][tex_bin == b].mean() for b in range(5)] for n in rs])
    win_sp = np.array([[(PP[n]["epe"][spd_bin == b] < gt_speed[spd_bin == b]).mean() for b in range(6)] for n in rs])
    win_tx = np.array([[(PP[n]["epe"][tex_bin == b] < gt_speed[tex_bin == b]).mean() for b in range(5)] for n in rs])
    q2[a] = dict(spd=sp.mean(0).tolist(), spd_sd=sp.std(0, ddof=1).tolist(), tex=tx.mean(0).tolist(),
                 tex_sd=tx.std(0, ddof=1).tolist(), win_spd=win_sp.mean(0).tolist(), win_tex=win_tx.mean(0).tolist(),
                 pix_win=float(np.mean([(PP[n]["epe"] < gt_speed).mean() for n in rs])))
R["q2"] = q2
save_csv("q2_epe_by_speed_bin.csv", ["arm", *spd_lab], [["zero-flow", *R["q2_zero_by_spd"]]] +
         [[ARM_NAME[a], *q2[a]["spd"]] for a in FLOW_ARMS])
save_csv("q2_epe_by_texture_bin.csv", ["arm", *tex_lab], [["zero-flow", *R["q2_zero_by_tex"]]] +
         [[ARM_NAME[a], *q2[a]["tex"]] for a in FLOW_ARMS])
# per-sequence EPE (mean over seeds) and zero-flow
zero_seq = np.array([gt_speed[s].mean() for s in seq_slices])
arm_seq_epe = {a: np.mean([[PP[n]["epe"][s].mean() for s in seq_slices] for n in arm_runs[a]], axis=0)
               for a in ARM_NAME}
R["q2_zero_seq"] = zero_seq.tolist()
R["q2_arm_seq_epe"] = {a: v.tolist() for a, v in arm_seq_epe.items()}
save_csv("q2_epe_per_sequence.csv", ["sequence", "zero_flow", *ARM_NAME],
         [[short[i], zero_seq[i], *[arm_seq_epe[a][i] for a in ARM_NAME]] for i in range(12)])
# M5 vs zero, M5 vs M4, M5 vs M1 per sequence and per seed
m5_minus_zero = arm_seq_epe["m5"] - zero_seq
R["q2_m5_minus_zero_seq"] = m5_minus_zero.tolist()
R["q2_m5_seeds"] = {info[n]["seed"]: test_epe[n] for n in arm_runs["m5"]}
R["q2_m5_per_seed_seq_minus_zero"] = {info[n]["seed"]: [float(PP[n]["epe"][s].mean() - zero_seq[i])
                                                       for i, s in enumerate(seq_slices)] for n in arm_runs["m5"]}
R["q2_m5_seqs_beating_zero_all_seeds"] = int(sum(all(v[i] < 0 for v in R["q2_m5_per_seed_seq_minus_zero"].values())
                                                 for i in range(12)))
R["q2_m5_minus_m1_seq"] = (arm_seq_epe["m5"] - arm_seq_epe["m1"]).tolist()
R["q2_epe_mean"] = {a: float(np.mean([test_epe[n] for n in arm_runs[a]])) for a in ARM_NAME}
# which sequences carry M5's gain relative to zero-flow; weight by pixel share (equal here)
R["q2_m5_gain_share"] = (-(m5_minus_zero) / (-(m5_minus_zero)).sum()).tolist()
# scene-level
scenes = list(dict.fromkeys(scene_of_seq))
R["q2_scenes"] = scenes
R["q2_scene_zero"] = [float(np.mean([zero_seq[i] for i, s in enumerate(scene_of_seq) if s == sc])) for sc in scenes]
R["q2_scene_arm"] = {a: [float(np.mean([arm_seq_epe[a][i] for i, s in enumerate(scene_of_seq) if s == sc]))
                         for sc in scenes] for a in ARM_NAME}
# bias of flow: does the model under-predict speed? (needs signed pred -> examples only)
sp_ratio = {}
for a in FLOW_ARMS:
    r = []
    for n in arm_runs[a]:
        e = np.load(RUNS / n / "test" / "examples.npz")
        for k in sorted({int(x.split("_")[0]) for x in e.files}):
            g = np.linalg.norm(e[f"{k}_flow_gt"].astype(np.float32), axis=1)
            p = np.linalg.norm(e[f"{k}_flow_pred"].astype(np.float32), axis=1)
            r.append((p.mean(), g.mean()))
    r = np.array(r)
    sp_ratio[a] = float(r[:, 0].mean() / r[:, 1].mean())
R["q2_pred_speed_ratio"] = sp_ratio

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
xs = np.arange(6)
ax[0].plot(xs, R["q2_zero_by_spd"], "s--", color=GREY, label="zero-flow (= GT speed)")
for a in FLOW_ARMS:
    ax[0].errorbar(xs, q2[a]["spd"], yerr=q2[a]["spd_sd"], marker="o", ms=3.5, lw=1.2, capsize=2, color=ARM_COLOR[a],
                   ls=ARM_LS[a], label=f"{ARM_NAME[a]} [{ARM_GROUP[a][:4]}.]")
ax[0].set_xticks(xs, spd_lab)
ax[0].set(xlabel="GT speed bin [hex-lattice units / frame]", ylabel="mean EPE [lattice units / frame]",
          title="(a) EPE by GT speed (L3)")
ax[0].legend(fontsize=6.5, ncol=2)
xt = np.arange(5)
ax[1].plot(xt, R["q2_zero_by_tex"], "s--", color=GREY, label="zero-flow")
for a in FLOW_ARMS:
    ax[1].errorbar(xt, q2[a]["tex"], yerr=q2[a]["tex_sd"], marker="o", ms=3.5, lw=1.2, capsize=2, color=ARM_COLOR[a],
                   ls=ARM_LS[a], label=ARM_NAME[a])
ax[1].set_xticks(xt, ["Q1\nflat", "Q2", "Q3", "Q4", "Q5\nrich"])
ax[1].set(xlabel="local luminance-variance quintile (test pooled)", ylabel="mean EPE [lattice units / frame]",
          title="(b) EPE by texture (L3)")
w = 0.11
for j, a in enumerate(["m1", "m4", "m5", "m6", "m7"]):
    ax[2].bar(np.arange(12) + (j - 2) * w, arm_seq_epe[a] - zero_seq, width=w, color=ARM_COLOR[a],
              hatch={"m1": "", "m4": "", "m5": "//", "m6": "..", "m7": "xx"}[a], edgecolor="k", lw=0.3, label=f"{ARM_NAME[a]} [{ARM_GROUP[a][:4]}.]")
ax[2].axhline(0, color=GREY, lw=1)
ax[2].set_xticks(np.arange(12), short, rotation=70, ha="right", fontsize=6)
ax[2].set(ylabel="EPE(model) - EPE(zero-flow) [lattice units / frame]\n(< 0: beats zero-flow)",
          title="(c) per-sequence gap to zero-flow (L3)")
ax[2].legend(fontsize=6.5, ncol=2)
fig.tight_layout()
savefig(fig, "ea_q2_flow.png")

# ================================================================== Q3 val vs test
val_epe, val_rmse = {}, {}
for n in full:
    rows = [json.loads(l) for l in open(RUNS / n / "val_log.jsonl") if l.strip()]
    b = min(rows, key=lambda r: r["loss"])
    val_epe[n], val_rmse[n] = b["epe"], b["depth_rmse"]


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(rank(x), rank(y))[0, 1])


vx = [val_epe[n] for n in full]
tx_ = [test_epe[n] for n in full]
R["q3_pearson_spearman"] = corr(vx, tx_)
R["q3_depth_pearson_spearman"] = corr([val_rmse[n] for n in full], [test_rmse[n] for n in full])
# zero-flow-subtracted: val zero-flow 5.074, test 5.620 => offset only; correlation unaffected.
# arm-mean correlation (10 arms) and within-arm (seed) correlation
arm_val = [np.mean([val_epe[n] for n in arm_runs[a]]) for a in ARM_NAME]
arm_test = [np.mean([test_epe[n] for n in arm_runs[a]]) for a in ARM_NAME]
R["q3_arm_pearson_spearman"] = corr(arm_val, arm_test)
R["q3_arm_val"] = dict(zip(ARM_NAME, arm_val))
R["q3_arm_test"] = dict(zip(ARM_NAME, arm_test))
# within-arm: demean per arm, then correlate (removes between-arm structure)
dv, dt = [], []
for a in ARM_NAME:
    v = np.array([val_epe[n] for n in arm_runs[a]])
    t = np.array([test_epe[n] for n in arm_runs[a]])
    dv += (v - v.mean()).tolist()
    dt += (t - t.mean()).tolist()
R["q3_within_arm_pearson_spearman"] = corr(dv, dt)
R["q3_val_range"] = [float(min(vx)), float(max(vx))]
R["q3_test_range"] = [float(min(tx_)), float(max(tx_))]
R["q3_n_val_beat_zero"] = int(sum(v < R["floor_val_zero_epe"] for v in vx))
R["q3_n_test_beat_zero"] = int(sum(t < R["floor_test_zero_epe"] for t in tx_))
R["q3_n_val_beat_lk"] = int(sum(v < 4.708866596221924 for v in vx))
R["q3_n_test_beat_lk"] = int(sum(t < R["floor_test_lk_epe"] for t in tx_))
R["q3_val_zero"], R["q3_val_lk"] = R["floor_val_zero_epe"], 4.708866596221924
R["q3_margin_val"] = float(np.mean([R["floor_val_zero_epe"] - v for v in vx]))
R["q3_margin_test"] = float(np.mean([R["floor_test_zero_epe"] - t for t in tx_]))
save_csv("q3_val_vs_test_epe.csv", ["run", "arm", "val_epe_best_ckpt", "test_epe", "val_depth_rmse", "test_depth_rmse"],
         [[n, info[n]["arm"], val_epe[n], test_epe[n], val_rmse[n], test_rmse[n]] for n in full])
# M8 vs M4 per sequence (seed-paired)
m8_seq = {s: np.array([PP[n]["epe"][sl].mean() for sl in seq_slices]) for s, n in
          ((info[n]["seed"], n) for n in arm_runs["m8"])}
m4_seq = {s: np.array([PP[n]["epe"][sl].mean() for sl in seq_slices]) for s, n in
          ((info[n]["seed"], n) for n in arm_runs["m4"])}
d84 = np.array([m8_seq[s] - m4_seq[s] for s in sorted(m8_seq)])  # (3,12) >0: M8 worse
R["q3_m8_minus_m4_seq_mean"] = d84.mean(0).tolist()
R["q3_m8_minus_m4_seq_min"] = d84.min(0).tolist()
R["q3_m8_minus_m4_seq_max"] = d84.max(0).tolist()
R["q3_m8_worse_all_seeds"] = int(sum((d84[:, i] > 0).all() for i in range(12)))
R["q3_m8_better_all_seeds"] = int(sum((d84[:, i] < 0).all() for i in range(12)))
R["q3_m8_minus_m4_overall"] = float(d84.mean())
vm4 = np.mean([val_epe[n] for n in arm_runs["m4"]])
vm8 = np.mean([val_epe[n] for n in arm_runs["m8"]])
R["q3_val_m4_m8"] = (float(vm4), float(vm8))
save_csv("q3_m8_vs_m4_per_sequence.csv", ["sequence", "m8_minus_m4_mean", "min_over_seeds", "max_over_seeds"],
         [[short[i], d84.mean(0)[i], d84.min(0)[i], d84.max(0)[i]] for i in range(12)])

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
for n in full:
    a = info[n]["arm"]
    ax[0].scatter(val_epe[n], test_epe[n], color=ARM_COLOR[a], s=30, edgecolor="k", lw=0.4,
                  marker={"m1": "o", "m2": "s", "m3": "^", "m4": "o", "m5": "D", "m6": "s", "m6f": "P", "m7": "s",
                          "m7f": "X", "m8": "v"}[a])
ax[0].axhline(R["floor_test_zero_epe"], color=GREY, ls="--", lw=1)
ax[0].axvline(R["floor_val_zero_epe"], color=GREY, ls="--", lw=1)
ax[0].text(R["floor_val_zero_epe"], ax[0].get_ylim()[1], " zero-flow", color=GREY, fontsize=7, va="top")
ax[0].set(xlabel="val EPE at best checkpoint [lattice units / frame]", ylabel="test EPE [lattice units / frame]",
          title=f"(a) val vs test, {len(full)} runs at f=1.0: r={R['q3_pearson_spearman'][0]:.2f} (L3)")
from matplotlib.lines import Line2D  # noqa: E402
MK = {"m1": "o", "m2": "s", "m3": "^", "m4": "o", "m5": "D", "m6": "s", "m6f": "P", "m7": "s", "m7f": "X", "m8": "v"}
ax[0].legend(handles=[Line2D([], [], marker=MK[a], ls="", color=c, label=f"{ARM_NAME[a]} [{ARM_GROUP[a]}]") for a, c in ARM_COLOR.items()],
             fontsize=6, ncol=2)
ax[1].bar(np.arange(12), d84.mean(0), color=TEAL, edgecolor="k", lw=0.4)
for i in range(12):
    ax[1].plot([i] * 3, d84[:, i], "k.", ms=3)
ax[1].axhline(0, color=GREY, lw=1)
ax[1].set_xticks(np.arange(12), short, rotation=70, ha="right", fontsize=6)
ax[1].set(ylabel="EPE(M8, K=4) - EPE(M4) [lattice units / frame]\n(> 0: M8 worse)",
          title="(b) M8 vs M4 per test sequence; dots = seeds (L3)")
fig.tight_layout()
savefig(fig, "ea_q3_val_test.png")

# ================================================================== Q4 any-time
kcurve = {}
for k in (1, 2, 3, 4):
    kcurve[k] = [float(jl(RUNS / n / f"test_k{k}" / "metrics.json")["epe"]) for n in sorted(arm_runs["m8"])]
R["q4"] = {k: dict(vals=v, mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for k, v in kcurve.items()}
R["q4_m4_mean"] = float(np.mean([test_epe[n] for n in arm_runs["m4"]]))
R["q4_k4_equals_test"] = bool(all(abs(kcurve[4][i] - test_epe[n]) < 1e-6 for i, n in enumerate(sorted(arm_runs["m8"]))))
R["q4_paired_drop_k1_k4"] = [kcurve[1][i] - kcurve[4][i] for i in range(3)]
save_csv("q4_m8_anytime.csv", ["K", "mean_epe", "sd_epe", "seed0", "seed1", "seed2"],
         [[k, R["q4"][k]["mean"], R["q4"][k]["sd"], *kcurve[k]] for k in kcurve])
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ks = np.array([1, 2, 3, 4])
for i in range(3):
    ax.plot(ks, [kcurve[k][i] for k in ks], "-", color=ARM_COLOR["m8"], alpha=0.35, lw=1)
ax.errorbar(ks, [R["q4"][k]["mean"] for k in ks], yerr=[R["q4"][k]["sd"] for k in ks], color=ARM_COLOR["m8"], marker="o",
            capsize=3, lw=2, ls="-", label="M8 (recurrent) mean $\\pm$ sd, 3 seeds")
ax.axhline(R["q4_m4_mean"], color=ARM_COLOR["m4"], ls="--", lw=1.5, label="M4 (feed-forward) mean")
ax.axhline(R["floor_test_zero_epe"], color=GREY, ls=":", lw=1.2, label="zero-flow floor")
ax.axhline(R["floor_test_lk_epe"], color=GREY, ls="-.", lw=1.2, label="Lucas-Kanade floor")
ax.set_xticks(ks)
ax.set(xlabel="inner recurrent steps K [count]", ylabel="test EPE [lattice units / frame]",
       title="M8 any-time curve (L3)")
ax.legend(fontsize=6.5)
fig.tight_layout()
savefig(fig, "ea_q4_anytime.png")

# ================================================================== Q5 qualitative
from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel  # noqa: E402

u, v = get_hex_coords(15)
hx, hy = hex_to_pixel(u, v)


def flow_rgb(f, vmax):
    """f: (2,721) channels (u,v) as saved. hue = direction, saturation = speed / vmax, value = 1."""
    ang = (np.arctan2(f[1], f[0]) + np.pi) / (2 * np.pi)
    sat = np.clip(np.hypot(f[0], f[1]) / vmax, 0, 1)
    return hsv_to_rgb(np.stack([ang, sat, np.ones_like(sat)], -1))


def hexplot(ax, vals, title, cmap=None, vmin=None, vmax=None, rgb=None):
    if rgb is not None:
        ax.scatter(hx, hy, c=rgb, s=14, marker="h", linewidths=0)
        sc = None
    else:
        sc = ax.scatter(hx, hy, c=vals, s=14, marker="h", linewidths=0, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def best_worst(run):
    e = np.load(RUNS / run / "test" / "examples.npz")
    cand = []
    for s in sorted({int(k.split("_")[0]) for k in e.files}):
        g, p = e[f"{s}_flow_gt"].astype(np.float32), e[f"{s}_flow_pred"].astype(np.float32)
        epe = np.linalg.norm(g - p, axis=1).mean(1)
        for t in range(len(epe)):
            cand.append((float(epe[t]), s, t))
    cand.sort()
    return e, cand[0], cand[-1]


qual = []
fig, axs = plt.subplots(4, 6, figsize=(13.5, 9.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1, 0.07]))
rows = []
for run, lab in (("m4_s0_f1.0", "M4"), ("m1_s0_f1.0", "M1")):
    e, b, w = best_worst(run)
    rows += [(run, lab, "best", e, b), (run, lab, "worst", e, w)]
for r, (run, lab, kind, e, (epe, s, t)) in enumerate(rows):
    g, p = e[f"{s}_flow_gt"][t].astype(np.float32), e[f"{s}_flow_pred"][t].astype(np.float32)
    dg = to_target(e[f"{s}_depth_gt"][t, 0].astype(np.float64)).astype(np.float32)
    dp = to_target(e[f"{s}_depth_pred"][t, 0].astype(np.float64)).astype(np.float32)
    lum = e[f"{s}_lum"][t, 0].astype(np.float32)
    vmax = float(np.percentile(np.hypot(g[0], g[1]), 99)) or 1.0
    spd = float(np.hypot(g[0], g[1]).mean())
    drm = float(np.sqrt(((dp - dg) ** 2).mean()))
    qual.append(dict(model=lab, kind=kind, seq=short[s], frame=t, epe=epe, gt_speed=spd, depth_rmse=drm,
                     zero_epe=spd))
    row = axs[r]
    hexplot(row[0], lum, f"{lab} {kind}: {short[s]} f{t}\nluminance [a.u.]", cmap="gray", vmin=0, vmax=1)
    hexplot(row[1], None, f"GT flow (EPE0={spd:.2f})", rgb=flow_rgb(g, vmax))
    hexplot(row[2], None, f"{lab} predicted flow (EPE={epe:.2f})", rgb=flow_rgb(p, vmax))
    s1 = hexplot(row[3], dg, "GT depth [std. log]", cmap="viridis", vmin=-2, vmax=2)
    hexplot(row[4], dp, f"{lab} pred depth (RMSE={drm:.2f})", cmap="viridis", vmin=-2, vmax=2)
    cb = fig.colorbar(s1, cax=row[5])
    cb.set_label("log-depth, standardised [unitless]", fontsize=6)
    cb.ax.tick_params(labelsize=6)
for ax in axs[:, :5].ravel():
    ax.set_xlabel("x [hex-lattice units]", fontsize=5, labelpad=1)
    ax.set_ylabel("y [hex-lattice units]", fontsize=5, labelpad=1)
fig.suptitle("Best and worst test frames by flow EPE (seed 0, saved example sequences) - L3 exploratory.\n"
             "Flow components plotted as (x,y) as saved; saturation scale = per-row 99th percentile of GT speed.", fontsize=9)
fig.tight_layout()
wax = fig.add_axes([0.46, -0.1, 0.08, 0.07])
X, Y = np.meshgrid(np.linspace(-1, 1, 80), np.linspace(-1, 1, 80))
wim = hsv_to_rgb(np.stack([(np.arctan2(Y, X) + np.pi) / (2 * np.pi), np.clip(np.hypot(X, Y), 0, 1),
                           np.ones_like(X)], -1))
wim = np.where((np.hypot(X, Y) <= 1)[..., None], wim, 1.0)
wax.imshow(wim, origin="lower", extent=[-1, 1, -1, 1])
wax.set_xticks([-1, 0, 1])
wax.set_yticks([-1, 0, 1])
wax.tick_params(labelsize=5)
wax.set_title("flow colour wheel: hue = direction,\nsaturation = speed / row vmax", fontsize=6)
wax.set_xlabel("u component [rel. speed]", fontsize=5)
wax.set_ylabel("v component [rel. speed]", fontsize=5)
savefig(fig, "ea_q5_qualitative.png")
R["q5"] = qual
save_csv("q5_qualitative_frames.csv", list(qual[0].keys()), [list(q.values()) for q in qual])

# -------------------------------------------------------------------- dump
(TABS / "error_analysis_numbers.json").write_text(json.dumps(R, indent=1, default=float))
print("numbers ->", TABS / "error_analysis_numbers.json")

# ================================================================== markdown (Thai); every number comes from R
def f(x, d=3):
    return f"{x:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def write_md():
    q2_, L = R["q2"], []
    A = L.append
    scs = R["q1_scene_stats"]
    arms_show = ["m1", "m4", "m5", "m6", "m7"]
    A("# Error analysis (pre-registered, PLAN/EVALUATION)")
    A("")
    A("> ทุกตัวเลขในเอกสารนี้สร้างโดย `error_analysis.py` จาก output ที่บันทึกไว้แล้วเท่านั้น (`per_pixel.npz`, `examples.npz`, `metrics.json`, `val_log.jsonl`, `floors_out/`, `depth_norm.json`) "
      "ไม่มีการรัน `eval.py`/`floors.py` ซ้ำบน test. ตัวชี้วัดที่ลงทะเบียนไว้ล่วงหน้า (EPE, RMSE, margin เหนือ floor) ระบุว่า pre-registered; "
      "การแบ่ง bin, bias/variance, correlation และภาพทั้งหมดเป็น **L3 (exploratory)** ไม่ใช่หลักฐานยืนยัน. "
      f"ตาราง CSV อยู่ที่ `results/error_analysis/`, ตัวเลขดิบทั้งหมดใน `error_analysis_numbers.json`.")
    A("")
    A(f"หมายเหตุ: มี {R['n_runs']} runs ทั้งหมด แต่ที่ f=1.0 มี **{R['n_runs_f1']}** runs (10 arm x 3 seed) ไม่ใช่ 33; "
      f"ใช้ {R['n_runs_f1']} runs สำหรับ Q1-Q3. ตรวจสอบแล้วว่าค่าเฉลี่ยจาก `per_pixel.npz` ตรงกับ `metrics.json` (ต่างสูงสุด {R['max_dev_pp_vs_metrics']:.1e}).")
    A("")
    # ---------------- Q1
    A("## 1. Depth: ทำไมไม่มีโมเดลชนะ train-mean floor บน test")
    A("")
    A(f"ข้อเท็จจริง (pre-registered metric): floor RMSE test = **{f(R['floor_test_depth_rmse'])}**, โมเดล {R['q1_runs_beating_floor']}/{R['n_runs']} runs ชนะ "
      f"(ช่วง {f(R['q1_min_model_rmse'])}-{f(R['q1_max_model_rmse'])}); บน val floor = {f(R['floor_val_depth_rmse'])} และโมเดลชนะ {R['q1_val_runs_beating_floor']}/{R['n_runs_f1']} runs (f=1.0, best checkpoint). "
      f"คำนวณ floor ใหม่จาก `gt_depth` ได้ {f(R['q1_floor_rmse_recomputed'], 4)} (test) และ {f(R['q1_val_floor_rmse_recomputed'], 4)} (val) ตรงกับไฟล์ floors.")
    A("")
    A("**(a) การกระจายของ log-depth ต่าง split มาก** (L3). train จาก `depth_norm.json` (ค่า clip แล้ว); val/test คำนวณจาก `gt_depth` (val จาก `m4_s0/val/per_pixel.npz`). "
      "target = (log depth - mean)/sd ของ train ดังนั้น train มี mean 0, sd 1 โดยนิยาม.")
    A("")
    for line in md_table(["split", "log-depth mean", "log-depth sd", "target mean", "target sd", "median depth (Sintel units)", "floor RMSE^2 = var + (mean-c)^2"], [
        ["train", f(R["q1_train_logd_mean"]), f(R["q1_train_logd_sd"]), f(0.0), f(1.0), "-", "-"],
        ["val", f(R["q1_val_logd_mean"]), f(R["q1_val_logd_sd"]), f(R["q1_val_target_mean"]), f(R["q1_val_target_sd"]), f(R["q1_val_depth_median"], 2),
         f"{f(R['q1_val_floor_var'])} + {f(R['q1_val_floor_bias2'])}"],
        ["test", f(R["q1_test_logd_mean"]), f(R["q1_test_logd_sd"]), f(R["q1_test_target_mean"]), f(R["q1_test_target_sd"]), f(R["q1_test_depth_median"], 2),
         f"{f(R['q1_test_floor_var'])} + {f(R['q1_test_floor_bias2'])}"]]):
        A(line)
    A("")
    A(f"val เป็นฉากไกล (market/mountain; target mean {f(R['q1_val_target_mean'])} เท่าของ sd train) ทำให้ train-mean floor แย่ ({f(R['q1_val_floor_bias2'])} ของ MSE มาจาก offset ของ mean). "
      f"test (bandage/cave) อยู่ใกล้ค่าเฉลี่ย train (target mean {f(R['q1_test_target_mean'])}) floor จึงเกือบเป็น oracle แบบค่าคงที่: "
      f"sd ภายในแต่ละ sequence เหลือเพียง {f(R['q1_within_seq_sd_rmse'])} (RMSE ที่ได้ถ้าเดาค่าคงที่ที่ถูกต้องต่อ sequence). "
      "การเทียบกับ floor บน test จึงเป็นเกณฑ์ที่เข้มกว่า val มาก ไม่ใช่ว่าโมเดลแย่ลงเมื่อเทียบกับ val.")
    A("")
    A("แยกตามฉาก (target units; floor RMSE ต่อฉาก):")
    A("")
    for line in md_table(["scene", "split", "target mean", "target sd", "floor RMSE"],
                         [[k.split(":")[1], k.split(":")[0], f(v[0]), f(v[1]), f(v[2])] for k, v in scs.items()]):
        A(line)
    A("")
    A("**(b) RMSE รายsequence: floor เทียบกับโมเดล (mean 3 seed, L3)**")
    A("")
    hdr = ["sequence", "floor"] + [ARM_NAME[a] for a in arms_show]
    for line in md_table(hdr, [[short[i], f(R["q1_floor_seq"][i], 2)] + [f(R["q1_arm_seq_rmse"][a][i], 2) for a in arms_show] for i in range(12)]):
        A(line)
    A("")
    bs = R["q1_beat_by_seq"]
    A(f"จาก {R['q1_seqs_runs_total']} คู่ (run x sequence) โมเดลชนะ floor {R['q1_seqs_runs_beat_floor']} คู่; จำนวน run (จาก {R['n_runs_f1']}) ที่ชนะ floor ต่อ sequence = {bs}. "
      "โมเดลชนะได้เฉพาะ bandage (6 sequence แรก) และแพ้ cave ทุก run: cave_2 (ฉากไกลกว่า, target mean "
      f"{f(scs['test:cave_2'][0])}) โมเดล RMSE ~1.2-1.4 กับ floor ~0.9-1.0; cave_4 floor เพียง {f(R['q1_floor_seq'][9], 2)}-{f(R['q1_floor_seq'][11], 2)} (ใกล้ train-mean มาก) โมเดลแพ้ถึง 2 เท่า.")
    A("")
    dm, dv_, ft = R["q1_dec_models"], R["q1_dec_val"], R["q1_dec_test_same_runs"]
    A("**(c) แยก MSE = bias^2 + variance ของ error** (L3). `per_pixel.npz` เก็บเฉพาะ sqerr ไม่มี sign จึงใช้ `examples.npz` (ทำนายจริง) "
      f"แปลงกลับเป็นหน่วย target; ครอบคลุม 7 จาก 12 test sequences ({', '.join(R['q1_test_example_seqs'])}) เท่านั้น และ {R['q1_nonfinite_example_px']} pixel ที่ไม่ finite (float16 overflow) ถูกตัดออก. "
      "ค่าเป็นค่าเฉลี่ยของตัวเลขรายsequence (ไม่ใช่ pooled RMSE) จึงไม่เท่าตัวเลข RMSE ใน (a)-(b) โดยตรง.")
    A("")
    for line in md_table(["กลุ่ม", "MSE", "bias^2", "variance", "corr(pred,GT) ใน seq", "slope pred~GT", "sd ของ pred / sd ของ GT"], [
        [f"floor (test, 7 seq)", f(R["q1_mse_mean_floor_ex"]), f(R["q1_bias2_mean_floor"]), f(R["q1_var_mean_floor"]), "-", "0", f"0 / {f(R['q1_gsd_test_ex'])}"],
        [f"โมเดล f=1.0 ({R['n_runs_f1']} runs, test)", f(R["q1_mse_mean_models_ex"]), f(R["q1_bias2_mean_models"]), f(R["q1_var_mean_models"]), f(dm["corr"], 2), f(dm["slope"], 2), f"{f(dm['psd'], 2)} / {f(dm['gsd'], 2)}"],
        [f"val floor (3 seq)", f(R["q1_val_mse_floor_ex"]), f(R["q1_val_floor_bias2_ex"]), f(R["q1_val_floor_var_ex"]), "-", "0", "-"],
        [f"โมเดล M1/M4/M6f s0 (val, 3 seq)", f(R["q1_val_mse_models_ex"]), f(R["q1_val_bias2_mean"]), f(R["q1_val_var_mean"]), f(dv_["corr"], 2), f(dv_["slope"], 2), f"{f(dv_['psd'], 2)} / {f(dv_['gsd'], 2)}"],
        [f"โมเดลชุดเดียวกัน (test, 7 seq)", f(ft["mse"]), "-", "-", f(ft["corr"], 2), f(ft["slope"], 2), f"{f(ft['psd'], 2)} / {f(ft['gsd'], 2)}"]]):
        A(line)
    A("")
    A(f"bias^2 คิดเป็น {f(100 * R['q1_bias2_share_models'], 0)}% ของ MSE ของโมเดล (mean ของสัดส่วนรายsequence); โมเดลทำนาย depth แทบเป็นค่าคงที่ (sd ของ pred ~{f(dm['psd'], 2)} เทียบกับ sd จริง {f(dm['gsd'], 2)}) "
      f"และ **ภายใน sequence ที่ test correlation ติดลบ ({f(dm['corr'], 2)})** ขณะที่บน val เป็นบวก ({f(dv_['corr'], 2)}). "
      "ดังนั้นคำตอบคือ: (1) ส่วนใหญ่เป็น **calibration/level shift ของ split** (floor test ต่ำเพราะ test อยู่ใกล้ mean ของ train ไม่ใช่เพราะ floor ฉลาด); "
      "(2) โมเดลไม่ได้เรียน depth ภายใน sequence ที่ generalise ข้ามฉาก (correlation เปลี่ยนเครื่องหมาย val -> test) จึงได้ MSE สูงกว่าการเดาค่าคงที่ "
      "เพราะทั้ง bias^2 และ variance ของโมเดลสูงกว่า floor. สรุปนี้ L3 จากตัวอย่าง 7+3 sequence และ 3 runs บน val.")
    A("")
    A("![depth](figs/ea_q1_depth.png)")
    A("")
    # ---------------- Q2
    A("## 2. Flow: EPE ตามความเร็วและ texture (L3)")
    A("")
    A("zero-flow EPE = ความเร็ว GT ของ pixel นั้นพอดี. bin ความเร็วกำหนดล่วงหน้าในสคริปต์ (หน่วย lattice units/frame), bin texture = quintile ของ luminance variance ท้องถิ่น (pooled test). ตัวเลข = mean ของ 3 seed.")
    A("")
    A("ตาราง EPE ตามความเร็ว GT (ตัวหนา = ต่ำกว่า zero-flow):")
    A("")
    spd_lab_ = ["0-1", "1-2", "2-4", "4-8", "8-16", ">16"]
    tex_lab_ = ["Q1 flat", "Q2", "Q3", "Q4", "Q5 rich"]

    def row(name, vals, zero):
        return [name] + [(f"**{f(v, 2)}**" if v < z - 1e-9 else f(v, 2)) for v, z in zip(vals, zero)]
    for line in md_table(["arm"] + spd_lab_, [["pixel share"] + [f(100 * x, 0) + "%" for x in R["q2_spd_frac"]], ["zero-flow"] + [f(x, 2) for x in R["q2_zero_by_spd"]]] +
                         [row(ARM_NAME[a], q2_[a]["spd"], R["q2_zero_by_spd"]) for a in FLOW_ARMS]):
        A(line)
    A("")
    A("ตาราง EPE ตาม texture:")
    A("")
    for line in md_table(["arm"] + tex_lab_, [["zero-flow"] + [f(x, 2) for x in R["q2_zero_by_tex"]]] + [row(ARM_NAME[a], q2_[a]["tex"], R["q2_zero_by_tex"]) for a in FLOW_ARMS]):
        A(line)
    A("")
    A("สัดส่วน pixel ที่โมเดลมี EPE ต่ำกว่า zero-flow (ทุก pixel; bin 0-1 ... >16):")
    A("")
    for line in md_table(["arm", "ทั้งหมด"] + spd_lab_, [[ARM_NAME[a], f(100 * q2_[a]["pix_win"], 1) + "%"] + [f(100 * x, 0) + "%" for x in q2_[a]["win_spd"]] for a in FLOW_ARMS]):
        A(line)
    A("")
    r_ = R["q2_pred_speed_ratio"]
    A(f"ข้อสังเกต: ในทุก bin ความแตกต่างจาก zero-flow เล็กมาก (M5 ดีสุด: bin 8-16 {f(q2_['m5']['spd'][4], 2)} vs {f(R['q2_zero_by_spd'][4], 2)}). "
      f"pixel ที่ GT เคลื่อนที่ < 1 lattice unit/frame (bin 0-1) โมเดลแย่กว่า zero-flow เกือบทุก arm (M1 {f(q2_['m1']['spd'][0], 2)}, M5 {f(q2_['m5']['spd'][0], 2)} vs {f(R['q2_zero_by_spd'][0], 2)}) -- โมเดลใส่ flow สัญญาณรบกวนในบริเวณที่นิ่ง. "
      f"ที่ความเร็วสูง (>16, {f(100 * R['q2_spd_frac'][5], 0)}% ของ pixel แต่คิดเป็น {f(100 * R['q2_spd_frac'][5] * R['q2_zero_by_spd'][5] / R['q2_zero_overall'], 0)}% ของ zero-flow EPE รวม) โมเดลได้เพียงเล็กน้อย. "
      f"อัตราส่วน mean speed ของ pred ต่อ GT (บน example sequences): " + ", ".join(f"{ARM_NAME[a]} {f(r_[a], 2)}" for a in FLOW_ARMS) +
      " -- โมเดลทุกตัวทำนายช้ากว่าจริงมาก (under-predict), M5 ใกล้จริงที่สุด; นี่คือเหตุที่ EPE ใกล้ zero-flow. "
      f"texture: ยิ่งเรียบ ยิ่งผิด (zero-flow {f(R['q2_zero_by_tex'][0], 2)} ที่ Q1 เทียบ {f(R['q2_zero_by_tex'][4], 2)} ที่ Q5) แต่ความเร็วกับ texture อาจ confound กัน (ไม่ได้ควบคุม) จึงยืนยันบทบาทของ aperture problem ไม่ได้.")
    A("")
    A("EPE รายsequence (mean 3 seed) เทียบ zero-flow:")
    A("")
    for line in md_table(["sequence", "zero-flow"] + [ARM_NAME[a] for a in arms_show if a != "m8"] + ["M5 - zero"],
                         [[short[i], f(R["q2_zero_seq"][i], 2)] + [f(R["q2_arm_seq_epe"][a][i], 2) for a in arms_show if a != "m8"] + [f(R["q2_m5_minus_zero_seq"][i], 2)] for i in range(12)]):
        A(line)
    A("")
    g = R["q2_m5_gain_share"]
    A(f"**M5 vs LK**: ไม่มี per-pixel/per-sequence ของ LK ที่บันทึกไว้ (floors เก็บแค่ค่ารวม EPE LK = {f(R['floor_test_lk_epe'], 3)}) จึงบอกไม่ได้ว่า sequence ไหนทำให้ M5 ({f(R['q2_epe_mean']['m5'], 3)}, mean 3 seed; seed = "
      + ", ".join(f"{f(v, 3)}" for v in R["q2_m5_seeds"].values()) +
      f") ชนะ LK; ทำได้เพียงเทียบกับ zero-flow: M5 ชนะ zero-flow ในทั้ง 3 seed ใน {R['q2_m5_seqs_beating_zero_all_seeds']}/12 sequences. "
      f"ส่วนแบ่งของ gain รวม (EPE ลดลงเทียบ zero-flow, เฉลี่ยต่อ sequence) มาจาก cave_2 รวม {f(100 * (g[6] + g[7] + g[8]), 0)}% และ bandage_1 {f(100 * sum(g[0:3]), 0)}%, "
      f"bandage_2 {f(100 * sum(g[3:6]), 0)}%, cave_4 {f(100 * sum(g[9:12]), 0)}% (ค่าลบ = ไม่ชนะ). ความแปรปรวนข้าม seed ของ M5 สูง (sd {f(float(np.std(list(R['q2_m5_seeds'].values()), ddof=1)), 2)}) -- seed 1 ({f(R['q2_m5_seeds'][1], 3)}) ทำให้ mean ดี. "
      "(L3; M5 vs LK เป็นการเทียบ mean 3 seed กับค่าเดียว ไม่มี test ทางสถิติ)")
    A("")
    A("![flow](figs/ea_q2_flow.png)")
    A("")
    # ---------------- Q3
    A("## 3. ความไม่สอดคล้อง val-test (L3)")
    A("")
    pv, sv = R["q3_pearson_spearman"]
    A(f"EPE ของ checkpoint ที่ val loss ต่ำสุด (`val_log.jsonl`) vs test EPE, {R['n_runs_f1']} runs f=1.0: Pearson r = **{f(pv, 2)}**, Spearman = {f(sv, 2)}. "
      f"ระดับ arm (10 arm, mean 3 seed): r = {f(R['q3_arm_pearson_spearman'][0], 2)}, Spearman {f(R['q3_arm_pearson_spearman'][1], 2)}. "
      f"**ภายใน arm** (ลบ mean ของ arm ออก, ดูเฉพาะความต่างระหว่าง seed): r = {f(R['q3_within_arm_pearson_spearman'][0], 2)}, Spearman {f(R['q3_within_arm_pearson_spearman'][1], 2)} -- "
      "val ไม่ทำนาย seed ไหนจะดีบน test. (depth RMSE: r = "
      f"{f(R['q3_depth_pearson_spearman'][0], 2)}, ไม่สัมพันธ์.)")
    A("")
    A(f"ช่วงค่า: val EPE {f(R['q3_val_range'][0])}-{f(R['q3_val_range'][1])} (zero-flow {f(R['q3_val_zero'])}, LK {f(R['q3_val_lk'])}); test EPE {f(R['q3_test_range'][0])}-{f(R['q3_test_range'][1])} "
      f"(zero-flow {f(R['floor_test_zero_epe'])}, LK {f(R['floor_test_lk_epe'])}). "
      f"ชนะ zero-flow: val {R['q3_n_val_beat_zero']}/{R['n_runs_f1']}, test {R['q3_n_test_beat_zero']}/{R['n_runs_f1']}; ชนะ LK: val {R['q3_n_val_beat_lk']}/{R['n_runs_f1']}, test **{R['q3_n_test_beat_lk']}**/{R['n_runs_f1']}. "
      f"margin เฉลี่ยเหนือ zero-flow: val {f(R['q3_margin_val'])}, test {f(R['q3_margin_test'])} (หดลงราว 3 เท่า). "
      "ลำดับ arm ยังพอเหมือนกัน (M5/M4 ดี, M1/M2/M3 แย่) แต่ขนาดผลต่างหายไป: เป็นสัญญาณว่า val (4 ฉาก) เป็นตัวแทนของ test (4 ฉากอื่น) ได้ไม่ดี "
      "และการเลือก checkpoint บน val ไม่ได้ช่วย test ในระดับ seed.")
    A("")
    A(f"**M8 (K=4) vs M4 ต่อ test sequence (จับคู่ seed):** M8 แย่กว่า M4 เฉลี่ย +{f(R['q3_m8_minus_m4_overall'], 3)} EPE; แย่กว่าในทั้ง 3 seed ใน {R['q3_m8_worse_all_seeds']}/12 sequences, ดีกว่าทั้ง 3 seed ใน {R['q3_m8_better_all_seeds']}/12. "
      f"ต่างมากสุดที่ bandage_1/00 (+{f(R['q3_m8_minus_m4_seq_mean'][0], 2)}) และ cave_2/02 (+{f(R['q3_m8_minus_m4_seq_mean'][8], 2)}); น้อยสุดที่ bandage_2/01 (+{f(R['q3_m8_minus_m4_seq_mean'][4], 2)}). "
      f"แต่บน val M8 ({f(R['q3_val_m4_m8'][1], 3)}) ดีกว่า M4 ({f(R['q3_val_m4_m8'][0], 3)}) -- ทิศทางกลับด้านระหว่าง val และ test (สอดคล้อง H6 NOT SUPPORTED).")
    A("")
    A("![val-test](figs/ea_q3_val_test.png)")
    A("")
    # ---------------- Q4
    A("## 4. Recurrent depth any-time curve ของ M8 (L3)")
    A("")
    q4 = R["q4"]
    for line in md_table(["K", "test EPE mean", "sd (3 seed)", "seed 0", "seed 1", "seed 2"], [[k, f(q4[k]["mean"]), f(q4[k]["sd"])] + [f(v) for v in q4[k]["vals"]] for k in q4]):
        A(line)
    A("")
    A(f"EPE ลดลงเมื่อ K เพิ่ม ({f(q4[1]['mean'])} -> {f(q4[4]['mean'])}; paired ลดลง " + ", ".join(f(x, 3) for x in R["q4_paired_drop_k1_k4"]) +
      f" ใน 3 seed, ทิศทางเดียวกันทั้งหมด) แต่ ผลตอบแทนลดลงเร็ว: K=1->2 ลด {f(q4[1]['mean'] - q4[2]['mean'], 3)}, 3->4 ลดเพียง {f(q4[3]['mean'] - q4[4]['mean'], 3)} (น้อยกว่า sd ข้าม seed {f(q4[4]['sd'], 3)}). "
      f"ทุก K ยังแย่กว่า M4 ({f(R['q4_m4_mean'])}); K ที่ mean EPE ต่ำกว่า zero-flow ({f(R['floor_test_zero_epe'])}) คือ {[k for k in q4 if q4[k]['mean'] < R['floor_test_zero_epe']]} (K=1: {f(q4[1]['mean'])}, K=2: {f(q4[2]['mean'])}). "
      "K=4 ตรงกับ `test/metrics.json` (ตรวจแล้ว). (ไม่มี val_k* ครบทุก seed จึงไม่เทียบ val)")
    A("")
    A("![anytime](figs/ea_q4_anytime.png)")
    A("")
    # ---------------- Q5
    A("## 5. ตัวอย่างเชิงคุณภาพ (L3)")
    A("")
    A("เลือกเฟรมที่ EPE ต่ำสุด/สูงสุดจาก 7 example sequences ที่บันทึกไว้ (seed 0 ของ M4 และ M1) พิกัด hex จาก `flyvis.utils.hex_utils.get_hex_coords(15)` -> `hex_to_pixel`; สี flow: hue = ทิศ, saturation = ความเร็ว.")
    A("")
    for line in md_table(["model", "kind", "sequence/frame", "EPE", "zero-flow EPE (=GT speed)", "depth RMSE (target units)"],
                         [[q["model"], q["kind"], f"{q['seq']} f{q['frame']}", f(q["epe"], 2), f(q["zero_epe"], 2), f(q["depth_rmse"], 2)] for q in R["q5"]]):
        A(line)
    A("")
    A("การเลือกด้วย EPE สัมบูรณ์ถูกกำหนดโดยความเร็วของฉาก: 'best' คือเฟรมเกือบนิ่ง (EPE ~ GT speed เล็ก และแย่กว่า zero-flow เล็กน้อย) ส่วน 'worst' คือ cave_2/00 f47 ที่ GT speed ~33 และโมเดลทำนายแทบเป็นศูนย์ (EPE ~ speed). "
      "flow ที่ทำนายใน worst frame แทบไม่มีสี (speed ต่ำ) ส่วน depth ที่ทำนายเป็นสีเดียวกันทั้งภาพ (ค่าคงที่) ขณะ GT มีโครงสร้างชัด -- ภาพสอดคล้องกับ Q1 และ Q2.")
    A("")
    A("![qualitative](figs/ea_q5_qualitative.png)")
    A("")
    A("## ข้อจำกัดและบทเรียน")
    A("")
    A("- test มีแค่ 4 ฉาก (2 ตระกูล) 12 sequences ที่ขึ้นกับฉากสูง; sequences จากฉากเดียวกัน (split 00-02) ไม่อิสระกัน ตัวเลขรายsequence จึงไม่ใช่ n=12 อิสระ. ไม่มี CI/ทดสอบนัยสำคัญใน error analysis นี้.")
    A("- ทุกการแยก (bin, bias/variance, correlation) เป็น L3: ทำหลังเห็นผล test (แม้ pre-registered ให้ทำ error analysis) และไม่ผ่านการแก้ multiple comparisons; ใช้เป็นคำอธิบาย ไม่ใช่การยืนยัน.")
    A("- bias/variance ของ depth คำนวณได้เฉพาะ 7/12 test sequences (และ 3 runs / 3 sequences บน val) เพราะ `per_pixel.npz` ไม่เก็บ error แบบมีเครื่องหมาย; ค่านี้เป็นค่าเฉลี่ยรายsequence ไม่ใช่ RMSE ของทั้ง split.")
    A("- ไม่มี per-pixel ของ Lucas-Kanade (floors เก็บเฉพาะค่ารวม) จึงตอบ 'sequence ไหนทำให้ M5 ชนะ LK' ไม่ได้; ใช้ zero-flow เป็นตัวแทนเท่านั้น. val per_pixel มีเพียง 3 runs จึงเทียบ val กับ test ได้จำกัด.")
    A("- val 4 ฉากและ test 4 ฉากต่างกันมากด้านระยะลึก: ผล 'ทุกโมเดลชนะ floor บน val' ส่วนหนึ่งเกิดจาก floor บน val ที่ไม่เหมาะสม (ฉากไกลกว่า train) ไม่ใช่ความสามารถของโมเดล. บทเรียน: ควรรายงาน floor แบบจับคู่ฉาก/ต่อ-sequence mean และเลือก val จากฉากที่กระจายใกล้ test; ควรมีการ stratify split ตามระยะลึก.")
    A("- โมเดลทุกตัวทำนาย flow ช้ากว่าจริงและ depth เกือบคงที่: ผลโดยรวมใกล้ floor ที่ไม่เรียนรู้ ข้อสรุปเชิงชีววิทยา (connectome vs null) จึงอ่านจาก margin เล็กน้อยเหนือ floor ซึ่งเสี่ยงต่อ noise. การเทรนสั้นกว่าเปเปอร์ต้นฉบับและข้อมูลน้อย (674 เฟรม) เป็นข้อจำกัดหลัก.")
    A("- ข้อควรระวังเชิงเทคนิค: ทิศ flow ในภาพใช้ช่อง (u,v) ตามที่บันทึกเป็น (x,y) ตรง ๆ (ใช้แบบเดียวกันทั้ง GT/pred จึงเทียบกันได้ แต่ไม่ยืนยันการหมุนแกนของ lattice); float16 ใน per_pixel ทำให้เกิดความคลาดเคลื่อนเล็กน้อย (ตรวจกับ metrics.json แล้ว).")
    (ROOT / "docs" / "ERROR_ANALYSIS.md").write_text("\n".join(L) + "\n")
    print("wrote docs/ERROR_ANALYSIS.md", len(L), "lines")


if "--no-md" not in sys.argv:
    write_md()
