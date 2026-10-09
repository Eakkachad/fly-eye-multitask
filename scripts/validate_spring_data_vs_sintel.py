import sys, json, torch
sys.path.insert(0, "/home/user/flyproj/course")
import data, spring_data as sd
splits = json.load(open("/home/user/flyproj/course/splits.json"))
scenes = sys.argv[1:]
SS = data.make_split_sintel_class()
ds = SS(scenes, augment=False, all_frames=True, random_temporal_crop=False)
names = list(ds.arg_df.name)
print(names)
seqs = sorted({n.split("_split_")[0].split("_", 2)[2] for n in names})
for sc in seqs:
    r = sd.render_sintel(sc)
    for j in range(3):
        idx = [i for i, n in enumerate(names) if n.endswith(f"{sc}_split_{j:02d}")][0]
        c = ds.cached_sequences[idx]
        rep = {}
        for k in ("lum", "flow", "depth"):
            a, b = r[k][j], c[k]
            assert a.shape == b.shape, (k, a.shape, b.shape)
            rep[k] = float((a - b).abs().max())
        # resampled eval path
        g = ds.get_item(idx)
        hs = sd.SpringHex.__new__(sd.SpringHex)
        from flyvis.datasets.augmentation.temporal import Interpolate
        pw = Interpolate(24, 1 / sd.DT, mode="nearest-exact"); li = Interpolate(24, 1 / sd.DT, mode="linear")
        h = {"lum": pw(r["lum"][j]), "flow": li(r["flow"][j]), "depth": li(r["depth"][j])}
        for k in h:
            rep[k + "_resampled"] = float((h[k] - g[k]).abs().max()); assert h[k].shape == g[k].shape
        print(sc, j, tuple(r["lum"][j].shape), rep, "ranges", float(c["flow"].abs().max()), float(c["depth"].median()))
