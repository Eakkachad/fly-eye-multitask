"""Fetch the selected Spring members (see spring_sequences.json) via HTTP range requests.
usage: fetch_spring.py <kind: frame|flowFW|disp1|cam> <outdir> <manifest.json> [seq_id ...]"""
import json, sys, time, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_zip import RemoteFile, fetch_members
IDS = dict(frame=(199097, "train_frame_left.zip", "frame_left/"), flowFW=(199011, "train_flow_FW_left.zip", "flow_FW_left/"),
           disp1=(198961, "train_disp1_left.zip", "disp1_left/"), cam=(198954, "train_cam_data.zip", "cam_data/"))
kind, out, mf = sys.argv[1:4]
only = sys.argv[4:]  # optional sequence ids; default: all selected
fid, arch, sub = IDS[kind]
sel = only or json.load(open(Path(__file__).resolve().parents[1] / "spring_sequences.json"))["test_selection"]["selected"]
url = f"https://darus.uni-stuttgart.de/api/access/datafile/{fid}"
with RemoteFile(url) as rf, zipfile.ZipFile(rf) as z:
    members = [n for n in z.namelist() if any(n.startswith(f"spring/train/{s}/{sub}") for s in sel)]
print(kind, "members", len(members), flush=True)
t0 = time.time()
recs, total = fetch_members(url, members, out, log=lambda s: print(s, flush=True))
for r in recs: r.update(file_id=fid, archive=arch)
json.dump(dict(file_id=fid, archive=arch, url=url, total_bytes_transferred=total, records=recs), open(mf, "w"), indent=1)
print("done", total, "bytes in", round(time.time() - t0), "s", flush=True)
