"""Write PROVENANCE.md + manifest.json for the Spring raw members, then delete the raw dir.
Run only after the hex cache is built and the tests pass."""
import glob, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
HEX = Path.home() / "flyproj/data/spring_hex"; RAW = Path.home() / "flyproj/data/spring_raw"
COURSE = Path.home() / "flyproj/course"
du = lambda p: subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True).stdout.split()[0]
recs, per_archive = [], {}
for f in sorted(glob.glob(str(HEX / "logs/manifest_*.json"))):
    d = json.load(open(f)); recs += d["records"]
    a = per_archive.setdefault(d["archive"], dict(file_id=d["file_id"], url=d["url"], bytes_transferred=0, members=0))
    a["bytes_transferred"] += d["total_bytes_transferred"]; a["members"] += len(d["records"])
for r in recs:  # re-verify on-disk bytes before deleting
    assert hashlib.sha256(Path(r["local_path"]).read_bytes()).hexdigest() == r["sha256"], r["member"]
sel = json.load(open(COURSE / "spring_sequences.json"))["test_selection"]
before = du(RAW); total = sum(a["bytes_transferred"] for a in per_archive.values())
shutil.rmtree(RAW)
deleted = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
for r in recs: r["deleted_utc"] = deleted
cache = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(HEX.glob("spring_*.npz"))}
refetch = "cd ~/flyproj/course && for s in " + " ".join(sel["selected"]) + "; do for k in frame flowFW disp1; do ../.venv/bin/python scripts/fetch_spring.py $k ~/flyproj/data/spring_raw /tmp/m_${k}_$s.json $s; done; done; ../.venv/bin/python scripts/fetch_spring.py cam ~/flyproj/data/spring_raw /tmp/m_cam.json"
man = dict(dataset="Spring", doi="10.18419/darus-3376", license="CC BY 4.0",
           citation="Mehl, Schmalfuss, Jahedi, Nalivayko, Bruhn. Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo. CVPR 2023.",
           landing_page="https://doi.org/10.18419/darus-3376", download_url_pattern="https://darus.uni-stuttgart.de/api/access/datafile/<file_id>",
           selection=sel, archives=per_archive, total_bytes_transferred_final_run=total, raw_size_before_delete=before,
           raw_deleted_utc=deleted, refetch_command=refetch, cache_files=cache, members=recs)
json.dump(man, open(HEX / "manifest.json", "w"), indent=1)
md = f"""# Provenance of the Spring test cache

Raw members were fetched with HTTP range requests (`scripts/remote_zip.py`, User-Agent `fly-eye-multitask-research`),
rendered by `spring_data.py` / `scripts/build_spring_cache.py` into the `spring_*.npz` files here, and then **deleted** at {deleted}
(raw size before deletion: {before}). Full per-member records (zip file id, archive name, member path, uncompressed size,
CRC32 from the zip directory, sha256 of the extracted bytes, local path, download timestamp, bytes transferred, deletion
timestamp) are in `manifest.json` ({len(recs)} members). Console logs: `logs/`.

* Source: Spring dataset, DOI 10.18419/darus-3376 (https://doi.org/10.18419/darus-3376), files via
  https://darus.uni-stuttgart.de/api/access/datafile/<id>
* License: CC BY 4.0. Cite: {man['citation']}
* Selection (pre-specified, frame content not inspected): {sel['rule']}
  Selected: {', '.join(sel['selected'])}.
* Bytes transferred in the final run: {total:,} (sum of per-process totals incl. zip directory reads) plus ~1.9 MB for the initial
  directory listings and a discarded aborted first attempt (see logs/aborted_first_attempt.txt).

| archive | file id | members | bytes transferred |
|---|---|---|---|
""" + "\n".join(f"| {k} | {v['file_id']} | {v['members']} | {v['bytes_transferred']:,} |" for k, v in per_archive.items()) + f"""

## Re-fetch the same members
```
{refetch}
```
(then `python scripts/build_spring_cache.py` re-renders the cache; compare against `cache_files` sha256 in manifest.json.)
"""
(HEX / "PROVENANCE.md").write_text(md)
print("raw before", before, "deleted", deleted, "exists", RAW.exists())
