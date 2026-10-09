"""Render the selected Spring raw sequences to the hex cache (data/spring_hex/*.npz)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spring_data as sd
sel = json.load(open(sd.COURSE_DIR / "spring_sequences.json"))["test_selection"]["selected"]
for s in sel:
    t = time.time()
    sd.build_cache([s])
    print(s, "built in", round(time.time() - t), "s", flush=True)
