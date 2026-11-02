r"""
check_pipe2_runs.py - are the second-pipe runs separate leak openings or segments of one recording?

Prints, for every leak / normal / continue folder: first and last file modification time, the gap to the previous
folder, and a fingerprint (MD5 of the first file) so duplicate copies become visible.

Usage (from F:\LeakCert\code):  python .\check_pipe2_runs.py            (finds Z:\*20250722* by itself)
                                python .\check_pipe2_runs.py "<root>"   (any other root)
"""
import datetime as dt
import glob
import hashlib
import os
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

root = sys.argv[1] if len(sys.argv) > 1 else next(iter(sorted(glob.glob(r"Z:\*20250722*"))), None)
if not root or not os.path.isdir(root):
    raise SystemExit("second-pipe folder not found - pass its path as an argument")
rows = []
for d, sub, fl in os.walk(root):
    mats = sorted(os.path.join(d, f) for f in fl if f.lower().endswith(".mat"))
    rel = os.path.relpath(d, root)
    if not mats or not ("leak" in rel.lower() or "normal" in rel.lower()):
        continue
    t = sorted(os.path.getmtime(m) for m in mats)
    with open(mats[0], "rb") as fh:
        h = hashlib.md5(fh.read()).hexdigest()[:12]
    rows.append((t[0], t[-1], rel, len(mats), h))
rows.sort()
dup = Counter(r[4] for r in rows)
print(f"root: {root}\n")
print(f"{'first file':<20}{'last file':<10}{'gap':>8}  {'files':>5}  {'md5':<13} folder")
prev = None
for t0, t1, rel, n, h in rows:
    gap = "" if prev is None else f"{t0 - prev:7.0f}s"
    print(f"{dt.datetime.fromtimestamp(t0):%Y-%m-%d %H:%M:%S}  {dt.datetime.fromtimestamp(t1):%H:%M:%S}{gap:>8}  {n:>5}  {h:<13}"
          f"{' DUPLICATE' if dup[h] > 1 else '          '} {rel}")
    prev = t1
print(f"\n{len(rows)} folders, {sum(1 for c in dup.values() if c > 1)} duplicated fingerprints")
print("Reading: separate openings -> leak folders minutes apart, each preceded by its own normal folder;"
      " one continuous leak -> leak folders back-to-back (gap ~0-1 s); same md5 -> copies of the same data.")
