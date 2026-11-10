"""Augment simulation/energyplus/results/summary.csv with cairo + riyadh rows
extracted from raw/{case_id}/ outputs.

The current summary.csv has phoenix + albuquerque + turpan in the dry zone
(per the latest config.yaml). The raw/ folder still contains the previous
cairo + riyadh simulation outputs (61 cases per city, full 11-Tm × 5-pos
grid + baseline + 4 PSM). This script re-summarizes those raw outputs and
APPENDS the cairo + riyadh rows to summary.csv without touching existing
data.

Run-once script. Idempotent: skips if cairo/riyadh already present.
"""
from __future__ import annotations
import csv, sys
from dataclasses import asdict
from pathlib import Path

EP_DIR = Path(__file__).resolve().parents[2] / "simulation" / "energyplus"
SRC    = EP_DIR / "src"
RAW    = EP_DIR / "results" / "raw"
SUM    = EP_DIR / "results" / "summary.csv"

sys.path.insert(0, str(SRC))
from case_generator import Case
from postprocess    import summarize_case

# ── Construct Case objects for cairo + riyadh (mirror generate_cases) ────────
PCM_POS  = [0, 1, 2, 3, 4]
PCM_TM   = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
PSM_POS  = [2]
PSM_TM   = [16, 20, 24, 28]
BAND     = 4.0

cases: list[Case] = []
for city in ["cairo", "riyadh"]:
    zone = "dry"
    cases.append(Case(case_id=f"{zone}_{city}_baseline", zone=zone, city=city,
                       is_baseline=True, position=None, melt_temp=None))
    for pos in PCM_POS:
        for tm in PCM_TM:
            cases.append(Case(case_id=f"{zone}_{city}_pos{pos}_tm{int(tm)}",
                               zone=zone, city=city, is_baseline=False,
                               position=pos, melt_temp=float(tm)))
    for pos in PSM_POS:
        for tm in PSM_TM:
            cases.append(Case(case_id=f"{zone}_{city}_pos{pos}_tm{int(tm)}_psm",
                               zone=zone, city=city, is_baseline=False,
                               position=pos, melt_temp=float(tm), is_psm=True))

print(f"Constructed {len(cases)} cases ({len(cases)//2} per city × 2 cities)")

# Read existing summary, detect what's already in it
with SUM.open("r", encoding="utf-8") as f:
    existing_rows = list(csv.DictReader(f))
existing_cities = {r["city"] for r in existing_rows}
print(f"Existing summary.csv: {len(existing_rows)} rows, "
      f"cities = {sorted(existing_cities)}")

if "cairo" in existing_cities and "riyadh" in existing_cities:
    print("cairo + riyadh already present — nothing to do.")
    sys.exit(0)

# Summarize each case from raw/
summaries = []
for c in cases:
    out_dir = RAW / c.case_id
    s = summarize_case(c, out_dir, band_width=BAND)
    summaries.append(s)

# Wire baseline_delta + pct_savings + PSM linkage (mimicking build_summary logic)
baseline_by_city = {}
psm_by_pair = {}
for s in summaries:
    if s.is_baseline and s.total_kWh is not None:
        baseline_by_city[(s.zone, s.city)] = s.total_kWh
    if s.is_psm and s.total_kWh is not None and s.position is not None and s.melt_temp is not None:
        psm_by_pair[(s.zone, s.city, s.position, float(s.melt_temp))] = s.total_kWh
for s in summaries:
    if s.total_kWh is None or s.is_baseline:
        continue
    base = baseline_by_city.get((s.zone, s.city))
    if base is not None:
        s.baseline_delta_total_kWh = s.total_kWh - base
        s.pct_savings = -100.0 * s.baseline_delta_total_kWh / base if base else None

# Identify the column order in current summary.csv
fieldnames = list(existing_rows[0].keys()) if existing_rows else None
if fieldnames is None:
    print("ERROR: summary.csv is empty"); sys.exit(1)
print(f"Schema: {fieldnames}")

# Convert CaseSummary → row dict matching the schema
new_rows = []
ok_count = 0; miss_count = 0
for s in summaries:
    d = asdict(s)
    if d.get("status") != "ok":
        miss_count += 1
        continue
    # Bool → string to match existing format
    d["is_baseline"] = "True" if d["is_baseline"] else "False"
    d["is_psm"]      = "True" if d["is_psm"]      else "False"
    # Fill missing columns
    row = {col: d.get(col, "") for col in fieldnames}
    new_rows.append(row)
    ok_count += 1

print(f"Generated {ok_count} new rows ({miss_count} skipped — missing raw outputs)")

# Append to summary.csv
with SUM.open("a", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    for r in new_rows:
        w.writerow(r)

print(f"Appended {len(new_rows)} rows → {SUM}")
print(f"Total rows now: {len(existing_rows) + len(new_rows)}")
