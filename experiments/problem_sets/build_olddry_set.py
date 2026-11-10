"""Create a *new* problem-set file with the original dry cities
(cairo, phoenix, riyadh) restored, while keeping the existing
problem_set_18city.json intact.

Strategy:
  - Start from the current 18-city problem set
  - Drop albuquerque and turpan (the new dry replacements)
  - Clone phoenix's record twice and rename to cairo, riyadh
    (all dry cities share the same Tucson IDF template, so the wall
    construction, layers, and setpoints are identical — only the
    city name and EPW filename differ)
  - Refresh ground_truth from the latest summary.csv (now augmented
    with cairo + riyadh rows)

Output: experiments/problem_sets/problem_set_olddry.json   (18 cities, restored dry)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE     = Path(__file__).resolve().parent
SRC      = HERE / "problem_set_18city.json"
OUT      = HERE / "problem_set_olddry.json"

sys.path.insert(0, str(HERE.parents[1] / "src"))
from bmrag.evaluation.lookup_scoring import build_lookup

CITY_TO_EPW = {
    "cairo":  "EGY_Cairo.623660_IWEC.epw",
    "riyadh": "SAU_Riyadh.404380_IWEC.epw",
}

records = json.loads(SRC.read_text(encoding="utf-8"))
phoenix = next(r for r in records if r["city"] == "phoenix")

# Filter out the new replacements
kept = [r for r in records if r["city"] not in ("albuquerque", "turpan")]

def clone_phoenix_as(city: str) -> dict:
    rec = json.loads(json.dumps(phoenix))   # deep copy
    rec["city"] = city
    rec["epw_filename"] = CITY_TO_EPW.get(city, "")
    # Replace city in all analyst_input fields that referenced phoenix
    ai = rec["analyst_input"]
    ai["climate"] = city
    return rec

# Insert cairo + riyadh after phoenix (preserve zone grouping)
out_records = []
for r in kept:
    out_records.append(r)
    if r["city"] == "phoenix":
        out_records.append(clone_phoenix_as("cairo"))
        out_records.append(clone_phoenix_as("riyadh"))

# Refresh ground_truth from the augmented summary.csv lookup
lk = build_lookup()
for rec in out_records:
    blk = lk.get(rec["city"], {})
    if blk.get("grid"):
        rec["ground_truth"] = {
            "gt_pos":         blk["gt_pos"],
            "gt_tm":          blk["gt_tm"],
            "gt_total_kWh":   blk["gt_total_kWh"],
            "gt_savings_pct": blk["gt_savings_pct"],
            "tied_tm_set":    blk.get("tied_tm_set", []),
        }

OUT.write_text(json.dumps(out_records, ensure_ascii=False, indent=2),
                encoding="utf-8")

# Summary
print(f"Wrote {len(out_records)} records → {OUT}")
for r in out_records:
    g = r.get("ground_truth", {})
    print(f"  {r['city']:13s} zone={r['zone']:13s} gt=(pos{g.get('gt_pos','?')}, "
          f"Tm{g.get('gt_tm','?')})  tied_tm={g.get('tied_tm_set','?')}")
