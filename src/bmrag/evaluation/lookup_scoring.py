"""Score predictions against an EnergyPlus summary.csv by lookup; nothing is re-simulated.

Headline metric: capture_loss = max(0, 1 - pct_savings(predicted) / pct_savings(optimum)).
When summary.csv is not available, bmrag.evaluation.ground_truth scores from the ground
truth embedded in the problem sets instead.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from bmrag.config import REPO_ROOT

SUMMARY_CSV = REPO_ROOT / "simulation" / "energyplus" / "results" / "summary.csv"
EQUAL_ENERGY_KWH = 0.01
POSITION_GRID = {0, 1, 2, 3, 4}
MELT_TEMPERATURE_GRID = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}


# ══════════════════════════════════════════════════════════════════════════════
# Lookup table built from summary.csv
# ══════════════════════════════════════════════════════════════════════════════
def build_lookup(summary_csv: Path = SUMMARY_CSV) -> dict[str, dict[str, Any]]:
    """{city: {baseline_total_kWh, grid: {(pos, tm): row_dict}}}."""
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {"grid": {}})
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":          continue
            if row.get("is_psm","").lower() in ("true","1"): continue
            city = row["city"]
            total = float(row["total_kWh"]) if row["total_kWh"] else None
            if total is None:                       continue
            if row["is_baseline"].lower() in ("true","1"):
                out[city]["baseline_total_kWh"] = total
                out[city]["zone"] = row["zone"]
                continue
            key = (int(row["position"]), float(row["melt_temp"]))
            out[city]["grid"][key] = {
                "total_kWh":           total,
                "pct_savings":         float(row["pct_savings"]) if row["pct_savings"] else 0.0,
                "baseline_delta_kWh":  float(row["baseline_delta_total_kWh"]) if row["baseline_delta_total_kWh"] else 0.0,
                "discomfort_hours":    float(row["discomfort_hours"]) if row["discomfort_hours"] else None,
                "latent_utilization":  float(row["latent_utilization"]) if row["latent_utilization"] else None,
            }
    # derive ground truth per city. Two notions of equivalence:
    #   * gt_pos = argmin position (strict — empirically the same pos
    #     dominates the optimum, equal-energy plateaus span Tm only)
    #   * tied_tm_set = all Tm at gt_pos whose total_kWh equals the best
    #     (within EPS) — handles floating-point ordering artifacts where
    #     several Tm values produce identical total_kWh at the optimum pos
    for city, blk in out.items():
        if not blk.get("grid"):  continue
        gt_key  = min(blk["grid"], key=lambda k: blk["grid"][k]["total_kWh"])
        gt_row  = blk["grid"][gt_key]
        gt_pos  = gt_key[0]
        best_E  = gt_row["total_kWh"]
        tied_tm = sorted([
            k[1] for k, row in blk["grid"].items()
            if k[0] == gt_pos and abs(row["total_kWh"] - best_E) <= EQUAL_ENERGY_KWH
        ])
        blk["gt_pos"]         = gt_pos
        blk["gt_tm"]          = gt_key[1]
        blk["gt_total_kWh"]   = best_E
        blk["gt_savings_pct"] = gt_row["pct_savings"]
        blk["tied_tm_set"]    = tied_tm   # Tm values at gt_pos with identical total_kWh
    return dict(out)


# ══════════════════════════════════════════════════════════════════════════════
# Per-city scoring
# ══════════════════════════════════════════════════════════════════════════════
def _snap_tm(tm: float) -> float:
    return min(MELT_TEMPERATURE_GRID, key=lambda g: abs(g - tm))


def score_one(city: str, pred_pos: int | None, pred_tm: float | None,
              lookup: dict[str, Any]) -> dict[str, Any]:
    blk = lookup[city]
    gt_pos = blk["gt_pos"]; gt_tm = blk["gt_tm"]
    gt_save = blk["gt_savings_pct"]

    # Defensive snap (Judge should have caught off-grid already)
    used_pos = pred_pos
    used_tm  = pred_tm
    snapped  = False
    if pred_pos is None or pred_pos not in POSITION_GRID:
        return {"status": "missing_pos", "raw_pos": pred_pos, "raw_tm": pred_tm,
                "gt_pos": gt_pos, "gt_tm": gt_tm}
    if pred_tm is None:
        return {"status": "missing_tm", "raw_pos": pred_pos, "raw_tm": pred_tm,
                "gt_pos": gt_pos, "gt_tm": gt_tm}
    if pred_tm not in MELT_TEMPERATURE_GRID:
        used_tm = _snap_tm(pred_tm)
        snapped = True

    row = blk["grid"].get((used_pos, used_tm))
    if row is None:
        return {"status": "lookup_miss", "raw_pos": pred_pos, "raw_tm": pred_tm,
                "gt_pos": gt_pos, "gt_tm": gt_tm}

    e_pred  = row["total_kWh"]
    e_gt    = blk["gt_total_kWh"]
    s_pred  = row["pct_savings"]

    savings_capture = (s_pred / gt_save) if gt_save else 0.0
    capture_loss    = max(0.0, 1.0 - savings_capture)   # headline, lower is better

    # ── Position: strict argmin only (no tied pos — empirically the same
    # pos dominates the optimum across all 18 cities).
    # ── Tm: tied to the set of Tm values at gt_pos whose total_kWh
    # equals the best (handles equal-energy plateaus along the Tm axis).
    tied_tm = blk.get("tied_tm_set") or [gt_tm]
    pos_dist      = abs(used_pos - gt_pos)
    exact_pos_hit = used_pos == gt_pos
    near_pos_hit  = pos_dist <= 1
    tm_dist       = min(abs(used_tm - t) for t in tied_tm)
    tm_hit_2c     = tm_dist <= 2
    tm_hit_4c     = tm_dist <= 4
    joint_strict  = exact_pos_hit and tm_hit_2c
    joint_loose   = near_pos_hit  and tm_hit_4c

    return {
        "status":           "ok",
        "raw_pos":          pred_pos,
        "raw_tm":           pred_tm,
        "used_pos":         used_pos,
        "used_tm":          used_tm,
        "snapped":          snapped,
        "gt_pos":           gt_pos,
        "gt_tm":            gt_tm,
        "tied_tm_set":      tied_tm,
        "tied_tm_set_size": len(tied_tm),
        "e_pred_kWh":       e_pred,
        "e_gt_kWh":         e_gt,
        "energy_gap_kWh":   e_pred - e_gt,
        "savings_pct_pred": s_pred,
        "savings_pct_gt":   gt_save,
        "savings_capture":  savings_capture,
        "capture_loss":     capture_loss,
        "is_acceptable":    capture_loss <= 0.10,    # ≤ 10% loss
        "exact_pos_hit":    exact_pos_hit,
        "near_pos_hit":     near_pos_hit,
        "tm_hit_2c":        tm_hit_2c,
        "tm_hit_4c":        tm_hit_4c,
        "pos_dist":         pos_dist,
        "tm_dist":          tm_dist,
        "joint_strict":     joint_strict,
        "joint_loose":      joint_loose,
    }
