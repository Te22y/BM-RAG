"""
Extract wall_assembly per city from baseline IDFs and emit a fixed evaluation problem set.

Output: experiments/problem_sets/problem_set_18city.json

Each record contains:
  - city                (string, only the city name — agent must derive climate)
  - zone                (metadata only, NOT shown to agents)
  - epw_filename        (metadata only, NOT shown to agents)
  - wall_assembly_text  (human-readable description fed to AnalystInput)
  - wall_layers         (structured list — parsed Material objects)
  - analyst_input       (dict ready to be passed to AnalystInput(**))
  - ground_truth        (read from results/summary.csv)
       {gt_pos, gt_tm, gt_total_kWh, gt_baseline_total_kWh,
        gt_savings_pct, response_surface_top3}

The "ground_truth" block is included so the evaluator can run from this
single file without re-reading summary.csv.

IDF format is rigid; we use simple regex parsing instead of eppy.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

# ── paths ────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
ROOT       = HERE.parents[1]
EP_DIR     = ROOT / "simulation" / "energyplus"
IDF_DIR    = EP_DIR / "generated_idfs"
SUMMARY_CSV= EP_DIR / "results" / "summary.csv"
CONFIG_YML = EP_DIR / "config.yaml"

OUT_PATH   = HERE / "problem_set_18city.json"

TARGET_CONSTRUCTION = "nonres_ext_wall_grd"


# ══════════════════════════════════════════════════════════════════════════════
# IDF parsing
# ══════════════════════════════════════════════════════════════════════════════
# An IDF "object" is: TypeName, comma-separated fields, terminated by ';'.
# Fields are on their own line with a trailing "!- field-name" comment.
# We split on lines, strip comments, and collect raw fields per object.

_OBJ_START_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_:]*)\s*,\s*$")
_FIELD_RE     = re.compile(r"^\s*([^,!;]*?)\s*[,;]")


def _parse_idf_objects(idf_text: str) -> list[tuple[str, list[str]]]:
    """Return a list of (object_type, [field_value, ...]) tuples.

    Liberal parser tolerant of inline comments after `!-`."""
    objects: list[tuple[str, list[str]]] = []
    lines = idf_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].split("!")[0].rstrip()    # strip comment
        i += 1
        if not line.strip():
            continue
        # Look for "Type," start of object
        m = _OBJ_START_RE.match(line)
        if not m:
            continue
        obj_type = m.group(1)
        fields: list[str] = []
        # Consume until we see a field terminated by ';'
        while i < len(lines):
            raw = lines[i]
            i += 1
            content = raw.split("!")[0]
            # find first ',' or ';'
            for j, ch in enumerate(content):
                if ch in ",;":
                    fields.append(content[:j].strip())
                    if ch == ";":
                        break
                    else:
                        # ignore trailing content after comma on same line
                        rest = content[j+1:].strip()
                        if rest and not rest.startswith("!"):
                            # second field on the same line — uncommon but safe
                            content = rest
                            j = -1
                        break
            else:
                # no terminator on this line (e.g. blank), continue
                continue
            if ";" in content:
                break
        objects.append((obj_type, fields))
    return objects


def _index_materials(objects: list[tuple[str, list[str]]]) -> dict[str, dict[str, Any]]:
    """Build {material_name: {type, fields_dict}}."""
    out: dict[str, dict[str, Any]] = {}
    for obj_type, fields in objects:
        ot = obj_type.lower()
        if ot == "material":
            # Material: Name, Roughness, Thickness, Conductivity, Density, Specific Heat,
            #           [Thermal Absorptance], [Solar Absorptance], [Visible Absorptance]
            if len(fields) < 6:
                continue
            name = fields[0]
            out[name] = {
                "object_type":          "Material",
                "name":                 name,
                "roughness":            fields[1] if len(fields) > 1 else "",
                "thickness_m":          float(fields[2]) if fields[2] else None,
                "conductivity_W_mK":    float(fields[3]) if fields[3] else None,
                "density_kg_m3":        float(fields[4]) if fields[4] else None,
                "specific_heat_J_kgK":  float(fields[5]) if fields[5] else None,
            }
        elif ot == "material:nomass":
            # Material:NoMass: Name, Roughness, Thermal Resistance, [thermal/solar/visible abs]
            if len(fields) < 3:
                continue
            name = fields[0]
            out[name] = {
                "object_type":  "Material:NoMass",
                "name":         name,
                "roughness":    fields[1] if len(fields) > 1 else "",
                "r_value_m2K_W": float(fields[2]) if fields[2] else None,
            }
    return out


def _find_construction_layers(objects: list[tuple[str, list[str]]],
                              target_name: str) -> Optional[list[str]]:
    """Construction: Name, OutsideLayer, Layer2, Layer3, ..."""
    target_lc = target_name.lower()
    for obj_type, fields in objects:
        if obj_type.lower() != "construction":
            continue
        if not fields:
            continue
        if fields[0].lower() == target_lc:
            return [f for f in fields[1:] if f]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Per-city extraction
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class WallLayer:
    name:                str
    object_type:         str
    thickness_m:         Optional[float] = None
    conductivity_W_mK:   Optional[float] = None
    density_kg_m3:       Optional[float] = None
    specific_heat_J_kgK: Optional[float] = None
    r_value_m2K_W:       Optional[float] = None


def _render_layer(layer: WallLayer) -> str:
    if layer.object_type == "Material:NoMass":
        return f"{layer.name} (NoMass, R={layer.r_value_m2K_W:.2f} m²K/W)"
    parts = [layer.name]
    if layer.thickness_m is not None:
        parts.append(f"{layer.thickness_m*1000:.0f}mm")
    if layer.conductivity_W_mK is not None:
        parts.append(f"k={layer.conductivity_W_mK:.2g} W/mK")
    return " ".join(parts)


# ── Setpoint schedule parsing ────────────────────────────────────────────────
# The ASHRAE 90.1 OfficeSmall template uses two Schedule:Compact objects
# named "HTGSETP_SCH_NO_OPTIMUM" and "CLGSETP_SCH_NO_OPTIMUM" that drive
# every zone's DualSetpoint thermostat. We extract the occupied-hour and
# setback temperatures so the Reasoner knows the actual indoor swing band.

_SETP_NUMBER_RE = re.compile(r"^\s*([0-9]+\.[0-9]+|[0-9]+),\s*!-")

def _extract_setpoint_values(idf_text: str, schedule_name: str) -> list[float]:
    """Pull all numeric temperature values out of a Schedule:Compact block."""
    target = schedule_name.strip().lower()
    lines = idf_text.splitlines()
    vals: list[float] = []
    in_target = False
    for i, line in enumerate(lines):
        if line.strip() == "Schedule:Compact,":
            # peek next line for name
            if i + 1 < len(lines):
                nxt = lines[i+1].split("!")[0].strip().rstrip(",").strip()
                if nxt.lower() == target:
                    in_target = True
                else:
                    in_target = False
            continue
        if not in_target:
            continue
        m = _SETP_NUMBER_RE.match(line)
        if m:
            vals.append(float(m.group(1)))
        if line.rstrip().endswith(";"):
            in_target = False
            if vals:
                break
    return vals


def extract_setpoints(idf_path: Path) -> dict[str, float]:
    """Return {cooling_setpoint_c, cooling_setback_c, heating_setpoint_c,
    heating_setback_c}.  Occupied = min cooling / max heating;
    setback = max cooling / min heating."""
    text = idf_path.read_text(encoding="utf-8", errors="replace")
    clg = _extract_setpoint_values(text, "CLGSETP_SCH_NO_OPTIMUM")
    htg = _extract_setpoint_values(text, "HTGSETP_SCH_NO_OPTIMUM")
    out: dict[str, float] = {}
    if clg:
        out["cooling_setpoint_c"] = round(min(clg), 2)
        out["cooling_setback_c"]  = round(max(clg), 2)
    if htg:
        out["heating_setpoint_c"] = round(max(htg), 2)
        out["heating_setback_c"]  = round(min(htg), 2)
    return out


# ── HVAC operation / occupancy time schedule ─────────────────────────────────
# Read the WeekDay block of the cooling-setpoint Schedule:Compact to derive
# (occupied_start, occupied_end). The pattern in ASHRAE OfficeSmall is:
#   Until: 6:00  <setback>
#   Until: 7:00  <occupied>    ← occupied_start = 06:00 (precool)
#   Until: 19:00 <occupied>    ← occupied_end   = 19:00 (heating side)
#   Until: 24:00 <setback>
# Weekends / Holidays are all setback. We return a short summary string.

_SCH_TIME_RE  = re.compile(r"Until:\s*([0-9]{1,2}):([0-9]{2})", re.IGNORECASE)
_SCH_DAY_RE   = re.compile(r"For:\s*([A-Za-z0-9]+)",            re.IGNORECASE)

def _extract_schedule_block(idf_text: str, schedule_name: str) -> list[str]:
    """Return the raw lines of a Schedule:Compact body (after the name line)."""
    target = schedule_name.strip().lower()
    lines = idf_text.splitlines()
    out: list[str] = []
    in_target = False
    for i, line in enumerate(lines):
        if line.strip() == "Schedule:Compact,":
            nxt = lines[i+1].split("!")[0].strip().rstrip(",").strip() if i+1 < len(lines) else ""
            in_target = (nxt.lower() == target)
            continue
        if not in_target:
            continue
        out.append(line)
        if line.rstrip().endswith(";"):
            break
    return out


def extract_schedule_summary(idf_path: Path) -> dict[str, Any]:
    """Summarise the cooling-setpoint weekday schedule into
    (occupied_start, occupied_end, occupied_hours_per_weekday,
     active_days_per_week, summary_text)."""
    text = idf_path.read_text(encoding="utf-8", errors="replace")
    lines = _extract_schedule_block(text, "CLGSETP_SCH_NO_OPTIMUM")
    if not lines:
        return {}

    # Parse WeekDays block: collect transitions (hh:mm → numeric setpoint).
    weekday_transitions: list[tuple[int, float]] = []  # (minutes, value)
    current_day = None
    pending_time: Optional[int] = None
    for raw in lines:
        line = raw.split("!")[0].strip().rstrip(",").rstrip(";").strip()
        if not line: continue
        m_day = _SCH_DAY_RE.match(line)
        if m_day:
            current_day = m_day.group(1).lower()
            continue
        m_t = _SCH_TIME_RE.search(line)
        if m_t:
            pending_time = int(m_t.group(1)) * 60 + int(m_t.group(2))
            continue
        # Numeric setpoint value
        try:
            val = float(line)
        except ValueError:
            continue
        if pending_time is not None and current_day == "weekdays":
            weekday_transitions.append((pending_time, val))
            pending_time = None

    if not weekday_transitions:
        return {}

    # Occupied = the band where setpoint is the LOW value (cooling setpoint).
    occ_value = min(v for _, v in weekday_transitions)
    occupied_blocks: list[tuple[int, int]] = []
    prev_t = 0
    for t, v in weekday_transitions:
        if abs(v - occ_value) < 1e-3:
            occupied_blocks.append((prev_t, t))
        prev_t = t
    if not occupied_blocks:
        return {}

    occ_start = min(b[0] for b in occupied_blocks)
    occ_end   = max(b[1] for b in occupied_blocks)
    occ_hours = sum((b[1] - b[0]) for b in occupied_blocks) / 60.0

    def _hm(mins: int) -> str:
        return f"{mins//60:02d}:{mins%60:02d}"

    summary = (f"WeekDays {_hm(occ_start)}-{_hm(occ_end)} occupied "
               f"({occ_hours:.0f} h/day at cooling setpoint); "
               f"nights, weekends and holidays at setback")
    return {
        "occupied_start_hhmm":      _hm(occ_start),
        "occupied_end_hhmm":        _hm(occ_end),
        "occupied_hours_per_weekday": occ_hours,
        "active_days_per_week":     5,
        "summary":                  summary,
    }


def extract_wall(idf_path: Path) -> tuple[list[WallLayer], str]:
    objs = _parse_idf_objects(idf_path.read_text(encoding="utf-8", errors="replace"))
    mats = _index_materials(objs)
    layer_names = _find_construction_layers(objs, TARGET_CONSTRUCTION)
    if not layer_names:
        raise RuntimeError(f"Construction {TARGET_CONSTRUCTION!r} not found in {idf_path}")

    layers: list[WallLayer] = []
    for n in layer_names:
        m = mats.get(n)
        if not m:
            raise RuntimeError(f"Material {n!r} referenced by Construction not found in {idf_path}")
        layers.append(WallLayer(
            name                = m["name"],
            object_type         = m["object_type"],
            thickness_m         = m.get("thickness_m"),
            conductivity_W_mK   = m.get("conductivity_W_mK"),
            density_kg_m3       = m.get("density_kg_m3"),
            specific_heat_J_kgK = m.get("specific_heat_J_kgK"),
            r_value_m2K_W       = m.get("r_value_m2K_W"),
        ))

    text = " | ".join(_render_layer(l) for l in layers)
    return layers, f"exterior → {text} → interior"


# ══════════════════════════════════════════════════════════════════════════════
# Ground truth from summary.csv
# ══════════════════════════════════════════════════════════════════════════════
def load_ground_truth() -> dict[str, dict[str, Any]]:
    by_city: dict[str, dict[str, Any]] = {}
    baseline_by_city: dict[str, float] = {}
    grid_by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            if row.get("is_psm", "").lower() in ("true", "1"):
                continue
            city = row["city"]
            zone = row["zone"]
            is_base = row["is_baseline"].lower() in ("true", "1")
            total = float(row["total_kWh"]) if row["total_kWh"] else None
            if total is None:
                continue
            if is_base:
                baseline_by_city[city] = total
                by_city.setdefault(city, {})["zone"] = zone
                by_city[city]["baseline_total_kWh"] = total
                continue
            grid_by_city[city].append({
                "pos":      int(row["position"]),
                "tm":       float(row["melt_temp"]),
                "total_kWh": total,
                "pct_savings": float(row["pct_savings"]) if row["pct_savings"] else None,
                "delta_kWh":   float(row["baseline_delta_total_kWh"]) if row["baseline_delta_total_kWh"] else None,
                "latent_utilization": float(row["latent_utilization"]) if row["latent_utilization"] else None,
                "discomfort_hours":   float(row["discomfort_hours"]) if row["discomfort_hours"] else None,
            })

    for city, rows in grid_by_city.items():
        rows.sort(key=lambda r: r["total_kWh"])
        best = rows[0]
        top5 = rows[:5]
        d = by_city.setdefault(city, {})
        d["gt_pos"]            = best["pos"]
        d["gt_tm"]             = best["tm"]
        d["gt_total_kWh"]      = best["total_kWh"]
        d["gt_savings_pct"]    = best["pct_savings"]
        d["gt_delta_kWh"]      = best["delta_kWh"]
        d["gt_discomfort_h"]   = best["discomfort_hours"]
        d["response_surface_top5"] = top5
        d["grid_row_count"]    = len(rows)
        d["worst_total_kWh"]   = rows[-1]["total_kWh"]
    return by_city


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def build():
    cfg = yaml.safe_load(CONFIG_YML.read_text(encoding="utf-8"))
    # city → (zone, epw)
    city_meta: dict[str, dict[str, str]] = {}
    for zone, blk in cfg["climate_zones"].items():
        for c in blk["cities"]:
            city_meta[c["name"]] = {"zone": zone, "epw_filename": c["epw_filename"]}

    gt = load_ground_truth()

    records: list[dict[str, Any]] = []
    for idf_path in sorted(IDF_DIR.glob("*_baseline.idf")):
        stem = idf_path.stem.replace("_baseline", "")
        # filename pattern: {zone}_{city}
        zone, _, city = stem.partition("_")
        if city not in city_meta:
            print(f"  [warn] city {city!r} not in config.yaml — skipping {idf_path.name}")
            continue

        layers, wall_text = extract_wall(idf_path)
        sp = extract_setpoints(idf_path)
        sched = extract_schedule_summary(idf_path)

        # Embed setpoints AND operation schedule directly into building_type.
        # The Reasoner needs both the temperature band AND the time-of-day
        # pattern to derive the indoor temperature envelope.
        building_desc = (
            "small office, exterior wall (Perimeter_ZN_1..4, sun-exposed); "
            "HVAC dual-setpoint thermostat — "
            f"occupied: cooling setpoint = {sp.get('cooling_setpoint_c')}°C, "
            f"heating setpoint = {sp.get('heating_setpoint_c')}°C; "
            f"setback: cooling = {sp.get('cooling_setback_c')}°C, "
            f"heating = {sp.get('heating_setback_c')}°C. "
            f"Operation schedule: {sched.get('summary','(schedule not parsed)')}."
        )

        analyst_input = {
            "climate":            city,                       # ★ only city name
            "building_type":      building_desc,
            "wall_construction":  wall_text,
            "pcm_parameters":     ("organic PCM, latent heat=200 kJ/kg, "
                                   "thickness=20 mm, phase-transition band = 4°C; "
                                   "choose Tm from "
                                   "{12,14,16,18,20,22,24,26,28,30,32} °C"),
            "goal":               "minimize total annual building energy (heating + cooling)",
            "orientation":        None,
            "season":             None,
            "optimization_target": "min_total_energy",
        }

        rec = {
            "city":               city,
            "zone":               city_meta[city]["zone"],            # metadata only
            "epw_filename":       city_meta[city]["epw_filename"],     # metadata only
            "wall_assembly_text": wall_text,
            "wall_layers":        [asdict(l) for l in layers],
            "setpoints":          sp,
            "schedule":           sched,
            "analyst_input":      analyst_input,
            "ground_truth":       gt.get(city, {}),
        }
        records.append(rec)

    OUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"Wrote {len(records)} records → {OUT_PATH}")
    # Brief stdout summary
    for r in records:
        g = r.get("ground_truth", {})
        r_values = [layer["r_value_m2K_W"] for layer in r["wall_layers"] if layer.get("r_value_m2K_W")]
        print(f"  {r['city']:15s} zone={r['zone']:14s}  "
              f"gt=(pos{g.get('gt_pos','?')}, Tm{g.get('gt_tm','?')}°C) "
              f"save={(g.get('gt_savings_pct') or 0):.2f}%  "
              f"wall_R={r_values}")


if __name__ == "__main__":
    build()
