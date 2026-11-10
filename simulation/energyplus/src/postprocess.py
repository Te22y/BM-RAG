"""Parse EnergyPlus outputs and build summary.csv.

v4: adds Method A (latent_utilization from hourly surface temperatures via SQLite)
    and Method B (psm_delta linking PCM <-> PSM twin cases).
"""
from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from case_generator import Case
from utils import get_logger

log = get_logger("postprocess")


@dataclass
class CaseSummary:
    case_id: str
    zone: str
    city: str
    is_baseline: bool
    is_psm: bool
    position: Optional[int]
    melt_temp: Optional[float]
    heating_kWh: Optional[float]
    cooling_kWh: Optional[float]
    total_kWh: Optional[float]
    discomfort_hours: Optional[float]
    baseline_delta_total_kWh: Optional[float] = None
    pct_savings: Optional[float] = None
    # Method A: fraction of year where the PCM operating band [Tm - bw/2, Tm + bw/2]
    # overlaps the simultaneous [T_outside_face, T_inside_face] of the host wall.
    # Coarse upper-bound proxy for actual phase-change activity.
    latent_utilization: Optional[float] = None
    activation_hours: Optional[int] = None
    # Method B: delta vs the matching phase-stable twin (psm_total_kWh - this_total_kWh).
    # Positive = PCM saves more energy than its no-latent twin -> real latent benefit.
    psm_total_kWh: Optional[float] = None
    psm_delta_kWh: Optional[float] = None
    psm_pct_latent: Optional[float] = None  # 100 * psm_delta / (PSM - baseline) ; share of savings due to latent
    status: str = "ok"


# ---------- table.htm parser ----------

_HTM_TAG = re.compile(r"<[^>]+>")


def _strip_tags(html: str) -> str:
    return _HTM_TAG.sub("", html).replace("&nbsp;", " ").strip()


def _read_tbl_html(htm_path: Path) -> str:
    return htm_path.read_text(encoding="utf-8", errors="replace")


def _find_annual_end_use_table(html: str) -> Optional[Dict[str, Dict[str, float]]]:
    idx = html.find("Annual Building Utility Performance Summary")
    if idx < 0:
        return None
    eu = html.find("End Uses", idx)
    if eu < 0:
        return None
    tbl_start = html.find("<table", eu)
    tbl_end = html.find("</table>", tbl_start)
    if tbl_start < 0 or tbl_end < 0:
        return None
    table_html = html[tbl_start:tbl_end]
    rows = re.findall(r"<tr.*?>(.*?)</tr>", table_html, flags=re.S)
    parsed: List[List[str]] = []
    for r in rows:
        cells = re.findall(r"<t[dh].*?>(.*?)</t[dh]>", r, flags=re.S)
        parsed.append([_strip_tags(c) for c in cells])
    if len(parsed) < 2:
        return None
    headers = parsed[0]
    data: Dict[str, Dict[str, float]] = {}
    for row in parsed[1:]:
        if not row or not row[0]:
            continue
        label = row[0]
        cols: Dict[str, float] = {}
        for col_name, cell in zip(headers[1:], row[1:]):
            try:
                cols[col_name] = float(cell)
            except (ValueError, TypeError):
                pass
        data[label] = cols
    return data


def _extract_annual_energy(out_dir: Path) -> Dict[str, Optional[float]]:
    candidates = list(out_dir.glob("*tbl.htm"))
    if not candidates:
        return {"heating_kWh": None, "cooling_kWh": None, "total_kWh": None}
    html = _read_tbl_html(candidates[0])
    table = _find_annual_end_use_table(html)
    if not table:
        return {"heating_kWh": None, "cooling_kWh": None, "total_kWh": None}

    GJ_TO_KWH = 1_000_000 / 3600

    def row_total_gj(label: str) -> Optional[float]:
        if label not in table:
            return None
        return sum(table[label].values())

    heating_gj = row_total_gj("Heating")
    cooling_gj = row_total_gj("Cooling")
    total_gj = None
    if heating_gj is not None and cooling_gj is not None:
        total_gj = heating_gj + cooling_gj
    return {
        "heating_kWh": heating_gj * GJ_TO_KWH if heating_gj is not None else None,
        "cooling_kWh": cooling_gj * GJ_TO_KWH if cooling_gj is not None else None,
        "total_kWh": total_gj * GJ_TO_KWH if total_gj is not None else None,
    }


def _extract_discomfort_hours(out_dir: Path) -> Optional[float]:
    candidates = list(out_dir.glob("*tbl.htm"))
    if not candidates:
        return None
    html = _read_tbl_html(candidates[0])
    for anchor in ("Adaptive Comfort Summary",
                   "Comfort and Setpoint Not Met Summary",
                   "Time Not Comfortable"):
        idx = html.find(anchor)
        if idx < 0:
            continue
        tbl_start = html.find("<table", idx)
        tbl_end = html.find("</table>", tbl_start)
        if tbl_start < 0 or tbl_end < 0:
            continue
        table_html = html[tbl_start:tbl_end]
        rows = re.findall(r"<tr.*?>(.*?)</tr>", table_html, flags=re.S)
        for r in rows:
            cells = re.findall(r"<t[dh].*?>(.*?)</t[dh]>", r, flags=re.S)
            txt = [_strip_tags(c) for c in cells]
            joined = " ".join(txt).lower()
            if "not" in joined and ("comfort" in joined or "acceptab" in joined):
                for cell in reversed(txt):
                    try:
                        return float(cell)
                    except ValueError:
                        continue
    return None


# ---------- Method A: latent-utilization proxy from SQLite ----------

def _compute_latent_utilization(out_dir: Path, melt_temp: float,
                                  band_width: float) -> Tuple[Optional[float], Optional[int]]:
    """Proxy for PCM activity fraction.

    For each hour, the PCM (embedded in the wall) is at some temperature
    bracketed by the simultaneous Surface Outside Face Temperature and
    Surface Inside Face Temperature. If the PCM operating band
    [Tm - bw/2, Tm + bw/2] overlaps that bracket, the hour is counted as
    "potentially active". Returns (fraction, hours) or (None, None) if data missing.

    Aggregates over the surfaces using the construction-name pattern of the
    target exterior wall (those tagged with "_pcm_" or "_psm_"), or any wall
    using nonres_ext_wall_grd as a fallback.
    """
    sql_files = list(out_dir.glob("*out.sql"))
    if not sql_files:
        return None, None
    sql_path = sql_files[0]

    t_lo = melt_temp - band_width / 2.0
    t_hi = melt_temp + band_width / 2.0

    try:
        conn = sqlite3.connect(f"file:{sql_path}?mode=ro", uri=True, timeout=10)
        cur = conn.cursor()

        # ReportDataDictionary schema (E+ 22.1): RTRDictionaryIndex/ReportDataDictionaryIndex,
        # IndexGroup, KeyValue, Name, ReportingFrequency, ...
        # We need indices for the two surface-temp variables on each exterior wall surface.
        cur.execute(
            "SELECT ReportDataDictionaryIndex, KeyValue, Name "
            "FROM ReportDataDictionary "
            "WHERE Name IN ('Surface Inside Face Temperature', "
            "               'Surface Outside Face Temperature') "
            "  AND ReportingFrequency IN ('Hourly', 'Zone Timestep')"
        )
        rows = cur.fetchall()
        if not rows:
            conn.close()
            return None, None

        # Group by KeyValue (surface name). Want surfaces whose construction includes the PCM/PSM material.
        # Simplest: take ALL surfaces that have BOTH variables present; if multiple, average across surfaces.
        # Build {surface_name: {'in': idx, 'out': idx}}
        per_surf: Dict[str, Dict[str, int]] = {}
        for idx, key, name in rows:
            slot = per_surf.setdefault(key, {})
            if name == "Surface Inside Face Temperature":
                slot["in"] = idx
            elif name == "Surface Outside Face Temperature":
                slot["out"] = idx

        # Need to restrict to exterior walls (the ones we retargeted). The construction-to-surface
        # mapping isn't in the SQL we configured (Simple option). As a coarse heuristic, we use
        # all surfaces that have BOTH temperature variables -- internal walls/floors typically
        # only have Inside Face Temp, so requiring both filters to exterior surfaces.
        exterior_surfs = [k for k, v in per_surf.items() if "in" in v and "out" in v]
        if not exterior_surfs:
            conn.close()
            return None, None

        in_idx = tuple(per_surf[s]["in"] for s in exterior_surfs)
        out_idx = tuple(per_surf[s]["out"] for s in exterior_surfs)
        all_idx = in_idx + out_idx

        # Pull all hourly values. Pair them by (TimeIndex, surface).
        placeholders = ",".join("?" * len(all_idx))
        cur.execute(
            f"SELECT TimeIndex, ReportDataDictionaryIndex, Value "
            f"FROM ReportData WHERE ReportDataDictionaryIndex IN ({placeholders})",
            all_idx,
        )
        # Map: (time, surface) -> {'in': T, 'out': T}
        # surface_of[idx] -> (surface_name, role)
        surface_of: Dict[int, Tuple[str, str]] = {}
        for s in exterior_surfs:
            surface_of[per_surf[s]["in"]] = (s, "in")
            surface_of[per_surf[s]["out"]] = (s, "out")

        per_hour_surf: Dict[Tuple[int, str], Dict[str, float]] = {}
        for time_idx, rdi, val in cur.fetchall():
            surf, role = surface_of[rdi]
            d = per_hour_surf.setdefault((time_idx, surf), {})
            d[role] = val
        conn.close()
    except sqlite3.Error as e:
        log.warning(f"sqlite read failed for {sql_path.name}: {e}")
        return None, None

    if not per_hour_surf:
        return None, None

    # Activation: this hour, any exterior surface has [min(T_out,T_in), max] overlapping [t_lo, t_hi].
    by_hour: Dict[int, bool] = {}
    for (time_idx, _surf), d in per_hour_surf.items():
        if "in" not in d or "out" not in d:
            continue
        lo_w = min(d["in"], d["out"])
        hi_w = max(d["in"], d["out"])
        active = (hi_w >= t_lo) and (lo_w <= t_hi)
        if active:
            by_hour[time_idx] = True
        else:
            by_hour.setdefault(time_idx, False)

    if not by_hour:
        return None, None
    active_hours = sum(1 for v in by_hour.values() if v)
    total_hours = len(by_hour)
    return (active_hours / total_hours, active_hours)


# ---------- per-case summarize ----------

def summarize_case(case: Case, out_dir: Path,
                   band_width: float) -> CaseSummary:
    if not out_dir.exists():
        return CaseSummary(case.case_id, case.zone, case.city, case.is_baseline,
                            case.is_psm, case.position, case.melt_temp,
                            None, None, None, None, status="missing_output")
    energy = _extract_annual_energy(out_dir)
    discomfort = _extract_discomfort_hours(out_dir)
    status = "ok" if energy["total_kWh"] is not None else "no_table_data"

    util_frac, util_hours = (None, None)
    if (not case.is_baseline) and case.melt_temp is not None:
        util_frac, util_hours = _compute_latent_utilization(
            out_dir, case.melt_temp, band_width
        )

    return CaseSummary(
        case_id=case.case_id,
        zone=case.zone,
        city=case.city,
        is_baseline=case.is_baseline,
        is_psm=case.is_psm,
        position=case.position,
        melt_temp=case.melt_temp,
        heating_kWh=energy["heating_kWh"],
        cooling_kWh=energy["cooling_kWh"],
        total_kWh=energy["total_kWh"],
        discomfort_hours=discomfort,
        latent_utilization=util_frac,
        activation_hours=util_hours,
        status=status,
    )


def build_summary(cases: List[Case], results_root: Path,
                  summary_path: Path, band_width: float = 4.0) -> List[CaseSummary]:
    summaries: List[CaseSummary] = []
    baseline_by_city: Dict[Tuple[str, str], float] = {}
    # Method B: PSM index for delta computation
    # key: (zone, city, position, melt_temp) -> psm total_kWh
    psm_by_pair: Dict[Tuple[str, str, int, float], float] = {}

    for c in cases:
        s = summarize_case(c, results_root / c.case_id, band_width)
        summaries.append(s)
        if c.is_baseline and s.total_kWh is not None:
            baseline_by_city[(c.zone, c.city)] = s.total_kWh
        if c.is_psm and s.total_kWh is not None and c.position is not None and c.melt_temp is not None:
            psm_by_pair[(c.zone, c.city, c.position, float(c.melt_temp))] = s.total_kWh

    for s in summaries:
        if s.total_kWh is None:
            continue
        # baseline delta (skip baselines themselves)
        if not s.is_baseline:
            base = baseline_by_city.get((s.zone, s.city))
            if base is not None:
                s.baseline_delta_total_kWh = s.total_kWh - base
                s.pct_savings = -100.0 * s.baseline_delta_total_kWh / base if base else None
        # Method B linkage: pair PCM cases with their PSM twin (only for cases in subset)
        if (not s.is_baseline) and (not s.is_psm) and s.position is not None and s.melt_temp is not None:
            psm_kWh = psm_by_pair.get((s.zone, s.city, s.position, float(s.melt_temp)))
            if psm_kWh is not None:
                s.psm_total_kWh = psm_kWh
                # delta: positive if PCM beats PSM (i.e., latent matters)
                s.psm_delta_kWh = psm_kWh - s.total_kWh
                # share of (PSM-baseline) sensible savings due to latent:
                base = baseline_by_city.get((s.zone, s.city))
                if base is not None and (psm_kWh - base) != 0:
                    s.psm_pct_latent = 100.0 * s.psm_delta_kWh / abs(psm_kWh - base)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    target = summary_path
    try:
        f = open(summary_path, "w", encoding="utf-8", newline="")
    except PermissionError:
        from datetime import datetime
        target = summary_path.with_name(
            f"{summary_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        log.warning(f"{summary_path.name} is locked; writing to {target.name} instead")
        f = open(target, "w", encoding="utf-8", newline="")
    with f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))
    log.info(f"summary -> {target} ({len(summaries)} rows; {len(baseline_by_city)} baselines, "
             f"{len(psm_by_pair)} PSM twins)")
    return summaries
