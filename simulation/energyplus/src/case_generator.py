"""Generate the case list: for each (zone, city), baseline + (positions x melt_temps).

v4: adds PSM (phase-stable material) twin cases for Method B latent-heat isolation.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import List, Optional

from config import Config


@dataclass(frozen=True)
class Case:
    case_id: str
    zone: str
    city: str
    is_baseline: bool
    position: Optional[int]
    melt_temp: Optional[float]
    is_psm: bool = False  # v4: phase-stable twin (no latent heat)


def generate_cases(cfg: Config,
                    zone_filter: Optional[List[str]] = None) -> List[Case]:
    """Generate all cases across configured zones x cities.

    zone_filter: if provided, only emit cases for these zone names.
    """
    cases: List[Case] = []
    psm_pos = set(cfg.psm.positions) if cfg.psm.enabled else set()
    psm_tm = set(float(t) for t in cfg.psm.melt_temps) if cfg.psm.enabled else set()

    for zone_name, ze in cfg.climate_zones.items():
        if zone_filter and zone_name not in zone_filter:
            continue
        for city in ze.cities:
            # baseline first
            cases.append(Case(
                case_id=f"{zone_name}_{city.name}_baseline",
                zone=zone_name, city=city.name,
                is_baseline=True, position=None, melt_temp=None,
            ))
            # PCM sweep
            for pos in cfg.pcm.insert_positions:
                for tm in cfg.pcm.melt_temps:
                    cases.append(Case(
                        case_id=f"{zone_name}_{city.name}_pos{pos}_tm{int(tm)}",
                        zone=zone_name, city=city.name,
                        is_baseline=False, position=pos, melt_temp=float(tm),
                    ))
                    # PSM twin if (pos, tm) is in the configured subset
                    if pos in psm_pos and float(tm) in psm_tm:
                        cases.append(Case(
                            case_id=f"{zone_name}_{city.name}_pos{pos}_tm{int(tm)}_psm",
                            zone=zone_name, city=city.name,
                            is_baseline=False, position=pos, melt_temp=float(tm),
                            is_psm=True,
                        ))
    return cases


def filter_cases(cases: List[Case], subset_pattern: Optional[str]) -> List[Case]:
    """fnmatch filter. subset_pattern can be comma-separated list of patterns (OR)."""
    if not subset_pattern:
        return cases
    patterns = [p.strip() for p in subset_pattern.split(",") if p.strip()]
    return [c for c in cases
            if any(fnmatch.fnmatch(c.case_id, p) for p in patterns)]
