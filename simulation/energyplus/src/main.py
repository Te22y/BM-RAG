"""Main entry point: orchestrate locate EPW -> build IDFs -> run -> summarize.

v3.1: EPWs are located on local disk (no network download).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure relative imports work when called as `python src/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from case_generator import filter_cases, generate_cases
from config import Config, load_config
from epw_fetcher import locate_epw
from idf_builder import build_case_idf
from postprocess import build_summary
from runner import run_cases
from utils import PROJECT_ROOT, ensure_dirs, get_logger

log = get_logger("main", log_file=PROJECT_ROOT / "logs" / "main.log")


def _parse_zone_filter(arg: str | None) -> List[str] | None:
    if not arg:
        return None
    return [z.strip().lower() for z in arg.split(",") if z.strip()]


def _resolve_city_epws(cfg: Config, zone_filter: List[str] | None
                        ) -> Tuple[Dict[Tuple[str, str], Path], List[Tuple[str, str, str]]]:
    """For each (zone, city), look up EPW. Returns (found_map, missing_list)."""
    found: Dict[Tuple[str, str], Path] = {}
    missing: List[Tuple[str, str, str]] = []
    for zone_name, ze in cfg.climate_zones.items():
        if zone_filter and zone_name not in zone_filter:
            continue
        for city in ze.cities:
            p = locate_epw(cfg.epw_root, city.epw_filename)
            if p is not None:
                found[(zone_name, city.name)] = p
            else:
                missing.append((zone_name, city.name, city.epw_filename))
    return found, missing


def cmd_check_env(cfg: Config, zone_filter: List[str] | None) -> int:
    import eppy
    from eppy.modeleditor import IDF
    log.info(f"Project root: {PROJECT_ROOT}")
    log.info(f"EnergyPlus exe: {cfg.energyplus.exe_path} (exists={cfg.energyplus.exe_path.exists()})")
    log.info(f"IDD path:       {cfg.energyplus.idd_path} (exists={cfg.energyplus.idd_path.exists()})")
    log.info(f"EPW root:       {cfg.epw_root}")
    log.info(f"eppy version:   {eppy.__version__}")

    IDF.setiddname(str(cfg.energyplus.idd_path))

    zones = cfg.climate_zones if not zone_filter else \
            {k: v for k, v in cfg.climate_zones.items() if k in zone_filter}

    # Verify IDF templates
    for zone_name, ze in zones.items():
        log.info(f"--- Zone: {zone_name} | template: {ze.template_idf.name}")
        idf = IDF(str(ze.template_idf))
        ver = idf.idfobjects["VERSION"][0].Version_Identifier
        cons_names = {c.Name for c in idf.idfobjects["CONSTRUCTION"]}
        for tc in cfg.target_constructions:
            if tc not in cons_names:
                log.error(f"  target_construction missing in {ze.template_idf.name}: {tc}")
                return 1
        log.info(f"  IDF version {ver}; all target constructions present")

    # EPW locator
    found, missing = _resolve_city_epws(cfg, zone_filter)
    log.info(f"--- EPW location summary: {len(found)} found, {len(missing)} missing")
    for (zone, city), p in sorted(found.items()):
        log.info(f"  OK  {zone:14s} {city:14s} -> {p.relative_to(PROJECT_ROOT)}")
    for zone, city, fname in missing:
        log.warning(f"  ---- MISSING {zone:14s} {city:14s} (expected: {fname})")

    if not found:
        log.error("No EPW files located. Place .epw files under inputs/epw/.")
        return 1
    log.info(f"Environment check OK. Will run {len(found)} cities.")
    return 0


def cmd_run(cfg: Config, args) -> int:
    zone_filter = _parse_zone_filter(args.zones)

    # Locate EPWs first; restrict cases to (zone, city) pairs with EPW present
    found, missing = _resolve_city_epws(cfg, zone_filter)
    if missing:
        log.warning(f"{len(missing)} cities have no EPW and will be skipped:")
        for z, c, f in missing:
            log.warning(f"  skip {z}/{c}  (expected {f})")
    if not found:
        log.error("No EPW files located.")
        return 1

    # Generate cases, keep cases with EPW present
    cases_all = generate_cases(cfg, zone_filter=zone_filter)
    cases_with_epw = [c for c in cases_all if (c.zone, c.city) in found]
    # cases_to_run = subset filter applied; cases_to_summarize = all w/ EPW (for full csv)
    cases_to_run = filter_cases(cases_with_epw, args.cases_subset)
    cases_to_summarize = cases_with_epw
    log.info(f"Cases to RUN: {len(cases_to_run)} "
             f"(summary will cover {len(cases_to_summarize)} cases with EPW)")
    if not cases_to_run:
        log.error("No cases match filters")
        return 2
    cases = cases_to_run

    gen_dir = PROJECT_ROOT / "generated_idfs"
    results_dir = PROJECT_ROOT / "results" / "raw"
    logs_dir = PROJECT_ROOT / "logs"
    summary_path = PROJECT_ROOT / "results" / "summary.csv"
    ensure_dirs(gen_dir, results_dir, logs_dir, summary_path.parent)

    # 1. Build IDFs
    log.info("Building IDFs ...")
    idf_map: Dict[str, Path] = {}
    for c in cases:
        idf_path = build_case_idf(cfg, c, gen_dir)
        idf_map[c.case_id] = idf_path
    log.info(f"Generated {len(idf_map)} IDF files in {gen_dir}")

    if args.dry_run:
        log.info("--dry-run: skipping simulation and postprocess")
        return 0

    # 2. Build case_run_map: case_id -> (idf, epw)
    case_run_map: Dict[str, Tuple[Path, Path]] = {}
    for c in cases:
        case_run_map[c.case_id] = (idf_map[c.case_id], found[(c.zone, c.city)])

    # 3. Run
    results = run_cases(
        case_run_map=case_run_map,
        exe_path=cfg.energyplus.exe_path,
        results_root=results_dir,
        logs_dir=logs_dir,
        num_workers=cfg.execution.num_workers,
        timeout_sec=cfg.execution.per_case_timeout_sec,
        retry_on_fail=cfg.execution.retry_on_fail,
        keep_eso=args.keep_eso,
    )
    n_ok = sum(1 for r in results if r.ok)
    log.info(f"Simulation complete: {n_ok}/{len(results)} succeeded")
    for r in results:
        if not r.ok:
            log.error(f"  FAIL {r.case_id}: {r.message}  (log: logs/{r.case_id}.log)")

    # 4. Postprocess — summarize EVERY case-with-EPW from disk, not just this run's subset.
    #    Cases without raw output (e.g. from a prior partial run) are tagged status='missing_output'.
    log.info("Building summary.csv (full set including prior results) ...")
    build_summary(cases_to_summarize, results_dir, summary_path,
                  band_width=cfg.pcm.band_width)

    return 0 if n_ok == len(results) else 3


def main() -> int:
    parser = argparse.ArgumentParser(description="EnergyPlus PCM wall-layer sweep (v3.1 local EPW)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--check-env", action="store_true",
                        help="Validate environment + EPW + IDFs, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate IDFs without running EnergyPlus")
    parser.add_argument("--zones", type=str, default=None,
                        help="Comma-separated zone names (e.g. 'tropical,oceanic'). Default = all")
    parser.add_argument("--cases-subset", type=str, default=None,
                        help="fnmatch pattern to filter case_id (e.g. 'oceanic_seattle_*')")
    parser.add_argument("--keep-eso", action="store_true", default=False,
                        help="Keep all EnergyPlus output files (default: delete .eso/.mtr to save disk)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.check_env:
        return cmd_check_env(cfg, _parse_zone_filter(args.zones))
    return cmd_run(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
