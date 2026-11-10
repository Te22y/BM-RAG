"""For each city's best PCM case (from summary.csv), run the matching PSM twin
and report the latent-heat contribution.

Output: results/best_psm_comparison.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from case_generator import Case
from config import load_config
from epw_fetcher import locate_epw
from idf_builder import build_case_idf
from postprocess import _compute_latent_utilization, _extract_annual_energy
from runner import run_cases
from utils import PROJECT_ROOT, ensure_dirs, get_logger

log = get_logger("best_psm", log_file=PROJECT_ROOT / "logs" / "best_psm.log")


def pick_best_per_city(summary_csv: Path,
                        cities: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    pcm = df[(~df.is_baseline) & (~df.is_psm)].copy()
    if cities:
        pcm = pcm[pcm.city.isin(cities)]
    idx = pcm.groupby(["zone", "city"])["pct_savings"].idxmax()
    return pcm.loc[idx].reset_index(drop=True)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", type=str, default=None,
                    help="comma-separated city names; default = all")
    args = ap.parse_args()
    city_filter = [c.strip() for c in args.cities.split(",")] if args.cities else None

    cfg = load_config()
    summary_csv = PROJECT_ROOT / "results" / "summary.csv"
    gen_dir = PROJECT_ROOT / "generated_idfs"
    results_dir = PROJECT_ROOT / "results" / "raw"
    logs_dir = PROJECT_ROOT / "logs"
    ensure_dirs(gen_dir, results_dir, logs_dir)

    best = pick_best_per_city(summary_csv, cities=city_filter)
    log.info(f"Picked {len(best)} best PCM cases (one per city)"
             + (f" filter={city_filter}" if city_filter else ""))

    # Build PSM twin cases with case_id suffix `_psm_best` to avoid clashing
    # with any pre-existing PSM twins from the configured psm_subset.
    psm_cases = [
        Case(
            case_id=f"{row.zone}_{row.city}_pos{int(row.position)}_tm{int(row.melt_temp)}_psm_best",
            zone=row.zone, city=row.city, is_baseline=False,
            position=int(row.position), melt_temp=float(row.melt_temp),
            is_psm=True,
        )
        for row in best.itertuples()
    ]

    # Build IDFs and resolve EPWs in one pass
    case_run_map = {}
    for c in psm_cases:
        idf_path = build_case_idf(cfg, c, gen_dir)
        epw_filename = next(
            city.epw_filename for city in cfg.climate_zones[c.zone].cities
            if city.name == c.city
        )
        epw_path = locate_epw(cfg.epw_root, epw_filename)
        case_run_map[c.case_id] = (idf_path, epw_path)
    log.info(f"Built {len(case_run_map)} PSM IDFs")

    # Run PSM cases
    results = run_cases(
        case_run_map=case_run_map,
        exe_path=cfg.energyplus.exe_path,
        results_root=results_dir,
        logs_dir=logs_dir,
        num_workers=cfg.execution.num_workers,
        timeout_sec=cfg.execution.per_case_timeout_sec,
        retry_on_fail=cfg.execution.retry_on_fail,
        keep_eso=False,
    )
    n_ok = sum(1 for r in results if r.ok)
    log.info(f"PSM runs done: {n_ok}/{len(results)} succeeded")

    # Build comparison table
    rows = []
    baseline_csv = pd.read_csv(summary_csv)
    bw = cfg.pcm.band_width
    for c, pcm_row in zip(psm_cases, best.itertuples()):
        psm_out = results_dir / c.case_id
        psm_energy = _extract_annual_energy(psm_out)
        psm_total = psm_energy["total_kWh"]
        psm_util_frac, psm_util_hrs = _compute_latent_utilization(
            psm_out, c.melt_temp, bw
        )
        base_total = baseline_csv[
            (baseline_csv.zone == c.zone)
            & (baseline_csv.city == c.city)
            & (baseline_csv.is_baseline)
        ]["total_kWh"].iloc[0]

        pcm_total = pcm_row.total_kWh
        # Latent savings = how much extra PCM saves over its no-latent twin
        latent_kWh = (psm_total - pcm_total) if psm_total is not None else None
        # Sensible-only savings (PSM vs baseline)
        sensible_kWh = (base_total - psm_total) if psm_total is not None else None
        # Total PCM savings (PCM vs baseline)
        total_savings_kWh = base_total - pcm_total
        # Share of total PCM savings attributable to latent heat
        latent_share_pct = (
            100.0 * latent_kWh / total_savings_kWh
            if (latent_kWh is not None and total_savings_kWh)
            else None
        )

        rows.append({
            "zone": c.zone, "city": c.city,
            "pos": c.position, "Tm_C": int(c.melt_temp),
            "baseline_kWh": round(base_total, 1),
            "pcm_kWh": round(pcm_total, 1),
            "psm_kWh": round(psm_total, 1) if psm_total is not None else None,
            "sensible_savings_kWh": round(sensible_kWh, 1) if sensible_kWh is not None else None,
            "latent_savings_kWh": round(latent_kWh, 1) if latent_kWh is not None else None,
            "total_savings_kWh": round(total_savings_kWh, 1),
            "pct_savings_pcm": round(pcm_row.pct_savings, 2),
            "pct_savings_psm": round(100.0 * sensible_kWh / base_total, 2) if sensible_kWh is not None else None,
            "latent_share_pct": round(latent_share_pct, 1) if latent_share_pct is not None else None,
            "latent_utilization_proxy": round(psm_util_frac, 3) if psm_util_frac is not None else None,
        })

    out = pd.DataFrame(rows).sort_values("latent_share_pct", ascending=False, na_position="last")
    out_path = PROJECT_ROOT / "results" / "best_psm_comparison.csv"
    out.to_csv(out_path, index=False)
    log.info(f"Wrote {out_path}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
