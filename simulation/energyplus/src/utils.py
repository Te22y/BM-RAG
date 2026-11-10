"""Common helpers: project paths, logging, PCM T-Enthalpy curve builder."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_logger(name: str = "pcm_runner", log_file: Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def build_enthalpy_curve(melt_temp: float, band_width: float, cp: float,
                          latent_heat: float, num_points: int = 16,
                          t_lo: float = -20.0, t_hi: float = 60.0
                          ) -> List[Tuple[float, float]]:
    """Build a strictly-monotonic Temperature-Enthalpy curve for EnergyPlus
    MaterialProperty:PhaseChange.

    The curve has three regions:
      - solid:  h(T) = cp * (T - t_lo)                       for T < Tm - band/2
      - phase:  h(T) = h_solid_end + (cp + L/band) * (T - Tm + band/2)
                                                              for Tm - band/2 <= T <= Tm + band/2
      - liquid: h(T) = h_phase_end + cp * (T - (Tm + band/2)) for T > Tm + band/2

    Returns list of (T, h) pairs with strictly increasing T and h.
    """
    half = band_width / 2.0
    t_solid_end = melt_temp - half
    t_liquid_start = melt_temp + half

    # Anchor t_lo / t_hi if they overlap phase region
    if t_lo > t_solid_end - 1:
        t_lo = t_solid_end - 5.0
    if t_hi < t_liquid_start + 1:
        t_hi = t_liquid_start + 5.0

    n_solid = max(2, num_points // 4)
    n_phase = max(4, num_points // 2)
    n_liquid = max(2, num_points - n_solid - n_phase)

    t_pts: List[float] = []
    # solid region: t_lo .. t_solid_end (exclusive of end to avoid dup; we'll add explicitly)
    for i in range(n_solid):
        frac = i / (n_solid - 1) if n_solid > 1 else 0.0
        t_pts.append(t_lo + frac * (t_solid_end - t_lo))
    # phase region: t_solid_end .. t_liquid_start
    for i in range(1, n_phase + 1):
        frac = i / n_phase
        t_pts.append(t_solid_end + frac * (t_liquid_start - t_solid_end))
    # liquid region: > t_liquid_start
    for i in range(1, n_liquid + 1):
        frac = i / n_liquid
        t_pts.append(t_liquid_start + frac * (t_hi - t_liquid_start))

    # Deduplicate while preserving order and ensure strict monotonicity (>= 0.01 step)
    cleaned: List[float] = []
    for t in t_pts:
        if not cleaned or t - cleaned[-1] > 0.01:
            cleaned.append(t)

    # Enthalpy: piecewise linear
    h_at_solid_end = cp * (t_solid_end - t_lo)
    phase_slope = cp + latent_heat / band_width
    h_at_liquid_start = h_at_solid_end + phase_slope * band_width

    def h(T: float) -> float:
        if T <= t_solid_end:
            return cp * (T - t_lo)
        if T <= t_liquid_start:
            return h_at_solid_end + phase_slope * (T - t_solid_end)
        return h_at_liquid_start + cp * (T - t_liquid_start)

    pairs = [(round(T, 4), round(h(T), 4)) for T in cleaned]

    # Final strict-monotonic check (defensive)
    out: List[Tuple[float, float]] = []
    for T, H in pairs:
        if out and (T <= out[-1][0] or H <= out[-1][1]):
            continue
        out.append((T, H))
    return out


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
