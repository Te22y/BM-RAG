"""Load and validate config.yaml (v3.1: local EPW lookup, no download)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from utils import PROJECT_ROOT


@dataclass
class CityEntry:
    name: str
    epw_filename: str         # filename to search for in epw_root tree
    resolved_epw: Optional[Path] = None   # populated after locate


@dataclass
class ZoneEntry:
    name: str
    template_idf: Path
    cities: List[CityEntry]


@dataclass
class PcmConfig:
    thickness: float
    conductivity: float
    density: float
    specific_heat: float
    latent_heat: float
    band_width: float
    melt_temps: List[float]
    thermal_absorptance: float
    solar_absorptance: float
    visible_absorptance: float
    num_curve_points: int
    insert_positions: List[int]


@dataclass
class PsmConfig:
    """Phase-stable-material twin sweep (Method B)."""
    enabled: bool
    positions: List[int]
    melt_temps: List[float]


@dataclass
class CondFdSettings:
    difference_scheme: str
    space_discretization_constant: float
    relaxation_factor: float
    inside_face_surface_temperature_convergence_criteria: float


@dataclass
class SimulationConfig:
    timestep: int
    heat_balance_algorithm: str
    cond_fd: CondFdSettings


@dataclass
class OutputConfig:
    variables: List[str]
    meters: List[str]
    reporting_frequency: str


@dataclass
class ExecutionConfig:
    num_workers: int
    per_case_timeout_sec: int
    retry_on_fail: int


@dataclass
class EnergyPlusConfig:
    exe_path: Path
    idd_path: Path
    version: str


@dataclass
class Config:
    energyplus: EnergyPlusConfig
    climate_zones: Dict[str, ZoneEntry]
    epw_root: Path
    target_constructions: List[str]
    pcm: PcmConfig
    psm: PsmConfig
    simulation: SimulationConfig
    output: OutputConfig
    execution: ExecutionConfig
    raw: Dict[str, Any] = field(default_factory=dict)


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def load_config(config_path: Path | None = None) -> Config:
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ep = raw["energyplus"]
    energyplus = EnergyPlusConfig(
        exe_path=_resolve(ep["exe_path"]),
        idd_path=_resolve(ep["idd_path"]),
        version=str(ep["version"]),
    )

    climate_zones: Dict[str, ZoneEntry] = {}
    for zone_name, zone_raw in raw["climate_zones"].items():
        cities = [CityEntry(name=c["name"], epw_filename=c["epw_filename"])
                  for c in zone_raw["cities"]]
        climate_zones[zone_name] = ZoneEntry(
            name=zone_name,
            template_idf=_resolve(zone_raw["template_idf"]),
            cities=cities,
        )

    pcm = PcmConfig(**raw["pcm"])

    psm_raw = raw.get("psm", {"enabled": False, "positions": [], "melt_temps": []})
    psm = PsmConfig(
        enabled=bool(psm_raw.get("enabled", False)),
        positions=list(psm_raw.get("positions", [])),
        melt_temps=list(psm_raw.get("melt_temps", [])),
    )

    sim_raw = raw["simulation"]
    cond_fd = CondFdSettings(**sim_raw["cond_fd"])
    simulation = SimulationConfig(
        timestep=int(sim_raw["timestep"]),
        heat_balance_algorithm=sim_raw["heat_balance_algorithm"],
        cond_fd=cond_fd,
    )

    output = OutputConfig(**raw["output"])
    execution = ExecutionConfig(**raw["execution"])

    cfg = Config(
        energyplus=energyplus,
        climate_zones=climate_zones,
        epw_root=_resolve(raw.get("epw_root", "inputs/epw")),
        target_constructions=list(raw["target_constructions"]),
        pcm=pcm,
        psm=psm,
        simulation=simulation,
        output=output,
        execution=execution,
        raw=raw,
    )

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    errors: List[str] = []
    if not cfg.energyplus.exe_path.exists():
        errors.append(f"energyplus.exe not found: {cfg.energyplus.exe_path}")
    if not cfg.energyplus.idd_path.exists():
        errors.append(f"Energy+.idd not found: {cfg.energyplus.idd_path}")
    if not cfg.epw_root.exists():
        errors.append(f"epw_root directory not found: {cfg.epw_root}")
    if not cfg.climate_zones:
        errors.append("climate_zones is empty")
    for zone_name, ze in cfg.climate_zones.items():
        if not ze.template_idf.exists():
            errors.append(f"template_idf missing for {zone_name}: {ze.template_idf}")
        if not ze.cities:
            errors.append(f"zone {zone_name} has no cities")
        seen_names = set()
        for city in ze.cities:
            if city.name in seen_names:
                errors.append(f"duplicate city name '{city.name}' in zone {zone_name}")
            seen_names.add(city.name)
    if not cfg.pcm.melt_temps:
        errors.append("pcm.melt_temps is empty")
    if not cfg.pcm.insert_positions:
        errors.append("pcm.insert_positions is empty")
    if not cfg.target_constructions:
        errors.append("target_constructions is empty")
    legal_ts = {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}
    if cfg.simulation.timestep not in legal_ts:
        errors.append(
            f"simulation.timestep={cfg.simulation.timestep} not in legal set {sorted(legal_ts)}"
        )
    if errors:
        msg = "Config validation failed:\n  - " + "\n  - ".join(errors)
        raise ValueError(msg)
