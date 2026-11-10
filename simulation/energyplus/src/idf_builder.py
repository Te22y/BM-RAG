"""Modify base IDF: switch to CondFD, inject PCM, rewrite target Construction(s),
sync Surface references, append outputs.

Uses eppy. One Case -> one .idf in generated_idfs/.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from eppy.modeleditor import IDF

from case_generator import Case
from config import Config
from utils import build_enthalpy_curve, get_logger

log = get_logger("idf_builder")

PCM_MATERIAL_NAME = "PCM_Material"
PSM_MATERIAL_NAME = "PSM_Material"  # Method B: phase-stable twin (no latent heat)
# Position 3 (NEW in v4): PCM at MIDDLE of insulation -- requires splitting
# the NoMass insulation layer into two equal R/2 halves around the PCM.
SPLIT_INSULATION_POSITION = 3


def _ensure_idd(cfg: Config) -> None:
    """Set IDD once per process. eppy raises if set twice with different path."""
    try:
        IDF.setiddname(str(cfg.energyplus.idd_path))
    except Exception:
        pass  # already set


def _set_heat_balance_algorithm(idf: IDF, algorithm: str) -> None:
    objs = idf.idfobjects["HEATBALANCEALGORITHM"]
    if objs:
        objs[0].Algorithm = algorithm
    else:
        idf.newidfobject("HEATBALANCEALGORITHM", Algorithm=algorithm)


def _set_timestep(idf: IDF, n: int) -> None:
    """Set IDF Timestep (steps per hour). Edit existing or create."""
    objs = idf.idfobjects["TIMESTEP"]
    if objs:
        objs[0].Number_of_Timesteps_per_Hour = n
    else:
        idf.newidfobject("TIMESTEP", Number_of_Timesteps_per_Hour=n)


def _ensure_cond_fd_settings(idf: IDF, cfg: Config) -> None:
    fd = cfg.simulation.cond_fd
    cls = "HEATBALANCESETTINGS:CONDUCTIONFINITEDIFFERENCE"
    existing = idf.idfobjects[cls]
    if existing:
        obj = existing[0]
    else:
        obj = idf.newidfobject(cls)
    obj.Difference_Scheme = fd.difference_scheme
    obj.Space_Discretization_Constant = fd.space_discretization_constant
    obj.Relaxation_Factor = fd.relaxation_factor
    obj.Inside_Face_Surface_Temperature_Convergence_Criteria = (
        fd.inside_face_surface_temperature_convergence_criteria
    )


def _add_pcm_material(idf: IDF, cfg: Config, melt_temp: float,
                       psm: bool = False) -> str:
    """Add a Material (+ MaterialProperty:PhaseChange unless psm=True). Return material name.

    psm=True is Method B's phase-stable twin: identical k/rho/cp/thickness but NO latent
    heat (no PhaseChange object), so EnergyPlus treats it as ordinary sensible-heat material.
    """
    p = cfg.pcm
    name = PSM_MATERIAL_NAME if psm else PCM_MATERIAL_NAME

    # Material (solid representation)
    for old in list(idf.idfobjects["MATERIAL"]):
        if old.Name == name:
            idf.removeidfobject(old)
    idf.newidfobject(
        "MATERIAL",
        Name=name,
        Roughness="MediumSmooth",
        Thickness=p.thickness,
        Conductivity=p.conductivity,
        Density=p.density,
        Specific_Heat=p.specific_heat,
        Thermal_Absorptance=p.thermal_absorptance,
        Solar_Absorptance=p.solar_absorptance,
        Visible_Absorptance=p.visible_absorptance,
    )

    if psm:
        # Method B: scrub any leftover PhaseChange object on this name -> sensible-only.
        cls = "MATERIALPROPERTY:PHASECHANGE"
        for old in list(idf.idfobjects[cls]):
            if old.Name == name:
                idf.removeidfobject(old)
        log.debug(f"PSM material added (no latent heat): {name}")
        return name

    # MaterialProperty:PhaseChange (T-Enthalpy pairs)
    cls = "MATERIALPROPERTY:PHASECHANGE"
    for old in list(idf.idfobjects[cls]):
        if old.Name == name:
            idf.removeidfobject(old)
    pc = idf.newidfobject(cls)
    pc.Name = name
    pc.Temperature_Coefficient_for_Thermal_Conductivity = 0.0
    pairs = build_enthalpy_curve(
        melt_temp=melt_temp,
        band_width=p.band_width,
        cp=p.specific_heat,
        latent_heat=p.latent_heat,
        num_points=p.num_curve_points,
    )
    if len(pairs) > 16:
        pairs = pairs[:16]
    for i, (T, H) in enumerate(pairs, start=1):
        setattr(pc, f"Temperature_{i}", T)
        setattr(pc, f"Enthalpy_{i}", H)
    log.debug(f"PCM curve Tm={melt_temp}: {len(pairs)} pts, T range [{pairs[0][0]}, {pairs[-1][0]}]")
    return name


def _find_insulation_layer_index(idf: IDF, layers: List[str]) -> int:
    """Return the index in `layers` of the first Material:NoMass insulation layer.
    Used by pos3 (split-insulation) logic. Raises if not found.
    """
    nomass_names = {m.Name for m in idf.idfobjects["MATERIAL:NOMASS"]}
    for i, lname in enumerate(layers):
        if lname in nomass_names and "insul" in lname.lower():
            return i
    # fallback: any NoMass layer
    for i, lname in enumerate(layers):
        if lname in nomass_names:
            return i
    raise ValueError(f"No Material:NoMass insulation layer found among {layers}")


def _ensure_half_insulation_material(idf: IDF, full_name: str) -> str:
    """Create (if absent) a Material:NoMass with half the R-value of `full_name`.
    Returns the half-material's name. Idempotent within the IDF.
    """
    half_name = f"{full_name}__HALF"
    # Already exists?
    for m in idf.idfobjects["MATERIAL:NOMASS"]:
        if m.Name == half_name:
            return half_name

    src = None
    for m in idf.idfobjects["MATERIAL:NOMASS"]:
        if m.Name == full_name:
            src = m
            break
    if src is None:
        raise ValueError(f"Material:NoMass not found: {full_name}")

    half = idf.newidfobject("MATERIAL:NOMASS")
    half.Name = half_name
    half.Roughness = src.Roughness
    half.Thermal_Resistance = float(src.Thermal_Resistance) / 2.0
    half.Thermal_Absorptance = src.Thermal_Absorptance
    half.Solar_Absorptance = src.Solar_Absorptance
    half.Visible_Absorptance = src.Visible_Absorptance
    log.debug(f"Created half-insulation: {half_name} (R={half.Thermal_Resistance:.4f})")
    return half_name


def _rewrite_construction(idf: IDF, orig_name: str, new_name: str,
                          insert_pos: int | None, pcm_name: str) -> None:
    """Clone the original Construction with PCM inserted at insert_pos, rename to new_name.

    v4 position semantics (4-layer wall [stucco, gypsum, insulation, gypsum]):
      0 -> insert before layer 0 (outside)
      1 -> insert before layer 1
      2 -> insert before layer 2
      3 -> SPLIT the NoMass insulation into R/2 halves, PCM at the centre  [NEW]
      4 -> insert before layer 3 (= old pos3, insulation/inner-gypsum interface)
    """
    src = None
    for c in idf.idfobjects["CONSTRUCTION"]:
        if c.Name == orig_name:
            src = c
            break
    if src is None:
        raise ValueError(f"Construction not found: {orig_name}")

    # Collect existing layers (in order). eppy uses Outside_Layer, Layer_2, Layer_3, ...
    layers: List[str] = []
    if src.Outside_Layer:
        layers.append(src.Outside_Layer)
    i = 2
    while True:
        attr = f"Layer_{i}"
        if not hasattr(src, attr):
            break
        val = getattr(src, attr)
        if not val:
            break
        layers.append(val)
        i += 1

    if insert_pos == SPLIT_INSULATION_POSITION:
        # pos3: split insulation into two R/2 halves around the PCM
        insul_idx = _find_insulation_layer_index(idf, layers)
        full_insul_name = layers[insul_idx]
        half_name = _ensure_half_insulation_material(idf, full_insul_name)
        new_layers = (layers[:insul_idx]
                      + [half_name, pcm_name, half_name]
                      + layers[insul_idx + 1:])
        log.debug(f"pos3 split: {full_insul_name} -> [{half_name}, PCM, {half_name}]")
    elif insert_pos == 4:
        # pos4 maps to old pos3 (insert before layer index 3)
        new_layers = list(layers)
        new_layers.insert(3, pcm_name)
    elif insert_pos is not None:
        if insert_pos < 0 or insert_pos > len(layers):
            raise ValueError(f"insert_pos {insert_pos} out of range for {len(layers)} layers")
        new_layers = list(layers)
        new_layers.insert(insert_pos, pcm_name)
    else:
        new_layers = list(layers)

    # Remove any existing construction with new_name
    for old in list(idf.idfobjects["CONSTRUCTION"]):
        if old.Name == new_name:
            idf.removeidfobject(old)

    new_obj = idf.newidfobject("CONSTRUCTION", Name=new_name)
    new_obj.Outside_Layer = new_layers[0]
    for idx, layer_name in enumerate(new_layers[1:], start=2):
        setattr(new_obj, f"Layer_{idx}", layer_name)


def _retarget_surfaces(idf: IDF, orig_name: str, new_name: str) -> int:
    """Rewrite all BuildingSurface:Detailed.Construction_Name from orig_name to new_name.
    Return count of surfaces updated."""
    count = 0
    for s in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        if s.Construction_Name == orig_name:
            s.Construction_Name = new_name
            count += 1
    return count


def _append_outputs(idf: IDF, cfg: Config) -> None:
    freq = cfg.output.reporting_frequency
    existing_vars = {(v.Variable_Name, v.Key_Value) for v in idf.idfobjects["OUTPUT:VARIABLE"]}
    for var in cfg.output.variables:
        key = (var, "*")
        if key in existing_vars:
            continue
        idf.newidfobject(
            "OUTPUT:VARIABLE",
            Key_Value="*",
            Variable_Name=var,
            Reporting_Frequency=freq,
        )
    existing_meters = {(m.Key_Name, m.Reporting_Frequency)
                       for m in idf.idfobjects["OUTPUT:METER"]}
    for meter in cfg.output.meters:
        if (meter, freq) in existing_meters:
            continue
        idf.newidfobject(
            "OUTPUT:METER",
            Key_Name=meter,
            Reporting_Frequency=freq,
        )
    # v4: Output:SQLite so postprocess can read hourly Surface Temperatures for Method A.
    if not idf.idfobjects["OUTPUT:SQLITE"]:
        idf.newidfobject("OUTPUT:SQLITE", Option_Type="Simple")


def build_case_idf(cfg: Config, case: Case, out_dir: Path) -> Path:
    """Generate the IDF file for one case. Return output path.

    v3: selects template IDF based on case.zone (cfg.climate_zones[zone].template_idf).
    """
    _ensure_idd(cfg)
    template = cfg.climate_zones[case.zone].template_idf
    idf = IDF(str(template))

    _set_heat_balance_algorithm(idf, cfg.simulation.heat_balance_algorithm)
    _set_timestep(idf, cfg.simulation.timestep)
    _ensure_cond_fd_settings(idf, cfg)

    if not case.is_baseline:
        assert case.melt_temp is not None and case.position is not None
        pcm_name = _add_pcm_material(idf, cfg, case.melt_temp, psm=case.is_psm)
        tag = "psm" if case.is_psm else "pcm"
        for orig in cfg.target_constructions:
            new_name = f"{orig}__{tag}_pos{case.position}_tm{int(case.melt_temp)}"
            _rewrite_construction(idf, orig, new_name, case.position, pcm_name)
            updated = _retarget_surfaces(idf, orig, new_name)
            log.debug(f"{case.case_id}: {orig} -> {new_name}, {updated} surfaces updated")
    else:
        # baseline: only algorithm change, keep original construction
        log.debug(f"{case.case_id}: baseline (no PCM, CondFD only)")

    _append_outputs(idf, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case.case_id}.idf"
    idf.saveas(str(out_path))
    return out_path
