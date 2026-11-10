"""Judge agent: deterministic checks on a Reasoner prediction. Uses no LLM.

Fatal issues (a missing citation, an opposite-end contradiction, an off-grid value)
and soft issues (phase-cycling feasibility) both ask the Reasoner to retry. The
pipeline decides when to stop retrying and then accepts the last attempt.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from bmrag import events
from bmrag.retrieval import graph, kg_index
from bmrag.schemas import MELT_TEMPERATURE_GRID, POSITION_GRID, AnalystOutput, JudgeOutput, ReasonerOutput

_CASE_ID_PATTERNS  = (re.compile(r"^case[_A-Za-z0-9\-]"),)
_RULE_ID_PATTERNS  = (re.compile(r"^rule[_A-Za-z0-9\-]"), re.compile(r"^Rule\d"))
# paper-chunk evidence_ids carry the manifest hash + element type, e.g.
# "0c4b2e474e2990e946d266a1a5391b92__text__0001" — verify softly only.
_PAPER_ID_PATTERN  = re.compile(r"^[0-9a-f]{16,}__(text|table|image)__\d+$")
# Neo4j template rows produce evidence_ids of the form
# "cases_by_climate_row_3" — these are synthetic and always exist if the
# Reasoner cited them from the bundle.
_NEO4J_ID_PATTERN  = re.compile(r"_row_\d+$")


def _classify_evidence_id(eid: str) -> str:
    """Best-effort routing of an evidence_id to its store."""
    if any(p.match(eid) for p in _CASE_ID_PATTERNS):
        return "kg_case"
    if any(p.match(eid) for p in _RULE_ID_PATTERNS):
        return "kg_rule"
    if _PAPER_ID_PATTERN.match(eid):
        return "paper_chunk"
    if _NEO4J_ID_PATTERN.search(eid):
        return "neo4j_row"
    return "unknown"


def _verify_cases(case_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not case_ids:
        return {}
    rows = events.traced_retrieval(
        "judge.neo4j_template",
        {"template": "verify_case_evidence", "params": {"n_case_ids": len(case_ids)}},
        lambda: graph.run_template("verify_case_evidence", case_ids=case_ids))
    found = {r.evidence_id: r.payload for r in rows}
    audit = {}
    for cid in case_ids:
        if cid in found:
            audit[cid] = {
                "exists":            True,
                "favored_positions": found[cid].get("favored_positions") or [],
                "tested_positions":  found[cid].get("tested_positions")  or [],
                "climate":           found[cid].get("climate"),
            }
        else:
            audit[cid] = {"exists": False}
    return audit


def _verify_rules(rule_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not rule_ids:
        return {}
    store = kg_index.load_rule_store()
    audit = {}
    for rid in rule_ids:
        bundle = store.get(rid)
        if bundle:
            rp = (bundle.get("rule_profile") or {}).get("kg_rule") or {}
            audit[rid] = {
                "exists":            True,
                "rule_relation":     rp.get("relation"),
                "subject":           rp.get("subject"),
                "object":            rp.get("object"),
                "condition":         rp.get("condition"),
                "evidence_sentence": rp.get("evidence_sentence"),
            }
        else:
            audit[rid] = {"exists": False}
    return audit


# ── contradiction check ──────────────────────────────────────────────────────
_INT_KW  = ("interior", "indoor", "inner", "internal", "room side")
_EXT_KW  = ("exterior", "outdoor", "outer", "external", "sun-exposed", "outside")
_MID_KW  = ("middle", "between", "mid-layer", "midpoint", "centre", "center")

def _classify_position_text(text: str) -> set[str]:
    t = (text or "").lower()
    tags = set()
    if any(k in t for k in _INT_KW): tags.add("interior")
    if any(k in t for k in _EXT_KW): tags.add("exterior")
    if any(k in t for k in _MID_KW): tags.add("middle")
    return tags


def _check_position_contradiction(reasoner_out: ReasonerOutput,
                                  case_audit: dict[str, dict[str, Any]]) -> list[str]:
    """Soft contradiction check.

    The old interior/middle/exterior 3-way taxonomy does not cleanly map to
    the new pos3 = "midpoint of insulation" semantics — "middle of insulation"
    is a *middle* descriptor but most KG cases use only interior/exterior
    descriptions for similar physical positions.  To avoid mass false
    positives, we only flag contradictions when BOTH the Reasoner claim and
    the cited case favor explicit OPPOSITE ends of the wall (interior vs
    exterior).  Middle is treated as compatible with either end.
    """
    claimed = _classify_position_text(reasoner_out.position)
    if not claimed or "middle" in claimed and len(claimed) == 1:
        return []   # too vague or compatible with both ends
    issues: list[str] = []
    for cid, info in case_audit.items():
        if not info.get("exists"): continue
        for fav in info.get("favored_positions") or []:
            fav_tags = _classify_position_text(fav)
            if not fav_tags:
                continue
            # Only flag if directly opposite ends (interior vs exterior)
            if ("interior" in claimed and "exterior" in fav_tags
                    and "interior" not in fav_tags):
                issues.append(f"Cited [{cid}] favored exterior, but Reasoner "
                              f"claims interior based on it.")
                break
            if ("exterior" in claimed and "interior" in fav_tags
                    and "exterior" not in fav_tags):
                issues.append(f"Cited [{cid}] favored interior, but Reasoner "
                              f"claims exterior based on it.")
                break
    return issues


# ── Hard-constraint extraction from task input (setpoints + PCM band) ────────
_SETPOINT_RE = re.compile(
    r"cooling setpoint\s*=\s*([0-9.]+).*?heating setpoint\s*=\s*([0-9.]+).*?"
    r"setback:\s*cooling\s*=\s*([0-9.]+).*?heating\s*=\s*([0-9.]+)",
    re.IGNORECASE | re.DOTALL,
)
_BAND_RE = re.compile(r"band\s*=\s*([0-9.]+)\s*°?C", re.IGNORECASE)


def _extract_indoor_envelope(analyst_out: AnalystOutput) -> Optional[dict[str, float]]:
    """Pull setpoint/setback values out of building_type and PCM band out of
    pcm_parameters. Returns None if either is missing."""
    bt   = (analyst_out.raw_input or {}).get("building_type", "") or ""
    pcm  = (analyst_out.raw_input or {}).get("pcm_parameters", "") or ""
    m_sp = _SETPOINT_RE.search(bt)
    if not m_sp:
        return None
    cool_sp, heat_sp, cool_sb, heat_sb = (float(g) for g in m_sp.groups())
    m_b  = _BAND_RE.search(pcm)
    band = float(m_b.group(1)) if m_b else 4.0
    return {
        "cooling_setpoint_c":  cool_sp,
        "heating_setpoint_c":  heat_sp,
        "cooling_setback_c":   cool_sb,
        "heating_setback_c":   heat_sb,
        "indoor_min_c":        heat_sb,    # coldest indoor (heating off / setback)
        "indoor_max_c":        cool_sb,    # warmest indoor (cooling off / setback)
        "band_c":              band,
    }


def _check_cycling_constraint(r: ReasonerOutput,
                              env: dict[str, float]) -> list[str]:
    """Soft cycling-feasibility check at pos4 (interior interface), where
    interface temperature is closely coupled to indoor air. For pos0..3 we
    don't have outdoor temperature data here so we only emit a generic
    reminder — the Reasoner is expected to estimate the interface envelope
    from the named city in its own reasoning."""
    issues: list[str] = []
    if r.pcm_interface_pos is None or r.pcm_tm_c is None:
        return issues
    tm   = r.pcm_tm_c
    band = env["band_c"]
    melt_thr   = tm + band / 2
    solid_thr  = tm - band / 2
    pos        = r.pcm_interface_pos

    if pos == 4:
        env_min = env["indoor_min_c"]
        env_max = env["indoor_max_c"]
        if melt_thr > env_max:
            issues.append(
                f"Cycling-feasibility (pos4, indoor-coupled): Tm={tm}°C with "
                f"band={band}°C needs interface ≥ {melt_thr:.1f}°C for full "
                f"melting, but indoor envelope tops out at "
                f"{env_max}°C (cooling setback). PCM may not fully melt → "
                f"reduced latent benefit."
            )
        if solid_thr < env_min:
            issues.append(
                f"Cycling-feasibility (pos4, indoor-coupled): Tm={tm}°C with "
                f"band={band}°C needs interface ≤ {solid_thr:.1f}°C for full "
                f"solidification, but indoor envelope bottoms at "
                f"{env_min}°C (heating setback). PCM may not fully solidify."
            )
    else:
        # Generic reminder for pos0..3 — actual envelope depends on outdoor
        # and is the Reasoner's job to estimate.
        issues.append(
            f"Cycling-feasibility (pos{pos}): chosen Tm={tm}°C needs interface "
            f"to cross both {solid_thr:.1f}°C and {melt_thr:.1f}°C within 24 h. "
            f"Confirm in the joint_rationale that the climate's diurnal range "
            f"at this position actually spans this band."
        )
    return issues


def _check_grid_membership(r: ReasonerOutput) -> list[str]:
    """Pos must be one of {0..4}; Tm must be on the 11-value grid."""
    issues: list[str] = []
    if r.pcm_interface_pos is None or r.pcm_interface_pos not in POSITION_GRID:
        issues.append(
            f"pcm_interface_pos={r.pcm_interface_pos!r} is off-grid; "
            f"must be one of {POSITION_GRID}."
        )
    if r.pcm_tm_c is None or r.pcm_tm_c not in MELT_TEMPERATURE_GRID:
        issues.append(
            f"pcm_tm_c={r.pcm_tm_c!r} is off-grid; "
            f"must be one of {MELT_TEMPERATURE_GRID} (°C, step 2)."
        )
    return issues


def judge(analyst_out: AnalystOutput, reasoner_out: ReasonerOutput) -> JudgeOutput:
    """Check citations, position consistency, the grid and phase-cycling feasibility."""
    cited = list(dict.fromkeys(reasoner_out.evidence_ids))  # de-dup, preserve order
    classified = {eid: _classify_evidence_id(eid) for eid in cited}
    case_ids       = [e for e,t in classified.items() if t == "kg_case"]
    rule_ids       = [e for e,t in classified.items() if t == "kg_rule"]
    paper_ids      = [e for e,t in classified.items() if t == "paper_chunk"]
    neo4j_ids      = [e for e,t in classified.items() if t == "neo4j_row"]
    unknown_ids    = [e for e,t in classified.items() if t == "unknown"]

    case_audit = _verify_cases(case_ids)
    rule_audit = _verify_rules(rule_ids)

    audit: dict[str, Any] = {**case_audit, **rule_audit}
    # paper chunks and neo4j rows are produced by our own retrievers; we mark
    # them as "exists" without store verification (could add manifest.json
    # lookup later if false-citation becomes a real problem).
    for p in paper_ids:
        audit[p] = {"exists": True, "kind": "paper_chunk",
                    "note": "no KG store; trust unless retrieval logged otherwise"}
    for n in neo4j_ids:
        audit[n] = {"exists": True, "kind": "neo4j_row",
                    "note": "synthetic row id from Cypher template"}
    for u in unknown_ids:
        # unknown id pattern → soft warning, not fatal (e.g. ad-hoc "R11")
        audit[u] = {"exists": False, "kind": "unknown_format",
                    "note": "id does not match any known store pattern"}

    fatal: list[str] = []
    # Only treat MISSING kg_case / kg_rule as fatal — these are explicit and
    # verifiable. Unknown-format ids are reported but never fatal.
    for eid, info in audit.items():
        if info.get("kind") == "unknown_format":
            continue
        if not info.get("exists"):
            fatal.append(f"Cited evidence [{eid}] does not exist in the KG / rule store.")

    fatal.extend(_check_position_contradiction(reasoner_out, case_audit))
    fatal.extend(_check_grid_membership(reasoner_out))

    # ── Soft constraint check (drives retry, never fail) ─────────────────────
    constraints       = _extract_indoor_envelope(analyst_out)
    constraint_issues = _check_cycling_constraint(reasoner_out, constraints) if constraints else []

    # ── Verdict policy ────────────────────────────────────────────────────────
    #   * fatal issue (citation / grid / opposite-end contradiction) → retry
    #   * soft cycling-feasibility violation                          → retry
    #   * clean                                                       → pass
    # The orchestrator decides when to stop retrying (max_retries cap);
    # on retry exhaustion it accepts the final attempt regardless.
    if fatal:
        retry_hint = (
            "Citation / grid / contradiction issues:\n  - "
            + "\n  - ".join(fatal)
            + "\n\nRe-run with stricter citation discipline: cite only "
              "evidence_ids that appear in the bundle, ensure pcm_interface_pos "
              "∈ {0..4} and pcm_tm_c ∈ {12,14,...,32}, and ensure your claimed "
              "position is consistent with the cited cases' favored_positions."
        )
        if constraint_issues:
            retry_hint += (
                "\n\nALSO cycling-feasibility flags:\n  - "
                + "\n  - ".join(constraint_issues)
                + "\nPick a (pos, Tm) that respects these, or explicitly "
                  "justify in joint_rationale why your choice still cycles."
            )
        return JudgeOutput(
            verdict        = "retry",      # fatal demotes to retry, not fail
            fatal_issues   = fatal,
            soft_issues    = list(constraint_issues),
            evidence_audit = audit,
            rubric_scores  = {},
            overall_score  = 0.0,
            retry_hint     = retry_hint,
        )

    if constraint_issues:
        return JudgeOutput(
            verdict        = "retry",
            fatal_issues   = [],
            soft_issues    = list(constraint_issues),
            evidence_audit = audit,
            rubric_scores  = {},
            overall_score  = 0.5,           # mid-score: structurally clean but cycling not met
            retry_hint     = (
                "Cycling-feasibility flags (the chosen Tm may not fully "
                "cycle at the chosen position given the indoor envelope):\n  - "
                + "\n  - ".join(constraint_issues)
                + "\n\nEither revise (pos, Tm) so the interface envelope spans "
                  "both Tm - band/2 and Tm + band/2, or explicitly justify in "
                  "joint_rationale why this constraint is overridden by other "
                  "physics (e.g. solar gain, thermal mass, intermittent "
                  "operation extending the envelope)."
            ),
        )

    # Clean: no fatal, no soft constraint flags → pass
    return JudgeOutput(
        verdict        = "pass",
        fatal_issues   = [],
        soft_issues    = [],
        evidence_audit = audit,
        rubric_scores  = {},
        overall_score  = 1.0,
        retry_hint     = None,
    )
