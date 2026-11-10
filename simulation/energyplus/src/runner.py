"""Run EnergyPlus on a set of generated IDFs in parallel.

v3: each case gets its own EPW; optional .eso/.mtr cleanup to save disk.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from utils import get_logger

log = get_logger("runner")

# Files we DELETE when keep_eso is False (large time-series / intermediate)
# Kept files: *out.err, *tbl.htm, *tbl.csv, *out.eio, *out.sql (v4: needed for Method A)
ESO_CLEANUP_SUFFIXES = ("out.eso", "out.mtr", "out.shd", "out.audit", "out.rvaudit",
                         "out.bnd", "out.mdd", "out.rdd", "out.mtd",
                         "ssz.csv", "zsz.csv", "mtr.csv")


def _run_one(case_id: str, idf_path: str, epw_path: str, out_dir: str,
             exe_path: str, timeout_sec: int, log_path: str,
             keep_eso: bool) -> Dict:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    # NOTE: -r removed (parallel readvars.audit collision); cwd isolation prevents shared file races
    cmd = [
        exe_path,
        "-w", epw_path,
        "-d", str(out),
        "-p", case_id,
        idf_path,
    ]
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write("CMD: " + " ".join(cmd) + "\n")
            lf.write(f"CWD: {out}\n\n")
            lf.flush()
            proc = subprocess.run(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                timeout=timeout_sec, check=False, cwd=str(out),
            )
        ok = proc.returncode == 0
        msg = f"exit={proc.returncode}"
    except subprocess.TimeoutExpired:
        ok = False
        msg = f"timeout after {timeout_sec}s"
    except Exception as e:
        ok = False
        msg = f"exception: {e}"
    elapsed = time.time() - t0

    # Cleanup large files on success when keep_eso=False
    if ok and not keep_eso:
        for f in out.iterdir():
            if any(f.name.endswith(suf) for suf in ESO_CLEANUP_SUFFIXES):
                try:
                    f.unlink()
                except OSError:
                    pass

    return {
        "case_id": case_id, "ok": ok, "elapsed_sec": elapsed,
        "out_dir": str(out), "message": msg,
    }


@dataclass
class RunResult:
    case_id: str
    ok: bool
    elapsed_sec: float
    out_dir: Path
    message: str = ""


def run_cases(case_run_map: Dict[str, Tuple[Path, Path]], exe_path: Path,
              results_root: Path, logs_dir: Path,
              num_workers: int, timeout_sec: int,
              retry_on_fail: int = 1,
              keep_eso: bool = False) -> List[RunResult]:
    """Run all cases.
    case_run_map: case_id -> (idf_path, epw_path)
    """
    results_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    pending: List[Tuple[str, Path, Path]] = [
        (cid, idf, epw) for cid, (idf, epw) in case_run_map.items()
    ]
    final: Dict[str, RunResult] = {}
    attempts: Dict[str, int] = {t[0]: 0 for t in pending}

    while pending:
        log.info(f"Submitting {len(pending)} cases to {num_workers} workers "
                 f"(keep_eso={keep_eso})")
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futs = {}
            for case_id, idf_path, epw_path in pending:
                out_dir = results_root / case_id
                log_path = logs_dir / f"{case_id}.log"
                f = ex.submit(_run_one, case_id, str(idf_path), str(epw_path),
                              str(out_dir), str(exe_path), timeout_sec,
                              str(log_path), keep_eso)
                futs[f] = case_id
            for fut in tqdm(as_completed(futs), total=len(futs), desc="EnergyPlus"):
                d = fut.result()
                final[d["case_id"]] = RunResult(
                    case_id=d["case_id"], ok=d["ok"],
                    elapsed_sec=d["elapsed_sec"],
                    out_dir=Path(d["out_dir"]), message=d["message"],
                )

        next_pending: List[Tuple[str, Path, Path]] = []
        for case_id, idf_path, epw_path in pending:
            r = final.get(case_id)
            if r and not r.ok and attempts[case_id] < retry_on_fail:
                attempts[case_id] += 1
                log.warning(f"Retry ({attempts[case_id]}/{retry_on_fail}) for {case_id}: {r.message}")
                next_pending.append((case_id, idf_path, epw_path))
        pending = next_pending

    return list(final.values())
