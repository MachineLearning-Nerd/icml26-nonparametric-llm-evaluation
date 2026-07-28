"""Fixed cumulative reproduction entrypoint.

The baseline repository documented four separate commands in
``.trackio/logbook/commands.md``. This runner executes those commands in that
order, prints their complete output to the OpenResearch log, and emits one
machine-readable runtime/resource summary. Scientific behavior remains in the
individual scripts; children extend this cumulative runner through committed
code while the project run command stays fixed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    ("exact_theory_claims_1_2_4", ROOT / "repro/src/exact_theory.py"),
    ("claim_5_table2_falsification", ROOT / "repro/src/claim5_falsification.py"),
    ("claim_1_gars", ROOT / "repro/tests/test_c1_gars.py"),
    ("claims_2_to_5", ROOT / "repro/src/verify.py"),
    ("claim_6_mechanism", ROOT / "repro/src/c6_demo.py"),
    ("negative_controls", ROOT / "repro/tests/test_controls.py"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def main() -> int:
    started = time.monotonic()
    step_records: list[dict[str, object]] = []
    print("=== FIXED CUMULATIVE REPRODUCTION ===", flush=True)
    print(
        json.dumps(
            {
                "git_sha": _git_sha(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "logical_cpu_allocation": _available_cpus(),
                "estimated_required_cores": 1,
                "selected_compute": "local CPU",
                "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for name, script in STEPS:
        step_started = time.monotonic()
        print(f"\n=== START {name}: {script.relative_to(ROOT)} ===", flush=True)
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=False,
        )
        elapsed = time.monotonic() - step_started
        record = {
            "name": name,
            "path": str(script.relative_to(ROOT)),
            "exit_code": completed.returncode,
            "runtime_seconds": round(elapsed, 3),
        }
        step_records.append(record)
        print(f"=== END {name}: {json.dumps(record, sort_keys=True)} ===", flush=True)
        if completed.returncode != 0:
            break

    manifest = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "logical_cpu_allocation": _available_cpus(),
        "estimated_required_cores": 1,
        "selected_compute": "local CPU",
        "runtime_seconds": round(time.monotonic() - started, 3),
        "steps": step_records,
        "passed": len(step_records) == len(STEPS)
        and all(record["exit_code"] == 0 for record in step_records),
    }
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nORX_RUN_SUMMARY={json.dumps(manifest, sort_keys=True)}", flush=True)
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
