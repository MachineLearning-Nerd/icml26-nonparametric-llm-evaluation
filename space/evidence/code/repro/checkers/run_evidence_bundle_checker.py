"""Prove that the final evidence verifier accepts real and rejects corrupt data."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "repro/checkers/verify_evidence_bundle.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    actual = _run()
    corrupted = _run("--corrupt")
    print(actual.stdout, end="", flush=True)
    print(corrupted.stdout, end="", flush=True)
    if actual.stderr:
        print(actual.stderr, end="", file=sys.stderr, flush=True)
    if corrupted.stderr:
        print(corrupted.stderr, end="", file=sys.stderr, flush=True)
    passed = actual.returncode == 0 and corrupted.returncode != 0
    summary = {
        "actual_exit_code": actual.returncode,
        "actual_passed": actual.returncode == 0,
        "corrupted_exit_code": corrupted.returncode,
        "corrupted_rejected": corrupted.returncode != 0,
        "passed": passed,
    }
    output = ROOT / "outputs/evidence_bundle_checker.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "EVIDENCE_BUNDLE_CHECKER_SUITE="
        + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
