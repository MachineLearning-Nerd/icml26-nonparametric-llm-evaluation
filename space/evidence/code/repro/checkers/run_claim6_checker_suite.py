"""Run the Claim 6 checker on real and deliberately corrupted evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "repro/checkers/check_claim6.py"
RESULT = ROOT / "outputs/arena_full_results.json"


def _run(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--input", str(RESULT), *extra],
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
        "actual_evidence_exit_code": actual.returncode,
        "actual_evidence_passed": actual.returncode == 0,
        "corrupted_evidence_exit_code": corrupted.returncode,
        "corrupted_evidence_rejected": corrupted.returncode != 0,
        "passed": passed,
    }
    (ROOT / "outputs/claim6_independent_checker.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "CLAIM6_INDEPENDENT_CHECKER=" + json.dumps(summary, sort_keys=True),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
