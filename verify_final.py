#!/usr/bin/env python3
"""Verify the published DMLRank claim-audit contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_IDENTITY = (
    "MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>"
)
EXPECTED_RECOVERY_SHA = (
    "0c7e874d1cb35c35490c9b4518417b5f6670c668f48606e9ec9b8f9d97205bee"
)
EXPECTED_SOURCE_TIP = "f7b0025c2d158594709d8ba7ce9ef0efc08d00ed"
EXPECTED_BRANCHES = {
    "main",
    "audit/arena-integrity",
    "audit/baseline-locked-env",
    "audit/chatbot-arena-full",
    "audit/claim6-checker",
    "audit/cumulative-evidence",
    "audit/table1-borda",
    "audit/table1-covariate-interpretation",
    "audit/table1-falsification",
    "audit/table1-metric-audit",
    "audit/table2-crossfit",
    "audit/table2-falsification",
    "audit/table2-oracle",
    "audit/table2-pilot-learned",
    "audit/theorem-contracts",
}
EXPECTED_STATUS = (
    "INCONCLUSIVE_C1_C2_C4_C6_VERIFIED_C3_C5_BLOCKED_"
    "NO_PAPER_CLAIMS_VERIFIED_NO_CURRENT_SCORE"
)


def fail(reason: str) -> None:
    print("FINAL_AUDIT=FAILED reason=" + reason)
    raise SystemExit(1)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail("git_" + "_".join(args))
    return result.stdout.strip()


def load(relative_path: str) -> dict:
    try:
        with (ROOT / relative_path).open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        fail(relative_path + "_invalid_" + type(error).__name__)
    raise AssertionError("unreachable")


local_heads = {
    line
    for line in git(
        "for-each-ref",
        "refs/heads",
        "--format=%(refname:short)",
    ).splitlines()
    if line
}
remote_heads = {
    line.removeprefix("origin/")
    for line in git(
        "for-each-ref",
        "refs/remotes/origin",
        "--format=%(refname:short)",
    ).splitlines()
    if line and line != "origin/HEAD"
}
if local_heads not in (EXPECTED_BRANCHES, {"main"}):
    fail("local_branches_" + ",".join(sorted(local_heads)))
if remote_heads and remote_heads != EXPECTED_BRANCHES:
    fail("remote_branches_" + ",".join(sorted(remote_heads)))
if git("branch", "--show-current") != "main":
    fail("head_not_main")

all_refs = git("for-each-ref", "--format=%(refname)").splitlines()
if any(
    ref.endswith("/master")
    or "/orx/" in ref
    or ref.endswith("/orx")
    for ref in all_refs
):
    fail("legacy_branch_ref")

commit_count = int(git("rev-list", "--count", "--all"))
if commit_count != 23:
    fail("commit_count_" + str(commit_count))

identity_rows = git(
    "log",
    "--all",
    "--format=%an <%ae>|%cn <%ce>",
).splitlines()
expected_row = EXPECTED_IDENTITY + "|" + EXPECTED_IDENTITY
if not identity_rows or any(row != expected_row for row in identity_rows):
    fail("noncanonical_commit_identity")

claims_doc = load("claims.json")
claims = {claim["id"]: claim for claim in claims_doc["claims"]}
expected_claims = {
    "C1": "VERIFIED_SCOPED",
    "C2": "VERIFIED_SCOPED",
    "C3": "BLOCKED",
    "C4": "VERIFIED_SCOPED",
    "C5": "BLOCKED",
    "C6": "VERIFIED_SCOPED",
}
if set(claims) != set(expected_claims):
    fail("claim_ids")
if {claim_id: claims[claim_id]["status"] for claim_id in expected_claims} != expected_claims:
    fail("claim_statuses")
if claims_doc.get("overall_status") != EXPECTED_STATUS:
    fail("claims_status")
audit = claims_doc.get("audit", {})
if audit.get("scoped_contracts_verified") != 4:
    fail("scoped_contracts_verified")
if audit.get("claims_blocked") != 2:
    fail("claims_blocked")
if audit.get("claims_total") != 6:
    fail("claims_total")
if audit.get("evidence_points") != 8:
    fail("evidence_points")
if audit.get("paper_claims_verified") != 0:
    fail("paper_claims_verified")
if audit.get("current_score_claim") is not False:
    fail("current_score_claim")
if audit.get("publication_allowed") is not False:
    fail("publication_allowed")

verdict = load("outputs/verdict.json")
if verdict.get("paper_reproduction") != "inconclusive":
    fail("paper_reproduction")
if verdict.get("claims_verified") != 4:
    fail("verdict_claims_verified")
if verdict.get("claims_blocked") != 2:
    fail("verdict_claims_blocked")
if verdict.get("paper_claims_verified") != 0:
    fail("verdict_paper_claims")
if verdict.get("current_score_claim") is not False:
    fail("verdict_current_score")
if verdict.get("publication_allowed") is not False:
    fail("verdict_publication_boundary")

gate = load("outputs/gate.json")
for field in (
    "tests_passed",
    "documentation_gate_passed",
    "publication_gate_passed",
):
    if gate.get(field) is not True:
        fail("gate_" + field)
for field in (
    "paper_reproduction_gate_passed",
    "paper_claims_reproduced",
    "current_score_claim",
    "publication_allowed",
):
    if gate.get(field) is not False:
        fail("gate_" + field)
if gate.get("paper_algorithm_implemented") is not True:
    fail("gate_paper_algorithm_implemented")
if gate.get("overall_status") != "INCONCLUSIVE":
    fail("gate_status")
if gate.get("scoped_contracts_verified") != 4:
    fail("gate_scoped_contracts")
if gate.get("claims_blocked") != 2:
    fail("gate_blocked")
if gate.get("claims_total") != 6:
    fail("gate_claims_total")
if gate.get("evidence_points") != 8:
    fail("gate_evidence")
if gate.get("paper_claims_verified") != 0:
    fail("gate_paper_claims")

verdicts = load("reproduction_verdicts.json")
if verdicts.get("overall_verdict") != (
    "INCONCLUSIVE_C1_C2_C4_C6_VERIFIED_C3_C5_BLOCKED"
):
    fail("reproduction_verdict")
if verdicts.get("claim_statuses") != expected_claims:
    fail("reproduction_claim_statuses")
if verdicts.get("evidence", {}).get("publication_allowed") is not False:
    fail("reproduction_publication_boundary")

state = load("AUTONOMOUS_STATE.json")
if state.get("status") != EXPECTED_STATUS:
    fail("state_status")
if state.get("repository", {}).get("recovery_bundle_sha256") != EXPECTED_RECOVERY_SHA:
    fail("state_recovery_sha")
if state.get("repository", {}).get("canonical_email") != (
    "MachineLearning-Nerd@users.noreply.github.com"
):
    fail("state_identity")
if state.get("repository", {}).get("branches") != 15:
    fail("state_branch_count")

manifest = load("EVIDENCE_MANIFEST.json")
missing = [
    path
    for path in manifest["required_paths"]
    if not (ROOT / path).is_file()
]
if missing:
    fail("missing_paths_" + ",".join(missing))

readme = (ROOT / "README.md").read_text(encoding="utf-8")
for marker in (
    "2601.21816",
    "CLAIM_EVIDENCE.md",
    "Thank you",
    "not an author-maintained implementation",
    "0/6",
    "MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>",
):
    if marker not in readme:
        fail("readme_" + marker.replace(" ", "_"))

branch_audit = (ROOT / "BRANCH_AUDIT.md").read_text(encoding="utf-8")
for marker in (
    EXPECTED_IDENTITY,
    EXPECTED_SOURCE_TIP,
    EXPECTED_RECOVERY_SHA,
    "audit/table1-borda",
    "audit/table2-crossfit",
):
    if marker not in branch_audit:
        fail("branch_audit_" + marker.replace(" ", "_"))

gate_ready = (ROOT / "GATE_READY.md").read_text(encoding="utf-8")
if "Complete paper-level claims independently verified: 0 of 6" not in gate_ready:
    fail("gate_ready_boundary")

print(
    "FINAL_AUDIT=VERIFIED "
    "branches=15 "
    "commits=23 "
    "claims=C1:C2:C4:C6_verified_scoped,C3:C5_blocked "
    "evidence_points=8 "
    "paper_claims_verified=0 "
    "current_score_claim=false "
    "publication_allowed=false"
)
