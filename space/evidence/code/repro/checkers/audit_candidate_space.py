"""Evaluator-blind audit of a downloaded Hugging Face candidate.

The auditor accepts only a candidate directory. It starts from README.md,
follows local Markdown links, and never consults the experiment repository.
It exits nonzero when navigation, evidence visibility, or historical safety
requirements are missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import deque
from pathlib import Path


LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
CLAIM_REQUIREMENTS = {
    1: ["VERIFIED", "Exact claim and source", "Raw output", "control", "Fixed command", "Git"],
    2: ["VERIFIED", "Exact claim and source", "pathwise", "Omitted-IPW", "Fixed command", "confidence is MEDIUM"],
    3: ["BLOCKED", "Exact claim and source", "Four materially different routes", "Controls", "Fixed command", "Unblocker"],
    4: ["VERIFIED", "Exact claim and source", "SLSQP", "KKT", "Fixed command", "control"],
    5: ["BLOCKED", "Exact claim and source", "Four materially different routes", "Controls", "Fixed command", "Unblocker"],
    6: ["VERIFIED", "Exact claim and source", "32,980", "Independent", "Fixed command", "confidence is MEDIUM"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_target(source: Path, raw_target: str, root: Path) -> Path | None:
    target = raw_target.split("#", 1)[0]
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    resolved = (source.parent / target).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"link escapes candidate: {source.relative_to(root)} -> {raw_target}")
    return resolved


def audit(root: Path) -> dict:
    errors: list[str] = []
    opened: list[str] = []
    entrypoint = root / "README.md"
    if not entrypoint.is_file():
        return {"passed": False, "errors": ["README.md missing"], "opened": []}

    queue = deque([entrypoint])
    seen: set[Path] = set()
    while queue:
        path = queue.popleft().resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            errors.append(f"missing linked file: {path}")
            continue
        opened.append(str(path.relative_to(root.resolve())))
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            try:
                target = local_target(path, raw_target, root)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if target is None:
                continue
            if not target.is_file():
                errors.append(
                    f"broken link: {path.relative_to(root)} -> {raw_target}"
                )
            elif target.suffix.lower() == ".md":
                queue.append(target)
            else:
                opened.append(str(target.relative_to(root.resolve())))

    readme = entrypoint.read_text(encoding="utf-8")
    for claim in range(1, 7):
        relative = f"pages/claims/claim-{claim}.md"
        path = root / relative
        if relative not in opened:
            errors.append(f"claim {claim} page not reachable from README")
        if not path.is_file():
            errors.append(f"claim {claim} page missing")
            continue
        text = path.read_text(encoding="utf-8")
        for required in CLAIM_REQUIREMENTS[claim]:
            if required.lower() not in text.lower():
                errors.append(f"claim {claim} missing visible field: {required}")

    index = root / "pages/index.md"
    if index.is_file():
        index_text = index.read_text(encoding="utf-8")
        for heading in [
            "Code visible",
            "Data inline",
            "Raw link",
            "Checker",
            "Control",
            "Exact claim tested",
            "Reviewer verdict",
        ]:
            if heading not in index_text:
                errors.append(f"visibility matrix missing column: {heading}")
    else:
        errors.append("pages/index.md missing")

    if "Historical rejected baseline" not in readme:
        errors.append("README does not label historical evidence")
    historical = root / "historical/judged-1e8c465/pages/executive-summary/page.md"
    if not historical.is_file():
        errors.append("exact historical executive summary missing")

    try:
        logbook = json.loads((root / "logbook.json").read_text(encoding="utf-8"))
        if logbook.get("space_id") != "DineshAI/rHndxbqWyh":
            errors.append("logbook space_id mismatch")
        if logbook.get("root", {}).get("file") != "pages/index.md":
            errors.append("logbook root is not current evidence index")
    except Exception as exc:
        errors.append(f"invalid logbook.json: {exc}")

    opened_unique = sorted(set(opened))
    return {
        "schema_version": 1,
        "candidate_sha256": {
            path: sha256(root / path)
            for path in opened_unique
            if (root / path).is_file()
        },
        "opened": opened_unique,
        "errors": errors,
        "passed": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.candidate.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
