# Evaluator-blind pre-publication review

The reviewer was given only a fresh copy of this candidate directory and the
rubric. It started at `README.md`, followed local links, and did not use the
research repository, OpenResearch logs, unpublished branches, or dashboard
knowledge. The executable audit is
[`audit_candidate_space.py`](../evidence/code/repro/checkers/audit_candidate_space.py).

## Pass 1

The reviewer opened 58 reachable files. The exact opened-file list and hashes
are in [`audit-pass1.json`](../evidence/audit-pass1.json). It found four
visibility defects:

1. `pages/red-team.md` did not yet exist.
2. `release-manifest.sha256` did not yet exist.
3. The historical exact-page link traversed one directory too far.
4. Claim 4 did not literally label its rejecting row as a control.

No scientific claim was upgraded because of this audit. The link and label
defects were fixed; the text manifest is generated only after the cumulative
run and final candidate freeze so its hashes cannot become stale.

## Pass 2

After the cumulative run, fixes, and initial manifest, the reviewer repeated
the audit from a second fresh directory. It opened 62 reachable files, found
zero broken links or missing visibility fields, and passed. The exact
opened-file list, hashes, and empty error array are in
[`audit-pass2.json`](../evidence/audit-pass2.json).

Because adding this record changes the candidate, the allowlist manifest is
regenerated once more and a final no-write audit is required. Publication
remains blocked unless that final audit also passes without modifying the
candidate.
