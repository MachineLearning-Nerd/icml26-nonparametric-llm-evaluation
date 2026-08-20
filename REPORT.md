# Publication report

## Current result

This repository is a documented independent audit of DMLRank. Four scoped
claim contracts pass and two finite claims are blocked:

- C1 GARS special cases: VERIFIED_SCOPED
- C2 finite EIF certificate: VERIFIED_SCOPED
- C3 Table 1 finite numbers: BLOCKED
- C4 A-optimal policy: VERIFIED_SCOPED
- C5 Table 2 finite numbers: BLOCKED
- C6 full-scale Arena intervals: VERIFIED_SCOPED

The complete technical report is
reports/reproduction/report.md. The claim-by-claim evidence ledger is
CLAIM_EVIDENCE.md and the evaluator-facing evidence matrix is
space/pages/index.md.

## Publication boundary

The repository is publishable as an evidence audit and documentation package.
It does not claim that all six paper results were reproduced, does not claim
author endorsement, and does not claim a current external judge score.
Complete paper-level claims independently verified: 0 of 6.

## Unblockers

C3 and C5 require the exact author-controlled finite-run realization,
including the relevant data/code revision, seeds, coefficient draws, and
metric conventions. Strengthening C6 requires access to the official gated
source or a verifiable byte-identical release.
