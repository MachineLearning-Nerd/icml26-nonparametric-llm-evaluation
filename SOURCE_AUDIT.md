# Source audit

## Paper identity

- Title: Nonparametric LLM Evaluation from Preference Data
- Method name: DMLRank
- arXiv: 2601.21816
- OpenReview: rHndxbqWyh
- Authors: Dennis Frauen; Athiya Deviyani; Mihaela van der Schaar; Stefan
  Feuerriegel
- Paper links: https://arxiv.org/abs/2601.21816 and
  https://openreview.net/forum?id=rHndxbqWyh
- Author-linked implementation:
  https://github.com/DennisFrauen/NonparametricLLMEval

## What was available

The repository records the paper claim contracts, theorem anchors, source
notes, finite certificates, simulation routes, and full-scale Arena evidence
under .openresearch/artifacts/ and space/. The arXiv HTML source used by the
evidence package was retrieved on 2026-07-28 and is recorded with SHA-256
25f14e3a172cfa9427c501e9b1ed973c4a0e1549ee03a7593e5312f9e6474847 in
space/index.md.

## Source limitations

- The anonymous or author-controlled implementation trail did not provide the
  exact finite-run seed, coefficient draw, metric convention, and revision
  needed to reproduce Table 1 and Table 2 exactly.
- The official LMSYS/Chatbot Arena file is gated in the compute environment.
  C6 uses a pinned mirror with matching cohort invariants; byte identity is
  not established.
- The repository is an independent audit and not an author-maintained
  implementation. Its evidence is not an endorsement by the paper authors.

## Audit conclusion

The paper identity and claim structure are pinned. C1, C2, C4, and C6 have
scoped evidence contracts that pass. C3 and C5 remain blocked rather than
being called failures. No complete paper-level claim or current external
score is claimed.
