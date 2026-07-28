# Claim 6 route 2 source audit

The official `lmsys/chatbot_arena_conversations` repository is public but its
41,572,998-byte parquet content is gated. Both HF `cpu-upgrade` attempts lacked
dataset authorization; the non-slim attempt reached an explicit HTTP 403.

The `dim/lmsys_chatbot_arena_conversations` repository is an ungated,
single-commit copy uploaded in 2023. Its only parquet file is 41,573,740 bytes
with LFS SHA-256
`d1b51b6052343ea0ef3762f9b1a6607146769d673e35a2b318cc77175daddd4d`.
The differing byte size may reflect parquet metadata, and the official API
redacts its LFS hash, so byte identity cannot be established.

Route 2 therefore uses two distinct gates:

1. Before inference, require the exact paper cohort: 33,000 raw rows, 32,980
   deduplicated `question_id` rows, 20 models, and 102 features.
2. Record a canonical content hash over identifiers, model pairs, winner,
   turn, both conversations, and the released toxicity tags.

Passing these gates supports a faithful mirror interpretation but is not
reported as proof of byte identity. The full claim remains sensitive to the
paper's unavailable author code and undocumented preprocessing defaults.
