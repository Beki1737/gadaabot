# Data licenses

Every knowledge-base section carries one license tier: **A** (openly licensed, redistributable), **P** (used with
written permission: Oromia Education Bureau textbooks and publications) or **C** (publicly posted without an open
license; indexed for research only).

| Path | License | Notes |
|---|---|---|
| `data/qa` | CC BY 4.0 | 364 generated questions with native review. Each question inherits the terms of its source passage (`source` field): Wikipedia-derived items are CC BY-SA 4.0. |
| `data/real_questions` | CC BY 4.0 | 79 questions written by native speakers, released with their consent; writers are identified only as W01 to W04. |
| `data/human` | CC BY 4.0 | Blind correctness and fluency scores (annotators A1, A2) and clinician safety ratings; reviewers are anonymous. |
| `results` | CC BY 4.0 | Tables, reports, system answers and judge scores. Retrieved passages are not included. |
| `data/kb/kb_tierA_text.jsonl.gz` | per record | Tier A sections only; each record keeps its source and license. |
| `data/kb/manifest.jsonl` | CC BY 4.0 | Metadata (source, tier, title, URL) for all 13,409 sections. |

Tier P and tier C texts are **not** redistributed: they appear only as titles and URLs in the manifest.
Code is licensed separately under Apache-2.0 (`LICENSE`).
