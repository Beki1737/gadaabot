# GadaaBot 2.0: native Afaan Oromoo question answering for health and education

GadaaBot answers health and school questions in **Afaan Oromoo**, grounded in Afaan Oromoo sources, with citations,
health disclaimers, an urgent-care screen and abstention. Retrieval, generation and evaluation stay in Afaan Oromoo.

**Paper:** Gudeta, B. F., Ojeda Rosales, D., and Kim, Y. *Native Evidence, Not More Agents: Retrieval-Augmented QA
for Afaan Oromoo Health and Education.* Submitted to the Journal of Intelligent Information Systems (2026).
Release v1.0 of this repository is the version described in the paper (see `CITATION.cff`).

## Key results (correctness 1 to 5)
| System (same 9B generator) | 184 test questions, LLM judge | 79 real questions, 2 blind native speakers |
|---|---|---|
| Closed book | 3.00 | 3.20 |
| RAG, evidence translated with NLLB-200 | 3.19 | 2.64 |
| RAG, model translates the evidence itself | 3.80 | |
| **Native RAG (production)** | **4.38** | **3.47** |
| Agent: refine + grade + reformulate | 4.27 | 3.39 |
| Agent with self-verification | 3.88 | |

- Native evidence beats the same evidence translated into English (real questions: +0.84, 95% CI 0.46 to 1.18).
- Agentic layers never improve correctness; self-verification hurts.
- On real questions the gain over closed book shrinks to +0.28 (CI includes zero): knowledge-base coverage sets the ceiling.
- A blind nurse rated native RAG safer (overall safety 3.60 vs 3.05 closed book and 2.90 NLLB cascade, 40 health questions).

Production uses single-pass native RAG: one LLM call, a median of 7.3 s per answer on one GPU.

## Repository
| Path | Content |
|---|---|
| `src/gadaabot/agent` | LangGraph pipeline (switchable refine / grade / reformulate / verify), retriever, vLLM client |
| `src/gadaabot/serve` | production pipeline, FastAPI web chat and API, Telegram bot |
| `src/gadaabot/eval` | LLM judge, ablations, human-label analysis, retrieval evaluation |
| `src/gadaabot/kb`, `retrieval` | knowledge-base building, Oromo BM25 analyzer (4-letter prefixes) |
| `scripts/tasks` | the task scripts (T0xx) that produced every result in the paper |
| `data/qa` | 364 generated questions with native review (test 184, dev, dropped) |
| `data/real_questions` | 79 questions written by native speakers (W01 to W04) with English key points, plus the leakage screen |
| `data/human` | blind scores: sheet 1 (150 answers) and real questions (316 answers), two annotators; clinician safety ratings (120) |
| `data/kb` | manifest of all 13,409 sections (source, tier, URL); full text for open (tier A) sources only |
| `results` | tables, reports, system answers and judge scores (`answers/`, gzipped JSONL) |
| `MANIFEST_v1.0.sha256` | checksums of every released file |

## Run it
Requirements: one GPU with 24 GB or more (48 GB recommended), Python 3.11.
```bash
pip install -r requirements.txt -r requirements-serve.txt
# 1) models: AfriqueQwen3.5-9B-Instruct (generator), McGill-NLP/AfriE5-Large-instruct, BAAI/bge-reranker-v2-m3
# 2) index: build the knowledge base (src/gadaabot/kb) and start Qdrant (scripts/qdrant_start.sh)
vllm serve <path-to-AfriqueQwen3.5-9B-Instruct> --served-model-name gadaabot-gen --max-model-len 16384 --port 48000
GADAABOT_ROOT=$PWD uvicorn gadaabot.serve.api:app --host 0.0.0.0 --port 8080     # web chat at http://host:8080
TELEGRAM_TOKEN=... GADAABOT_API=http://127.0.0.1:8080 python -m gadaabot.serve.telegram_bot
```

## Safety
GadaaBot is an information tool, not a clinician. It cites sources, adds a disclaimer to health answers, never gives
medicine doses, sends an urgent-care message for danger-sign wording, and abstains when no reliable source is found.
In the clinician review 6 of 40 native answers were still rated potentially harmful, so a closed pilot with health
workers is required before open launch. See `docs/SYSTEM_CARD.md`.

## Licenses
Code: Apache-2.0. Data: see `DATA_LICENSE.md`. School textbooks are used in the assistant with the permission of
the Oromia Education Bureau and are **not** redistributed here.

## Acknowledgements
The Oromia Education Bureau (permission to use its textbooks and publications), and the anonymous second annotator,
question writers and nurse reviewer.
