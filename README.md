# GadaaBot 2.0: native Afaan Oromoo question answering for health and education

GadaaBot answers health and school questions in **Afaan Oromoo**, grounded in Afaan Oromoo sources, with citations,
health disclaimers and an urgent-care screen. Nothing is translated through English.

**Paper:** *Native Evidence, Not More Agents: A Controlled Study of Retrieval-Augmented Question Answering for
Afaan Oromoo Health and Education* (`paper/`).

## Key results (184 native-speaker-reviewed held-out questions, correctness 1 to 5)
| System | Judge | Human (blind) |
|---|---|---|
| Closed book (same 9B model) | 2.99 | 2.40 |
| Translate-everything RAG (NLLB-200) | 3.19 | 2.53 |
| Translate-everything RAG (model self-translates) | 3.80 | |
| **Native RAG (this system)** | **4.38** | **3.83** |
| Full agent (refine + grade + verify) | 3.93 | 3.33 |

Agentic refinement and grading did not improve correctness; LLM self-verification hurt it. Production therefore
uses single-pass native RAG (1 LLM call, about 7 s per answer on one GPU).

## Repository
| Path | Content |
|---|---|
| `src/gadaabot/agent` | LangGraph pipeline (switchable refine / grade / verify), retriever, vLLM client |
| `src/gadaabot/serve` | production pipeline, FastAPI web chat + API, Telegram bot |
| `src/gadaabot/eval` | LLM judge, held-out ablation, final analysis, retrieval evaluation |
| `src/gadaabot/kb`, `retrieval` | knowledge-base building, Oromo BM25 analyzer (4-letter prefixes) |
| `data/qa` | 364 generated questions with native review (split: test 184, dev, dropped) |
| `data/human` | blind correctness scores by two native speakers (150 answers, QWK 0.90) |
| `data/kb` | manifest of all 13,409 sections (source, tier, URL); full text for open (tier A) sources only |
| `results` | paper tables, held-out answers with judge scores |
| `paper` | LaTeX source (ACM TALLIP format) |

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
The danger-sign list needs clinical review before large-scale deployment. See `docs/SYSTEM_CARD.md`.

## Licenses
Code: Apache-2.0. Data: see `DATA_LICENSE.md`. School textbooks are used in the assistant with the permission of
the Oromia Education Bureau and are **not** redistributed here.

## Acknowledgements
Gemechu Bekele (independent annotation); the Oromia Education Bureau (textbook permission).
