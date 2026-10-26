# 🧪 RAG Lab

Retrieval-augmented generation, built four times: a first chatbot, then three
documents whose **shapes** force three different designs. One Streamlit app,
one page each, every design measured against the same naive baseline.

| Page | Document | Shape | What it forces |
|---|---|---|---|
| 📋 [Compliance Copilot](docs/0-compliance-copilot.md) | your uploads | policies and questionnaires | answers checked against verbatim quotes |
| 📄 [1 · GPO](docs/1-gpo.md) | 2018 Annual Report, 82 pp | narrative text | clean text, precise retrieval |
| 📚 [2 · World Bank](docs/2-world-bank.md) | Global Economic Prospects, 200 pp | chapters, sections, tables | a section map, section-aware retrieval, vision for tables |
| 📊 [3 · Apple iMac](docs/3-imac.md) | Environmental Report, 12 pp | charts, tables, footnotes | vision transcription, footnote linking |

Each assignment page has three tabs: **Ask** (with a side-by-side naive
comparison and the exact passages the model saw), **How it works** (what was
measured in the document and why the design follows), and **Evaluation**
(every variant, question by question).

## Results

Each pipeline answered every question three times (`gemini-3.5-flash-lite`); a question
passes only with the expected facts, a quote verified word for word in the document,
and that quote on the right page or section.

| Document | Naive RAG | This strategy | What made the difference |
|---|---|---|---|
| GPO (narrative) | 10 · 10 · 10 of 10 | 9 · 10 · 10 of 10 | Nothing beyond naive is needed — the baseline case. Merged neighbours fix sentences cut at chunk boundaries. |
| World Bank (long, structured) | 5 · 5 · 4 of 8 | **7 · 8 · 7** of 8 | Section-aware chunks and context: naive never passes the executive-summary, 2026-vs-average or annex questions. Reranking finds table rows. |
| iMac (visual) | 8 · 8 · 7 of 9 | **9 · 9 · 9** of 9 | Vision transcription: naive cannot read the carbon chart's values. |

The write-ups ([GPO](docs/1-gpo.md) · [World Bank](docs/2-world-bank.md) ·
[iMac](docs/3-imac.md)) explain each result, including what did *not* help.

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env          # add GOOGLE_API_KEY - free key: https://aistudio.google.com/apikey
.venv/bin/python -m streamlit run streamlit_app.py
```

`data/chroma/` (the vector database) and `data/results/` (the evaluation)
are committed, so the app runs straight away.

## Rebuild the index and the scores

```bash
.venv/bin/python ingest.py                     # PDFs -> data/chroma/  (local, no API quota)
.venv/bin/python evaluate.py --model gemini-3.5-flash-lite          # every variant, one answer model
.venv/bin/python evaluate.py gpo --chunk-sizes 300,500,1000         # + the chunk-size experiment
```

Vision transcriptions are cached in `data/vision/`; ingestion only calls
Gemini for a page that has no cached Markdown.

**Free-tier limits shape the design.** Every Gemini model has its own daily
cap (500 requests for the Flash-Lite models, 20 for Flash), embeddings count
every text as a request, and busy models answer 503. So: embeddings and
reranking run locally; vision runs once and is cached; the app's answers fall
back down `CHAT_PREFERENCE` when a model is spent; and an evaluation pins ONE
answer model (`--model`, or `GEMINI_CHAT_MODEL`) so its variants stay
comparable.

## How it's built

```
streamlit_app.py       navigation: Overview · Compliance Copilot · 1 · 2 · 3
ui.py                  shared page layout (Ask / How it works / Evaluation) and cached clients
views/                 one thin file per page
rag/core.py            parsing, cleaning, the quote- and number-checked answer
rag/retrieval.py       local embeddings, BM25, fusion, rerank, context expansion, ask()
rag/narrative.py       assignment 1 strategy (GPO)
rag/structured.py      assignment 2 strategy (World Bank): section map, routing, tables
rag/visual.py          assignment 3 strategy (iMac): vision pages, footnote links
rag/vision.py          Gemini page -> Markdown, cached, numbers cross-checked against the PDF
rag/evaluation.py      scoring: facts + grounding + right page, no LLM judge
rag/compliance.py      questionnaire logic for the Compliance Copilot page
ingest.py, evaluate.py command-line entry points
data/                  PDFs, vision cache, vector database, eval set, results
docs/                  one write-up per page, also shown inside the app
tests/                 offline checks + live UI checks
```

Everything runs inside the one Streamlit app - frontend, retrieval, the
vector database and the embedding/reranking models. The only outside call is
Gemini, for the final answer. `rag/` never imports Streamlit.

## Tests

```bash
.venv/bin/python tests/test_assignments.py    # assignments, offline (local models, fake LLM)
.venv/bin/python tests/test_rag.py            # shared core + Compliance Copilot, offline
.venv/bin/python tests/test_rag.py --live     # + checks against the real API
.venv/bin/python tests/test_app.py            # live UI checks, all pages
```

## Deploy (Streamlit Community Cloud)

Main file `streamlit_app.py`, Python 3.12, `GOOGLE_API_KEY` under
**App settings → Secrets**. Keep the repo and the app private: the Apple
report is "All rights reserved". The GPO report is a U.S. government work and
the World Bank report is CC BY 3.0 IGO.

On first use the app downloads the two local models (~190 MB) from Hugging
Face; loaded, they add ~415 MB of memory.
