# 📋 Compliance Copilot

A RAG chatbot that **answers questionnaires from your own documents** — and tells you
which questions your documents *cannot* answer.

Upload your policies, point a security questionnaire at them, and get a drafted answer
for every question with a verified citation — or a flagged gap that needs a human.

> New here? [**PRODUCT.md**](PRODUCT.md) explains what it does and why it's useful, with an example.

---

## Quick start

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

cp .env.example .env          # paste your key into it
# free key: https://aistudio.google.com/apikey

.venv/bin/python -m streamlit run app.py
```

Click **Try it with sample documents** → **Audit 12 questions**. No files needed.

---

## What you get

| Question | Verdict | Result |
|---|---|---|
| Is data encrypted at rest? | ✅ Covered | "Yes — AES-256 on all volumes." · *policy.txt* |
| Is pen testing done annually? | ⚠️ Partial | "A test ran in March 2026, but no annual cadence is stated." · *pentest.pdf, p.1* |
| Do you carry cyber insurance? | ❌ Gap | Not in your documents — needs a human. |

Then download the answers as CSV, or ask free-form follow-up questions in the chat.

---

## The pipeline

```
Upload → Parse → Chunk → Embed → Store → Retrieve → Generate → Verify
```

| Stage | How |
|---|---|
| **Upload** | Streamlit, multi-file — PDF, DOCX, TXT, MD |
| **Parse** | `pypdf` / `docx2txt`, page-aware so citations can say `p.4` |
| **Chunk** | Split at section headings, falling back to size-based |
| **Embed** | `gemini-embedding-2`, 3072-dim, asymmetric `task_type` |
| **Store** | Chroma — embedded, persists to `.chroma/` |
| **Retrieve** | Per-question top-k by cosine distance |
| **Generate** | Gemini, strict JSON verdict, then quote-verified |

---

## Coverage is verified, not claimed

Telling a model "don't hallucinate" is a request. This makes it checkable.

Every answer must include a **verbatim quote** from the retrieved text. That quote is
then searched for in the real chunks **by code**. No quote, or a quote that isn't
really there, and the verdict is forced down to `NOT_FOUND`.

So `COVERED` never means *"the model said so"*. It means *"the supporting sentence was
located in your document."* Tested against a deliberately lying model — invented
quotes, **paraphrased** quotes, and confident answers with no quote are all rejected.

---

## Design decisions

**Retrieval is per-question, not global.** One shared top-k across a 12-question
questionnaire would let one document crowd out the others, silently answering some
questions from the wrong source.

**Chunking follows section headings.** Policies are written as numbered sections; a
fixed-size splitter cuts across them, blurring each chunk's meaning and making
citations confusing. Measured: both approaches rank the right chunk first, but
section-aware splitting separates it from the runner-up far more confidently.

**Embeddings are asymmetric.** A question and the clause answering it aren't phrased
alike, so documents are embedded as `RETRIEVAL_DOCUMENT` and queries as
`RETRIEVAL_QUERY`.

**One collection per session.** Sessions once shared a collection, so a second user's
upload destroyed the first user's data and could serve them someone else's documents.
Each session now gets its own `kb_<uuid>`, published in the URL so a page refresh
recovers it without re-embedding. Abandoned collections are pruned after 24h.

**Questions run concurrently.** 4 workers took a 12-question audit from 48s → 11s,
with retry-with-backoff for free-tier rate limits.

---

## Tests

```bash
.venv/bin/python test_rag.py          # 60 offline checks, no API calls
.venv/bin/python test_rag.py --live   # + 12 checks against the real API
.venv/bin/python test_app.py          # 39 UI checks, drives the real app
```

`rag.py` has no Streamlit import, so the whole pipeline is testable without a browser.

---

## Files

```
app.py          Streamlit UI — presentation and state only
rag.py          the pipeline: parse, chunk, embed, store, retrieve, answer, verify
test_rag.py     pipeline checks
test_app.py     UI checks via Streamlit's AppTest
samples/        a fictional company's policy, architecture and pen-test report
```

---

## Deploying

Works on [Streamlit Community Cloud](https://share.streamlit.io) free tier (1 GB RAM;
this app peaks at ~175 MB). Point it at `app.py` and add `GOOGLE_API_KEY` under
**App settings → Secrets** — no code changes needed.

Note the container filesystem is rebuilt on reboot, so `.chroma/` is scratch space.

---

## Limitations

- **No OCR** — scanned PDFs are rejected rather than silently mis-answered.
- **Dense retrieval only** — adding keyword search would help on exact identifiers
  like "SOC 2 Type II".
- **No query rewriting** — terse questionnaire jargon would retrieve better if expanded.
- **Chat is strictly grounded** — ask "which gap is worst?" and it correctly says the
  documents don't say. Reasoning over findings would be a separate mode.
- **`PARTIAL` relies on model judgement** — the quote requirement still applies, so it
  can't be evidence-free, but the covered/partial line is a judgement call.
