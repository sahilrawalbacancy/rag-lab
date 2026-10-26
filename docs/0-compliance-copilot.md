# 📋 Compliance Copilot

A RAG chatbot that **answers questionnaires from your own documents** — and tells you
which questions your documents *cannot* answer.

Upload your policies, point a security questionnaire at them, and get a drafted answer
for every question with a verified citation — or a flagged gap that needs a human.

> New here? [**The product page**](0-compliance-copilot-product.md) explains what it does and why it's useful, with an example.

This was the initial chatbot. It now lives as page 0 of RAG Lab — setup is in
the [main README](../README.md).

---

## Quick start

Run the app, open **Compliance Copilot**, click **Try it with sample documents**
→ **Audit 12 questions**. No files needed.

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
| **Retrieve** | Per-question top-k by vector distance |
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

A real quote can still sit next to an invented answer ("7 years" beside a verified
"retained for 12 months"), so every number in the answer must also appear in a
passage the answer verifiably quoted (so a table header like "256GB" counts). Curly quotes, dashes and ligatures are folded before matching, so a
genuine quote is not rejected for typography.

A question that fails to run (API down, quota exhausted) is reported as 🛑 **Error**,
never as a gap — a gap claims the documents were checked.

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
.venv/bin/python tests/test_rag.py          # 80 offline checks, no API calls
.venv/bin/python tests/test_rag.py --live   # + 12 checks against the real API
.venv/bin/python tests/test_app.py          # 45 live UI checks (39 for this page)
```

`rag/` has no Streamlit import, so the whole pipeline is testable without a browser.

---

## Files

```
views/compliance_copilot.py   the page — presentation and state only
rag/core.py                   shared pipeline: parse, chunk, embed, store, retrieve, answer, verify
rag/compliance.py             questionnaire-only: presets, question extraction, batch audit, CSV
data/samples/                 a fictional company's policy, architecture and pen-test report
```

Uploads are indexed into `.chroma/`, which is scratch space: Streamlit Cloud
rebuilds the container filesystem on reboot.

---

## Limitations

- **The session URL is a key** — anyone given it can open (and replace) that
  session's documents for 24 hours. The page warns about this.

- **No OCR** — scanned PDFs are rejected rather than silently mis-answered.
- **Dense retrieval only** — adding keyword search would help on exact identifiers
  like "SOC 2 Type II".
- **No query rewriting** — terse questionnaire jargon would retrieve better if expanded.
- **Chat is strictly grounded** — ask "which gap is worst?" and it correctly says the
  documents don't say. Reasoning over findings would be a separate mode.
- **`PARTIAL` relies on model judgement** — the quote requirement still applies, so it
  can't be evidence-free, but the covered/partial line is a judgement call.
