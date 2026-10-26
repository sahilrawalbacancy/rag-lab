# 📚 2 · Long, structured report

| | |
|---|---|
| **Document** | World Bank — Global Economic Prospects, June 2026, 200 pages |
| **Shape** | Chapters → sections → boxed articles, plus forecast tables and a statistical annex |
| **Challenge** | Pull the *right section*, not a page that happens to share a few words |
| **Code** | [`rag/structured.py`](../rag/structured.py) |

## The assignment

1. What is the growth outlook for South Asia in this report?
2. What does Chapter 3 say about sovereign debt in developing economies?
3. According to the executive summary, what are the main risks to the global outlook?

**Pass bar:** the right chapter and section, with no facts mixed in from another region or chapter.

## What the document actually contains

Measured by running the PDF through the parser:

- **Size:** 185 pages with text (the other 15 are blank, so no OCR is needed),
  708,000 characters, ~950 chunks.
- **Every question's topic is scattered** — exactly the "mixed up" failure the pass bar describes:
  - "South Asia" appears on 16 pages; the South Asia section is only p.96–98.
  - "Sovereign debt" appears on 21 pages, 10 of them outside Chapter 3 (p.107–142).
  - "Risk" appears on 88 pages; the executive summary is only p.19–20, and it restates
    Chapter 3's findings almost word for word.
  - East Asia's growth is "4.2 percent in 2026" — and, one paragraph later, "4.3 percent on
    average over 2026–28".
- **Structure is available, but imperfect:**
  - 18 PDF bookmarks: Executive Summary, Chapters 1–4, Boxes 1.1 / 1.2 / 2.1 / 3.1, six
    regions, Statistical Annex.
  - They are flat, out of order, and sometimes a page early (South Asia points to p.95; its
    heading is on p.96). Boxes have no end — the bookmark alone would make Box 1.1 run
    p.30–43 instead of p.30–35.
  - The abbreviations page (p.21) sits inside the executive summary's bookmark range.
- **Text artefacts:** 1,559 ligature characters (`ﬁ ﬂ ﬀ`) and drop-cap junk (`7T_he`, `7'_he`).
- **Tables are mangled by plain extraction:**
  - footnote markers sit inside rows and shift every column — `Iraq 1 0.5 -1.5 -2.2 -8.9 …`
    has 9 values under 8 headers (the annex row has two markers: `Iraq 2 5 0.5 …`);
  - p.98's 2028 column is torn off and dumped below the table;
  - the regression tables split numbers with stray spaces (`3. 020`, `-7.8 44`);
  - each forecast table repeats `2026f | 2027f` — the second pair is the *change* from the
    January forecast, not the forecast.
- **Page numbers:** PDF page 96 is printed page 74.

## What was built, and why

| Stage | Choice | Why |
|---|---|---|
| Structure | A section map from the bookmarks: sorted, each snapped to the page where its heading really appears; boxes end where their "BOX n.n … (continued)" headers stop; the abbreviations page split out | The structure exists — it just has to be corrected before anything can rely on it |
| Parse | Ligatures unfolded (NFKC), drop caps repaired, running headers stripped; **32 table and annex pages transcribed by Gemini vision** | Plain extraction cannot keep a table's columns aligned; vision can |
| Vision check | Every number in every transcription must exist in the PDF's own text layer (split numbers rejoined pairwise) | Vision output is model-generated text. Result: **all 32 pages confirmed** |
| Chunk | Text in ~1,000-character chunks; tables split into row groups that each repeat the title and header; repeated `2026f` columns relabelled "change vs January 2026 forecast" | A row is meaningless without the years above it, and the duplicate header is a trap |
| Hierarchy | Every chunk carries chapter / region / box metadata **and** a breadcrumb line ("Chapter 2 · Regional Outlooks › South Asia") in its text | The location reaches the embedding, the keyword index and the citation |
| Route | A question that names a region, chapter, box, the executive summary or the annex is searched only inside it, and always sees that section's opening chunk | Chapter 3 opens with its own abstract (p.109) — the best answer to "what does Chapter 3 say" |
| Retrieve | Hybrid vectors + BM25, cross-encoder rerank, ±1 neighbour merged into one passage | |
| Cite | PDF page *and* printed page ("p.96 (printed 74) · Chapter 2 · Regional Outlooks › South Asia") | A reader can find it in a PDF viewer or on paper |

**A deliberate change from the plan:** no LLM-written section summaries. The report already
opens each chapter with its own abstract, and that text can be *quoted and verified*; a
generated summary cannot be cited as evidence. The "section opening" rule uses the
document's own summaries instead.

## Results

Eight questions (3 from the brief, 5 traps), answer model `gemini-3.5-flash-lite` for every
variant. Full answers, quotes and pages: the app's **Evaluation** tab, or `data/results/gep.json`.

| Pipeline | Passed | Right evidence retrieved |
|---|---|---|
| Naive RAG (initial chatbot) | **5 · 5 · 4** (3 runs) | 5/7 |
| This strategy | **7 · 8 · 7** (3 runs) | 7/7 |
| − routing and section opening | 7 | 7/7 |
| − rerank | 6 | 6/7 |
| dense only (no BM25, no rerank) | 6 | 6/7 |

**Naive RAG fails the brief's traps in every run:**

| Question | What naive did | The strategy |
|---|---|---|
| gep-3 "According to the executive summary…" | Never retrieved the executive summary — its context was Chapter 1's risk pages (p.29, p.80), the right topic in the wrong section — so it found no answer | Cites p.19 |
| gep-4 East Asia's 2026 forecast | "4.3 percent … over 2026–28" — the multi-year average from the next sentence | 4.2% for 2026 |
| gep-7 euro area growth in the annex | Never retrieved the annex; answered "not in the document" | Cites the annex table, p.187 |

**What this shows**

- **Knowing where text sits is the gain.** Breadcrumbs in every chunk, section metadata
  and the section-opening rule turn "the right topic" into "the right section".
- **Routing as a hard filter added nothing measurable on top of breadcrumbs.** Because every
  chunk's text starts with its location, BM25 and the reranker already prefer chunks from
  a section the question names. The filter remains as a guarantee — and it is what the
  app shows when it says "search restricted to South Asia".
- **Reranking is what finds a table row.** Without it, gep-5 (Iraq's 2026 forecast) fails:
  the right row never reaches the top 5.
- **Vision fixed the tables.** Iraq's row is `| Iraq [^1] | 0.5 | -1.5 | -2.2 | -8.9 | …`,
  with −8.9 under 2026f, instead of a row whose columns are shifted by a footnote marker.
  Every number on all 32 transcribed pages is confirmed by the PDF's own text layer.
- **The one unstable question (gep-2, 1 of 3 runs)** is a summary over 36 pages. Retrieval
  was right every time (Chapter 3's abstract, p.109, retrieved and quoted), but in two runs
  the four-sentence answer named fewer than three of the chapter's key findings. That is
  the answer model, not retrieval.

**Lesson:** for a long, structured document, structure is retrieval. Correct the
structure the PDF already has (bookmarks, box headers), attach it to every chunk, and give
tables to a model that can see them.
