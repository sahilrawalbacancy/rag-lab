# 📊 3 · Short, visual document

| | |
|---|---|
| **Document** | Apple — iMac Product Environmental Report, October 2024, 12 pages |
| **Shape** | Facts in charts, icon call-outs, small tables and footnotes — not paragraphs |
| **Challenge** | Pull the right number from a chart or table, not a general sentence nearby |
| **Code** | [`rag/visual.py`](../rag/visual.py), [`rag/vision.py`](../rag/vision.py) |

## The assignment

1. How much less energy does iMac use compared to the ENERGY STAR requirement?
2. What is the total carbon footprint of the iMac, and how much of that comes from product use versus manufacturing?
3. What material is the iMac stand made from, and what percentage of it is recycled?

**Pass bar:** the right number from a chart or table.

## What the document actually contains

Measured by running the PDF through the parser:

- **All three answers are in the text layer** — 58% less energy (p.7, qualified by
  footnote 18); 346 kg CO2e with Production 50% / Product use 45% (p.10 table); 100%
  recycled aluminium stand (p.1, p.2, p.4).
- **…but each one has a trap:**
  - **Wording:** the question says "manufacturing", the table says "Production".
  - **Two configurations:** two ports / 256GB = 346 kg and four ports / 512GB = 391 kg sit
    side by side, and the flattened text runs them together.
  - **Charts:** the p.2 carbon chart's labels and values come out of the text layer in a
    jumble (`–99 kg Emissions reductions 346 kg iMac emissions10 … 445 kg Baseline emissions7`).
  - **Footnotes are far from their claims:** claims sit on p.1–8, endnotes on p.11–12 —
    the "58% less" chunk never sees footnote 18's test conditions (143W adapter).
  - **Footnote markers glued to words:** `content1`, `STAR.18`, `energy2`, `emissions10`.
  - **Kerning splits words:** `T esting`, `T o`.
- The three sample questions barely test charts or footnotes, so the evaluation adds
  questions whose answers live *only* in a chart, a table cell, a call-out or an endnote.

## What was built, and why

| Stage | Choice | Why |
|---|---|---|
| Parse | **Every page transcribed by Gemini vision** into Markdown: charts as label/value bullets, tables with headers, footnote markers as `[^n]` | Keeps each value attached to its label, and separates markers from words |
| Vision check | Every number in the transcription must exist in the PDF's text layer | Vision is model-generated text. Result: **all 12 pages confirmed** (one apparent miss was the text layer's `2025.3` — the year plus a glued footnote marker) |
| Chunk | One chunk per heading block — a chart, a table, a call-out group; tables never split | A number never travels without its caption or header |
| Footnotes | Endnotes become their own chunks, **linked** to every chunk that cites them; when a claim is retrieved its footnotes come with it | The claim on p.7 and its conditions on p.12 reach the model together |
| Clean-up | A footnote defined twice keeps the longest definition | Vision once invented `[^7]: GHG reductions achieved` under a table; the real note 7 is on p.11 |
| Retrieve | Hybrid vectors + BM25, cross-encoder rerank | "230V", "kg CO2e", "ENERGY STAR" match by keyword |
| Answer | Every number must appear in a passage the answer quoted; one repair round | A plausible but wrong number is the main failure for this document |

## Results

Nine questions (3 from the brief, 6 traps), answer model `gemini-3.5-flash-lite` for every
variant. Full answers, quotes and pages: the app's **Evaluation** tab, or `data/results/imac.json`.

| Pipeline | Passed | Right evidence retrieved |
|---|---|---|
| Naive RAG (initial chatbot) | **8 · 8 · 7** (3 runs) | 8/8 |
| This strategy | **9 · 9 · 9** (3 runs) | 8/8 |
| − footnote links | 9 | 8/8 |
| − rerank | 9 | 8/8 |
| dense only (no BM25, no rerank) | 9 | 8/8 |

**What this shows**

- **The gain is the vision-built index.** Every ablation still scores 9/9: once the pages are
  transcribed into labelled charts, headed tables and separate footnotes, even plain dense
  retrieval finds the right chunk. The work is in parsing, not in retrieval tricks.
- **Naive never answers the chart question (imac-7)** in any run: from the text layer's jumble
  (`–99 kg Emissions reductions 346 kg iMac emissions10 … 445 kg Baseline emissions7`) it
  cannot tell which number is the baseline, and says so. The strategy reads
  "Baseline emissions: 445 kg" and "Emissions reductions: –99 kg" straight from the chart.
- **Naive is unstable on a table cell (imac-5, passed 2 of 3 runs).** In the text layer the
  row "Sleep 0.56W 0.57W 0.59W" and its "100V 115V 230V" header are separate lines, so the
  model has to work out which column is 230 V; the strategy's Markdown table names every column.
- **Footnote linking changed no score here** — the endnote question (imac-4, the 143 W adapter)
  is found directly by keyword search. Linking still matters for what the model reads: the
  "58% less energy" claim now arrives with its test conditions (see the context table in the
  Ask tab), which is the difference between a fact and a fact with its caveats.
- **Vision is checked, not trusted.** All 12 pages pass the number cross-check against the
  PDF's text layer; a stray footnote definition the model invented under a table is discarded.

**Lesson:** for a visual document, fix the representation first. Once each number is
attached to its label, header and footnote, retrieval becomes easy.
