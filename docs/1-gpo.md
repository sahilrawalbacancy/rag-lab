# 📄 1 · Plain-text business report

| | |
|---|---|
| **Document** | U.S. Government Publishing Office — 2018 Annual Report, 82 pages |
| **Shape** | Ordinary narrative paragraphs, with numbers written into the sentences |
| **Role** | The baseline case the other two documents are compared against |
| **Code** | [`rag/narrative.py`](../rag/narrative.py) |

## The assignment

1. How many depository libraries did GPO staff visit in FY 2018?
2. What new digital system replaced FDsys, and when was it fully deployed?
3. How much did GPO procure in printing and information products in FY 2018 on behalf of federal agencies?

**Pass bar:** the exact fact from the surrounding paragraph, with nothing added.

## What the document actually contains

Measured by running the PDF through the parser:

- **All three answers are in plain text** — 166 libraries (p.5, p.21); govinfo replaced
  FDsys, deployed during FY 2018 with FDsys retired December 2018 (p.4, p.8, p.10, p.22);
  more than $375 million procured (p.5; $375.7M on p.14, p.15, p.17).
- **Facts repeat.** The director's message, the program sections, the management
  discussion and the financial statements restate the same numbers — the evaluation
  accepts every page a fact genuinely appears on.
- **Look-alike numbers.** "$58.9 million" is FY 2017 net income on p.26 *and* an unrelated
  cash increase on p.40 and p.69. Operating revenue was $874.5M in FY 2018 and $874.3M in
  FY 2017, in the same sentence.
- **Noise.** The running header "U.S. GOVERNMENT PUBLISHING OFFICE | 2018 ANNUAL REPORT"
  repeats on 14 pages and was embedded into every chunk it touched.
- **Charts without values.** The p.48 "Smartcards Produced" chart prints no data labels —
  even Gemini vision correctly refuses to read numbers off it, so no question depends on it.

## What was built, and why

| Stage | Choice | Why |
|---|---|---|
| Parse | Unicode normalisation; running headers, footers and page numbers stripped by repetition | Noise in every chunk pulls unrelated chunks towards any question that mentions GPO |
| Chunk | 500-character chunks (chosen by experiment, below) | One chunk ≈ one fact, so the vector matches the question precisely |
| Context | **Sentence window**: each hit is merged with its neighbours on the same page into one continuous passage | Precision without losing context — and a sentence that straddles a chunk boundary arrives whole, so it can be quoted |
| Retrieve | Hybrid: local vectors + BM25, fused by reciprocal rank | Exact tokens — "FDLP", "FY 2017", numbers — match by keyword |
| Rerank | Local cross-encoder over the top 20 | Scores the question and the passage together |
| Answer | Quote-checked, number-checked, one repair round | A withheld answer gets the check's reason once and tries again — the check never relaxes |

Vision was **not** used here: the only visual content is unlabelled charts, and measured
vision output for them contains no numbers to gain.

## Results

Ten questions (3 from the brief, 7 traps), answer model `gemini-3.5-flash-lite` for every
variant. Full answers, quotes and pages: the app's **Evaluation** tab, or `data/results/gpo.json`.

| Pipeline | Passed | Right evidence retrieved |
|---|---|---|
| Naive RAG (initial chatbot) | **10 · 10 · 10** (3 runs) | 9/9 |
| This strategy | **9 · 10 · 10** (3 runs) | 9/9 |
| − neighbours | 10 | 9/9 |
| − rerank | 9 | 9/9 |
| dense only (no BM25, no rerank) | 10 | 9/9 |

| Chunk size | Chunks | Passed |
|---|---|---|
| 300 | 754 | 10/10 |
| **500** (kept) | 451 | 10/10 |
| 1,000 | 232 | 10/10 |

**What this shows**

- **Narrative text is where naive RAG is already strong.** With the quote and number
  checks, the initial chatbot's pipeline answers all ten — including the look-alike-number
  traps (FY 2017 vs FY 2018 net income; $874.5M vs $874.3M). There is no headroom for the
  strategy to win; the job here is not to make things worse.
- **Chunk boundaries are the narrative failure mode.** With the earlier answer model
  (`gemini-3.1-flash-lite`), naive and "− neighbours" both failed gpo-1: the sentence
  "GPO staff visited 166 depository libraries" straddled two chunks, the model quoted it
  whole, and the quote check rejected it. Merging neighbours into one passage fixed it.
- **Chunk size did not matter** between 300 and 1,000 characters for these questions; 500
  is kept as the middle ground — precise matching, with neighbours supplying context.
- **The one unstable question (gpo-2, 2 of 3 runs)** is a wording mismatch: the question
  says "fully deployed", the page that answers it says "full-scale deployment", and that
  passage never reaches either pipeline's top 5. The report also gives three plausible dates
  (out of beta December 2017, full deployment in FY 2018, FDsys retired December 2018).
  Query rewriting would be the next technique to try.

**Lesson:** measure the baseline before adding machinery. For plain narrative text, clean
input and a strict answer check matter more than clever retrieval.
