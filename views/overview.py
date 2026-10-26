"""Overview - what this app is, and the scorecard across all three documents."""

import pandas as pd
import streamlit as st

import ui
from rag.assignments import registry

st.title("🧪 RAG Lab")
st.markdown(
    """
Retrieval-augmented generation, built four times: a first chatbot, then three
documents whose **shapes** force three different designs.

**Document → Parse → Chunk → Embed → Retrieve → Rerank → Context → LLM → Evaluate**
"""
)

SHAPES = {
    "gpo": ("📄 1 · GPO Annual Report", "82 pp · narrative text",
            "Nothing beyond naive is needed - narrative text is the baseline case. Merging "
            "neighbours fixes answers whose sentence straddles a chunk boundary."),
    "gep": ("📚 2 · World Bank GEP", "200 pp · chapters, sections, tables",
            "Section-aware context: naive answers the executive-summary question from "
            "Chapter 1, East Asia's 2026 forecast with the 2026-28 average, and never finds "
            "the annex. Reranking is what finds the right table row (Iraq)."),
    "imac": ("📊 3 · Apple iMac Report", "12 pp · charts, tables, footnotes",
             "Vision transcription: the text layer jumbles the carbon chart's labels and "
             "values, so naive cannot say what the baseline was."),
}

rows = []
for doc_id, strategy in registry().items():
    data = ui.results(doc_id)
    name, shape, difference = SHAPES[doc_id]
    if not data or "advanced" not in data["variants"] or "naive" not in data["variants"]:
        rows.append({"document": name, "shape": shape, "naive RAG": "not run", "this strategy": "not run"})
        continue
    v = data["variants"]
    n, a = v["naive"]["summary"], v["advanced"]["summary"]
    stability = data.get("stability", {})

    def score(name: str, summary: dict) -> str:
        runs = stability.get(name, {}).get("passed")
        if runs and len(runs) > 1:
            return f"{min(runs)}–{max(runs)}/{summary['total']} over {len(runs)} runs"
        return f"{summary['passed']}/{summary['total']}"

    rows.append({
        "document": name, "shape": shape,
        "naive RAG": score("naive", n),
        "this strategy": score("advanced", a),
        "right evidence retrieved": f"{n['retrieved_expected']} → {a['retrieved_expected']} of {a['retrieval_scored']}",
        "what made the difference": difference,
        "strategy": strategy.SETTINGS.describe(),
    })

st.markdown("### Scorecard")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(
    "A question passes only if the answer states every expected fact, is backed by a quote "
    "found word for word in the document, and that quote comes from the right page or "
    "section. Questions, expected answers and pages: `data/eval_set.json`."
)

st.markdown(
    """
### What stays the same on every page
Local embeddings (`bge-small-en-v1.5`) and a local reranker
(`ms-marco-MiniLM-L-12-v2`) running inside this app, a Chroma vector store
built offline, and Gemini only for the final answer. Every answer must quote
the document word for word; the quote is searched for in code, and every
number in the answer must appear in a passage it quoted. Fail either and the
answer is withheld.

### What changes per document
Parsing, chunking, retrieval and context: exactly the stages where the
documents differ. Each page's **How it works** tab shows what was measured in
that document and why its design follows; its **Evaluation** tab shows every
variant, question by question.
"""
)
