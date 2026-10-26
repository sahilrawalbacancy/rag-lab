"""
Shared Streamlit pieces: the API key, cached clients, the vector database, and
the three-tab layout every assignment page uses (Ask / How it works /
Evaluation). Pages stay thin; all RAG logic lives in rag/.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag import core, evaluation
from rag.assignments import CHROMA_DIR, RESULTS_DIR, ROOT, VISION_DIR, registry
from rag.retrieval import NAIVE, Index, ask, open_client

# find_dotenv() inspects caller stack frames and breaks under Streamlit's
# script runner, so the path is always explicit.
load_dotenv(ROOT / ".env")

VERDICT = {
    core.COVERED: "✅ **Answered from the document**",
    core.PARTIAL: "⚠️ **Partly answered**",
    core.NOT_FOUND: "❌ **Not in the document**",
    core.ERROR: "🛑 **Could not run**",
}


# ---------------------------------------------------------------------------
# keys, models, data
# ---------------------------------------------------------------------------


def api_key() -> str:
    """
    Streamlit secrets first, then the environment.

    Locally the key lives in .env; on Streamlit Community Cloud it is injected
    via st.secrets. Reading st.secrets raises when no secrets file exists at
    all, so it is guarded.
    """
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass
    return (os.environ.get("GOOGLE_API_KEY") or "").strip()


@st.cache_resource(show_spinner=False)
def get_llm(model: str, key: str, temperature: float = 0.0):
    """
    The answer model, falling back down CHAT_PREFERENCE when a call fails.

    On the free tier every model has its own DAILY cap (500 requests for
    gemini-3.1-flash-lite, 20 for the Flash models) and goes away with a 503
    on busy days. A deployed app cannot stop answering because one model is
    spent, so each call tries the next model instead. Cached so a rerun
    reuses one client.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    def client(name: str):
        return ChatGoogleGenerativeAI(model=name, temperature=temperature, google_api_key=key)

    return client(model).with_fallbacks([client(m) for m in core.CHAT_PREFERENCE if m != model])


@st.cache_data(show_spinner=False, ttl=3600)
def cached_models(key: str) -> tuple[str, str]:
    return core.resolve_models(key)


@st.cache_resource(show_spinner="Opening the vector database...")
def database():
    return open_client(CHROMA_DIR)


@st.cache_resource(show_spinner=False)
def load_index(name: str) -> Index:
    return Index.load(database(), name)


@st.cache_resource(show_spinner="Loading the local embedding and reranking models (first use only)...")
def warm_models() -> bool:
    from rag import retrieval

    retrieval.embed_query("warm up")
    retrieval.rerank_scores("warm up", ["warm up"])
    return True


def answer_to_dict(answer) -> dict:
    f = answer.finding
    return {
        "verdict": f.verdict, "answer": f.answer, "note": f.note, "route": answer.route,
        "seconds": answer.seconds,
        "citations": [{"quote": c.quote, "label": c.label()} for c in f.citations],
        "context": [{
            "why": h.why, "page": h.doc.metadata["page"], "section": h.doc.metadata.get("section", ""),
            "kind": h.doc.metadata.get("kind", ""), "dense": h.dense_rank, "bm25": h.sparse_rank,
            "rerank": None if h.rerank is None else round(h.rerank, 2), "text": h.doc.page_content,
        } for h in answer.context],
    }


@st.cache_data(show_spinner=False, max_entries=500)
def cached_answer(doc_id: str, variant: str, question: str, chat_model: str, key: str) -> dict:
    """One answer per (document, variant, question): repeat demo clicks cost no quota."""
    strategy = registry()[doc_id]
    settings = NAIVE if variant == "naive" else strategy.SETTINGS
    try:
        answer = ask(load_index(f"{doc_id}__{variant}"), question, get_llm(chat_model, key),
                     settings, strategy.route)
        return answer_to_dict(answer)
    except Exception as exc:
        return {"verdict": core.ERROR, "answer": "", "note": f"{type(exc).__name__}: {exc}",
                "route": "", "seconds": 0, "citations": [], "context": []}


def results(doc_id: str) -> dict | None:
    path = RESULTS_DIR / f"{doc_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_answer(d: dict, title: str = "") -> None:
    if title:
        st.markdown(f"#### {title}")
    st.markdown(VERDICT[d["verdict"]])
    if d["answer"]:
        st.markdown(d["answer"])
    if d["citations"]:
        st.caption("**Evidence** - quoted from the document and checked to be there")
        for c in d["citations"]:
            st.markdown(f"> {c['quote']}")
            st.caption(f"↳ {c['label']}")
    if d["note"]:
        st.caption(f"ℹ️ {d['note']}")
    with st.expander(f"What the model was given ({len(d['context'])} passages, {d['seconds']}s)"):
        if d["route"]:
            st.info(f"Search restricted to **{d['route']}** - the section the question names.")
        if d["context"]:
            st.dataframe(pd.DataFrame([{
                "#": i, "why": c["why"], "page": c["page"], "section": c["section"][:50],
                "kind": c["kind"], "dense rank": c["dense"], "BM25 rank": c["bm25"],
                "rerank score": c["rerank"], "text": c["text"][:160].replace("\n", " "),
            } for i, c in enumerate(d["context"], 1)]), hide_index=True, width="stretch")


def _ask_tab(doc_id: str) -> None:
    key = api_key()
    if not key:
        st.warning("Add `GOOGLE_API_KEY` to `.env` (or Streamlit secrets) to ask questions. "
                   "The other two tabs work without it.")
        return
    try:
        chat_model, _ = cached_models(key)
    except Exception as exc:
        st.error(f"Could not reach the Gemini API: `{exc}`")
        return
    warm_models()

    specs = evaluation.questions(doc_id)
    picked = st.pills(
        "Sample questions - the three from the brief first, then the traps",
        [q["question"] for q in specs], key=f"{doc_id}_pick",
    )
    with st.form(f"{doc_id}_form", clear_on_submit=False):
        typed = st.text_input("Or ask your own", key=f"{doc_id}_typed",
                              placeholder="Ask anything about this document...")
        compare = st.toggle("Compare with naive RAG (the initial chatbot's pipeline)", value=True,
                            key=f"{doc_id}_compare")
        submitted = st.form_submit_button("Ask", type="primary")
    question = (typed.strip() if submitted and typed.strip() else picked)
    if not question:
        st.caption("Pick a sample question or type one.")
        return

    st.markdown(f"**Q:** {question}")
    with st.spinner("Searching the document and checking the answer..."):
        advanced = cached_answer(doc_id, "advanced", question, chat_model, key)
        naive = cached_answer(doc_id, "naive", question, chat_model, key) if compare else None
    if naive is None:
        render_answer(advanced)
        return
    left, right = st.columns(2)
    with left:
        render_answer(advanced, "This document's strategy")
    with right:
        render_answer(naive, "Naive RAG")


def _mark(record: dict | None) -> str:
    if record is None:
        return ""
    return "✅" if record["passed"] else ("🛑" if record.get("error") else "❌")


def _evaluation_tab(doc_id: str) -> None:
    data = results(doc_id)
    if not data:
        st.info("No results yet - run `.venv/bin/python evaluate.py`.")
        return
    variants = data["variants"]
    naive, advanced = variants.get("naive"), variants.get("advanced")
    if naive and advanced:
        n, a = naive["summary"], advanced["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Naive RAG", f"{n['passed']}/{n['total']}")
        c2.metric("This strategy", f"{a['passed']}/{a['total']}", delta=a["passed"] - n["passed"])
        c3.metric("Right evidence retrieved", f"{a['retrieved_expected']}/{a['retrieval_scored']}",
                  delta=a["retrieved_expected"] - n["retrieved_expected"])
        c4.metric("Seconds per answer", a["avg_seconds"])
    st.caption(f"Answer model `{data['chat_model']}` · embeddings `{data['embed_model']}` · "
               f"reranker `{data['rerank_model']}` · run {advanced['summary']['when'] if advanced else ''}")

    stability = data.get("stability")
    if stability:
        st.markdown("##### Repeated runs - how much one run can be trusted")
        st.caption("The answer model is not perfectly deterministic, so naive and the strategy "
                   "were run several times. A difference smaller than the spread is noise.")
        st.dataframe(pd.DataFrame([{
            "variant": name, "runs": len(s["passed"]),
            "passed in each run": " · ".join(f"{p}/{s['total']}" for p in s["passed"]),
            "not stable (passed in some runs only)": ", ".join(
                f"{q} ({n}/{len(s['passed'])})" for q, n in s["per_question"].items()
                if 0 < n < len(s["passed"])) or "—",
            "never passed": ", ".join(q for q, n in s["per_question"].items() if n == 0) or "—",
        } for name, s in stability.items()]), hide_index=True, width="stretch")

    st.markdown("##### Every variant - each ablation is the strategy minus one technique")
    st.dataframe(pd.DataFrame([{
        "variant": name, "pipeline": v["description"], "index": v["index"].split("__")[1],
        "passed": f"{v['summary']['passed']}/{v['summary']['total']}",
        "right evidence retrieved": f"{v['summary']['retrieved_expected']}/{v['summary']['retrieval_scored']}",
        "grounded": v["summary"]["grounded"], "s/answer": v["summary"]["avg_seconds"],
    } for name, v in variants.items()]), hide_index=True, width="stretch")

    st.markdown("##### Question by question")
    by_id = {name: {q["id"]: q for q in v["questions"]} for name, v in variants.items()}
    rows = []
    for spec in evaluation.questions(doc_id):
        row = {"id": spec["id"], "kind": spec["kind"], "question": spec["question"], "tests": spec["tests"]}
        for name in variants:
            row[name] = _mark(by_id[name].get(spec["id"]))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    for spec in evaluation.questions(doc_id):
        a = by_id.get("advanced", {}).get(spec["id"])
        n = by_id.get("naive", {}).get(spec["id"])
        with st.expander(f"{_mark(a)} {spec['id']} · {spec['question']}"):
            cols = st.columns(2)
            for col, label, rec in ((cols[0], "This strategy", a), (cols[1], "Naive RAG", n)):
                with col:
                    st.markdown(f"**{label}** {_mark(rec)}")
                    if rec is None:
                        continue
                    if rec.get("error"):
                        st.error(rec["error"])
                        continue
                    st.markdown(rec["answer"] or "_(no answer)_")
                    for c in rec["citations"]:
                        st.caption(f"> {c['quote'][:200]}  \n↳ {c['label']}")
                    if rec["missing"]:
                        st.caption("Missing: " + "; ".join(rec["missing"]))
                    if not rec["evidence"]:
                        st.caption(f"Evidence came from p.{rec['cited_pages']}, not an expected page.")

    _rerun_live(doc_id)

    experiments = data.get("experiments", {}).get("chunk_size")
    if experiments:
        st.markdown("##### Chunk-size experiment")
        st.dataframe(pd.DataFrame([{
            "chunk size (chars)": int(size), "chunks": e["chunks"],
            "passed": f"{e['summary']['passed']}/{e['summary']['total']}",
            "right evidence retrieved": f"{e['summary']['retrieved_expected']}/{e['summary']['retrieval_scored']}",
        } for size, e in experiments.items()]), hide_index=True)


def _rerun_live(doc_id: str) -> None:
    """Reproduce the naive and strategy scores in front of the viewer (not saved)."""
    n = len(evaluation.questions(doc_id))
    st.markdown("##### Reproduce it live")
    key = api_key()
    if not st.button(f"Re-run naive + this strategy now ({2 * n} answers, a few minutes on a free key)",
                     key=f"{doc_id}_rerun", disabled=not key):
        return
    strategy = registry()[doc_id]
    chat_model, _ = cached_models(key)
    llm = get_llm(chat_model, key)
    warm_models()
    rows = []
    with st.status("Running every question through both pipelines...", expanded=True) as status:
        for name, variant, settings in (("naive", "naive", NAIVE), ("advanced", "advanced", strategy.SETTINGS)):
            st.write(f"{name}...")
            result = evaluation.run_variant(doc_id, load_index(f"{doc_id}__{variant}"), settings, llm, strategy.route)
            s = result["summary"]
            saved = (results(doc_id) or {}).get("variants", {}).get(name, {}).get("summary", {})
            rows.append({"variant": name, "live": f"{s['passed']}/{s['total']}",
                         "saved run": f"{saved.get('passed', '?')}/{saved.get('total', '?')}",
                         "passed": ", ".join(q["id"] for q in result["questions"] if q["passed"])})
        status.update(label="Done", state="complete")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Small run-to-run differences are expected: the answer model is not perfectly "
               "deterministic even at temperature 0.")


def chunk_browser(doc_id: str) -> None:
    """The same page, chunked naively and by this document's strategy."""
    naive, advanced = load_index(f"{doc_id}__naive"), load_index(f"{doc_id}__advanced")
    pages = sorted({d.metadata["page"] for d in advanced.docs})
    page = st.select_slider("Page", options=pages, key=f"{doc_id}_chunk_page")
    left, right = st.columns(2)
    for col, label, index in ((left, "Naive chunks", naive), (right, "This strategy's chunks", advanced)):
        with col:
            docs = [d for d in index.docs if d.metadata["page"] == page]
            st.markdown(f"**{label}** - {len(docs)} on p.{page}")
            for d in docs:
                meta = d.metadata
                tags = " · ".join(filter(None, [meta.get("kind"), meta.get("origin"), meta.get("section")[:40],
                                                f"footnotes {meta['footnotes']}" if meta.get("footnotes") else ""]))
                st.caption(tags)
                st.code(d.page_content, language=None, wrap_lines=True)


@st.cache_data(show_spinner=False)
def text_layer(doc_id: str) -> dict[int, str]:
    strategy = registry()[doc_id]
    return dict(core.parse_document(strategy.PDF.read_bytes(), strategy.PDF.name))


def vision_compare(doc_id: str) -> None:
    """What the PDF's text layer holds vs what vision transcribed - and any invented number."""
    from rag import vision

    files = sorted((VISION_DIR / doc_id).glob("*.md"))
    if not files:
        st.info("No vision pages for this document.")
        return
    pages = [int(f.stem) for f in files]
    page = st.selectbox("Page sent to vision", pages, key=f"{doc_id}_vision_page")
    markdown = (VISION_DIR / doc_id / f"{page:03d}.md").read_text()
    raw = text_layer(doc_id).get(page, "")
    invented = vision.unconfirmed_numbers(markdown, raw)
    if invented:
        st.warning(f"Numbers in the transcription that are not in the PDF's text layer: {', '.join(invented)}")
    else:
        st.success("Every number in the transcription also appears in the PDF's own text layer.")
    left, right = st.columns(2)
    with left:
        st.markdown("**PDF text layer (what pypdf extracts)**")
        st.code(raw, language=None, wrap_lines=True)
    with right:
        st.markdown("**Vision transcription (Markdown)**")
        st.code(markdown, language="markdown", wrap_lines=True)


def assignment_page(doc_id: str, writeup: str, explorer=None) -> None:
    strategy = registry()[doc_id]
    st.title(strategy.TITLE)
    st.caption(f"Strategy: {strategy.SETTINGS.describe()}")
    ask_tab, how_tab, eval_tab = st.tabs(["💬 Ask", "🔍 How it works", "📏 Evaluation"])
    with ask_tab:
        _ask_tab(doc_id)
    with how_tab:
        st.markdown((ROOT / "docs" / writeup).read_text())
        if explorer:
            explorer(doc_id)
        st.markdown("#### Chunk browser")
        chunk_browser(doc_id)
    with eval_tab:
        _evaluation_tab(doc_id)
