"""
Compliance Copilot - Streamlit UI.

All pipeline logic lives in rag.py; this file is presentation and state only.

Streamlit re-runs this entire script on every interaction, so anything
expensive (the embedded index, audit results, chat history) is kept in
st.session_state and never rebuilt implicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import rag

# find_dotenv() inspects caller stack frames and breaks under Streamlit's
# script runner, so the path is always explicit.
load_dotenv(Path(__file__).parent / ".env")

PERSIST_DIR = str(Path(__file__).parent / ".chroma")


def _query_param(name: str) -> str | None:
    """
    Read a query parameter as a single string.

    A URL can legitimately carry a repeated parameter (?s=a&s=b), and some
    Streamlit surfaces hand back a list for that case. Calling a string method
    on a list would raise, so the first value is taken and anything unusable
    is treated as absent.
    """
    value = st.query_params.get(name)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return value if isinstance(value, str) else None


def session_collection() -> str:
    """
    A Chroma collection name unique to this browser session, kept in the URL.

    Two requirements pull in opposite directions:

    1. Sessions must not share a collection. They did once, and a second user
       clicking "Build index" destroyed the first user's data and could serve
       them someone else's confidential documents.
    2. A browser reload should not throw the work away. session_state is wiped
       on reload, so a name held only there is lost - and the collection it
       named is orphaned on disk, holding the user's document text forever.

    Putting the name in a query parameter satisfies both: it survives reload,
    it is unique per session, and a random uuid is not guessable by anyone who
    was not given the URL.
    """
    existing = _query_param("s")
    if existing and existing.startswith("kb_"):
        st.session_state.collection = existing
        return existing

    if "collection" not in st.session_state:
        import uuid

        st.session_state.collection = f"kb_{uuid.uuid4().hex}"
    st.query_params["s"] = st.session_state.collection
    return st.session_state.collection


def recover_session(embed_model: str, key: str) -> None:
    """
    After a reload, reopen this session's collection instead of re-embedding.

    The vectors are already on disk. Reopening costs one call; rebuilding
    would re-embed every chunk and spend API quota to produce the same
    result.
    """
    if st.session_state.store is not None:
        return
    name = _query_param("s")
    if not name or not name.startswith("kb_"):
        return
    try:
        store = rag.open_index(embed_model, name, persist_directory=PERSIST_DIR, api_key=key)
        if store._collection.count() == 0:
            return
        st.session_state.store = store
        st.session_state.indexed = rag.collection_summary(store)
    except Exception:
        # A collection that has been pruned or never existed is not an error;
        # the user simply starts fresh.
        return

st.set_page_config(page_title="Compliance Copilot", page_icon="📋", layout="wide")


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

# Factories, not literals: a shared mutable default would be aliased into
# session_state, so appending a chat turn would mutate the defaults themselves
# and leak across a reset.
DEFAULTS = {
    "store": lambda: None,
    "indexed": list,       # [(filename, n_chunks)]
    "findings": lambda: None,
    "questions": lambda: None,
    "chat": list,          # [(user, assistant)]
    "models": lambda: None,  # (chat_model, embed_model)
}


def reset_state() -> None:
    for name, factory in DEFAULTS.items():
        st.session_state[name] = factory()


for _name, _factory in DEFAULTS.items():
    if _name not in st.session_state:
        st.session_state[_name] = _factory()


def api_key() -> str:
    """
    Read the key from Streamlit secrets first, then the environment.

    Locally the key lives in .env; on Streamlit Community Cloud there is no
    .env and the key is injected via st.secrets. Supporting both is what makes
    the same code deployable without edits. Accessing st.secrets raises when no
    secrets file exists at all, so it is guarded.
    """
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass
    return (os.environ.get("GOOGLE_API_KEY") or "").strip()


@st.cache_resource(show_spinner=False)
def get_llm(model: str, key: str, temperature: float = 0.0):
    """Cached so a rerun reuses one client instead of building a new one."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model, temperature=temperature, google_api_key=key
    )


@st.cache_data(show_spinner=False, ttl=3600)
def cached_models(key: str) -> tuple[str, str]:
    return rag.resolve_models(key)


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Setup")

    key = api_key()
    if not key:
        st.error(
            "No `GOOGLE_API_KEY` found.\n\n"
            "**Locally** — add it to `.env` in the project root and rerun:\n"
            "```\nGOOGLE_API_KEY=your-key\n```\n"
            "**On Streamlit Cloud** — add it under *App settings -> Secrets*.\n\n"
            "Free key: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)"
        )
        st.stop()

    if st.session_state.models is None:
        with st.spinner("Checking which models this key can reach..."):
            try:
                st.session_state.models = cached_models(key)
            except Exception as exc:
                st.error(f"Could not reach the Gemini API:\n\n`{exc}`")
                st.stop()

    chat_model, embed_model = st.session_state.models

    # Reopen this session's collection after a reload, and sweep away
    # collections whose sessions are long gone.
    recover_session(embed_model, key)
    # NB: st.session_state supports .get() on the live runtime but not under
    # Streamlit's own AppTest harness, so membership tests are used instead -
    # otherwise the whole UI test suite dies on an AttributeError.
    if "pruned" not in st.session_state:
        keep = st.session_state.collection if "collection" in st.session_state else None
        rag.prune_collections(PERSIST_DIR, keep=keep)
        st.session_state.pruned = True

    st.success("Gemini connected")
    st.caption(f"**Chat** `{chat_model}`  \n**Embeddings** `{embed_model}`")

    st.divider()
    with st.expander("Advanced settings"):
        top_k = st.slider(
            "Passages per question", 3, 10, rag.TOP_K,
            help="How much of your document the assistant reads before "
                 "answering each question.",
        )
        workers = st.slider(
            "Questions answered at once", 1, 8, rag.MAX_WORKERS,
            help="Higher is faster. Lower this if answers start failing on a "
                 "free API key.",
        )
        structured = st.toggle(
            "Split documents by section", value=True,
            help="Keeps each numbered section together, so an answer cites "
                 "one clean section rather than a fragment.",
        )

    st.divider()
    if st.session_state.indexed:
        st.subheader("Knowledge base")
        for name, n in st.session_state.indexed:
            st.caption(f"{name} — {n} passages")
        st.caption("Refreshing the page keeps these documents loaded.")
        if st.button("Clear knowledge base", use_container_width=True):
            # Drop the collection too, not just the handle, so the uploaded
            # content does not linger on the server's disk after the user
            # has explicitly cleared it.
            store = st.session_state.store
            if store is not None:
                try:
                    store.delete_collection()
                except Exception:
                    pass
            reset_state()
            st.session_state.pop("collection", None)
            st.query_params.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------

st.title("📋 Compliance Copilot")
st.caption(
    "Upload your documents, point a questionnaire at them, and get a drafted "
    "answer for every question - with a verified citation, or an explicit gap."
)

# ---------------------------------------------------------------------------
# step 1 - knowledge base
# ---------------------------------------------------------------------------

st.header("1 · Your documents")
st.caption(
    "The source of truth the answers must come from - policies, architecture "
    "docs, certifications, contracts."
)

uploads = st.file_uploader(
    "Upload one or more documents",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
    key="kb_files",
)

col_a, col_b = st.columns([1, 3])
with col_a:
    build_clicked = st.button(
        "Add documents", type="primary", disabled=not uploads, use_container_width=True
    )
with col_b:
    use_samples = st.button(
        "Try it with sample documents", use_container_width=True,
        help="A fictional company's security policy, architecture overview "
             "and penetration-test report.",
    )

if build_clicked or use_samples:
    chunker = rag.chunk_pages_structured if structured else rag.chunk_pages

    if use_samples:
        # An explicit list, not a glob: samples/ also holds acme_handbook.pdf,
        # which is these two documents concatenated. Indexing it alongside them
        # would put duplicate chunks in the store and crowd the top-k.
        sample_dir = Path(__file__).parent / "samples"
        sample_names = [
            "acme_security_policy.txt",
            "acme_architecture.txt",
            "acme_pentest_assessment.pdf",   # a real PDF, for page-level citations
        ]
        sources = [
            (name, (sample_dir / name).read_bytes())
            for name in sample_names
            if (sample_dir / name).exists()
        ]
    else:
        sources = [(f.name, f.getvalue()) for f in uploads]

    docs, indexed, failures = [], [], []
    for name, data in sources:
        try:
            pages = rag.parse_document(data, name)
            chunks = chunker(pages, name)
            docs += chunks
            indexed.append((name, len(chunks)))
        except (rag.UnsupportedFile, rag.NoExtractableText) as exc:
            failures.append(str(exc))

    for message in failures:
        st.error(message)

    if not docs:
        st.error("Nothing could be indexed.")
    else:
        with st.spinner(f"Embedding {len(docs)} chunks..."):
            try:
                st.session_state.store = rag.build_index(
                    docs,
                    embed_model,
                    persist_directory=PERSIST_DIR,
                    collection_name=session_collection(),
                    api_key=key,
                )
                st.session_state.indexed = indexed
                st.session_state.findings = None
            except Exception as exc:
                st.error(f"Indexing failed: `{exc}`")
        if st.session_state.store is not None:
            total = sum(n for _, n in indexed)
            st.success(f"Ready — {len(indexed)} document(s), {total} passages indexed.")

if st.session_state.store is None:
    st.info("Add your documents above to get started.")
    st.stop()


# ---------------------------------------------------------------------------
# step 2 - the checklist
# ---------------------------------------------------------------------------

st.header("2 · The questionnaire")

mode = st.radio(
    "Where do the questions come from?",
    ["Use a preset", "Upload a questionnaire"],
    horizontal=True,
    label_visibility="collapsed",
)

questions: list[str] = []
if mode == "Use a preset":
    preset = st.selectbox("Preset checklist", list(rag.PRESETS))
    questions = rag.PRESETS[preset]
    with st.expander(f"{len(questions)} questions"):
        for i, q in enumerate(questions, 1):
            st.write(f"{i}. {q}")
else:
    qfile = st.file_uploader(
        "Upload the questionnaire", type=["pdf", "docx", "txt", "md"], key="q_file"
    )
    if qfile is not None:
        try:
            pages = rag.parse_document(qfile.getvalue(), qfile.name)
            text = "\n".join(t for _, t in pages)
            with st.spinner("Reading the questions out of the file..."):
                questions = rag.extract_questions(
                    text, llm=get_llm(chat_model, key)
                )
            st.success(f"Found {len(questions)} questions in `{qfile.name}`.")
            st.caption("Edit below if the extraction missed or merged anything.")
            edited = st.text_area(
                "Questions (one per line)",
                value="\n".join(questions),
                height=200,
            )
            questions = [q.strip() for q in edited.splitlines() if q.strip()]
        except (rag.UnsupportedFile, rag.NoExtractableText) as exc:
            st.error(str(exc))

# ---------------------------------------------------------------------------
# step 3 - run
# ---------------------------------------------------------------------------

st.header("3 · Run the audit")

if st.button(
    f"Audit {len(questions)} questions",
    type="primary",
    disabled=not questions,
    use_container_width=True,
):
    llm = get_llm(chat_model, key)
    progress = st.progress(0.0, text="Starting...")
    done = {"n": 0}

    def on_result(_index, finding):
        done["n"] += 1
        progress.progress(
            done["n"] / len(questions),
            text=f"{done['n']}/{len(questions)} answered - {finding.question[:60]}",
        )

    try:
        st.session_state.findings = rag.run_audit(
            questions,
            st.session_state.store,
            llm,
            embed_model,
            api_key=key,
            k=top_k,
            max_workers=workers,
            on_result=on_result,
        )
        st.session_state.questions = questions
    except Exception as exc:
        st.error(f"Audit failed: `{exc}`")
    progress.empty()

findings = st.session_state.findings

if findings:
    counts = rag.summarise(findings)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions", counts["total"])
    c2.metric("✅ Covered", counts[rag.COVERED])
    c3.metric("⚠️ Partial", counts[rag.PARTIAL])
    c4.metric("❌ Gaps", counts[rag.NOT_FOUND])

    answered = counts[rag.COVERED] + counts[rag.PARTIAL]
    st.progress(answered / counts["total"] if counts["total"] else 0.0,
                text=f"{answered} of {counts['total']} answerable from your documents")

    gaps = [f for f in findings if f.verdict == rag.NOT_FOUND]
    if gaps:
        st.warning(
            f"**{len(gaps)} question(s) need a human.** Nothing in the uploaded "
            "documents answers them:\n\n"
            + "\n".join(f"- {f.question}" for f in gaps)
        )

    st.download_button(
        "⬇ Download answers as CSV",
        data=rag.findings_to_csv(findings),
        file_name="questionnaire_answers.csv",
        mime="text/csv",
    )

    st.subheader("Answers")
    only_gaps = st.checkbox("Show only gaps", value=False)

    for i, f in enumerate(findings, 1):
        if only_gaps and f.verdict != rag.NOT_FOUND:
            continue
        sources = ", ".join(dict.fromkeys(c.label() for c in f.citations))
        with st.expander(
            f"{f.icon} **{i}. {f.question}**" + (f" — {sources}" if sources else ""),
            expanded=False,
        ):
            if f.answer:
                st.markdown(f.answer)
            else:
                st.markdown("_Your documents don't answer this — it needs a human._")

            if f.citations:
                st.caption("**Evidence** — quoted from your document, and checked to be there")
                for c in f.citations:
                    st.markdown(f"> {c.quote}")
                    st.caption(f"↳ {c.label()}")
            if f.note:
                st.caption(f"ℹ️ {f.note}")

# ---------------------------------------------------------------------------
# step 4 - chat
# ---------------------------------------------------------------------------

st.header("4 · Ask a follow-up")
st.caption("Same documents, same grounding — free-form questions.")

for user_msg, bot_msg in st.session_state.chat:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)

prompt = st.chat_input("Ask anything about your documents…")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                hits = rag.retrieve(
                    st.session_state.store, prompt, embed_model, k=top_k, api_key=key
                )
                answer, citations = rag.chat_answer(
                    get_llm(chat_model, key), prompt, hits, history=st.session_state.chat
                )
            except Exception as exc:
                answer, citations = f"Something went wrong: `{exc}`", []
        st.markdown(answer)
        if citations:
            with st.expander("Where this came from"):
                for c in citations:
                    st.caption(f"↳ {c.label()}")
                    st.markdown(f"> {c.quote}")
    st.session_state.chat.append((prompt, answer))
