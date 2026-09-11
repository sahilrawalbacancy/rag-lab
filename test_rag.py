"""
Runnable checks for the RAG pipeline.   Usage:

    .venv/bin/python test_rag.py           # offline checks only (fast, no API calls)
    .venv/bin/python test_rag.py --live    # also hits the Gemini API

No pytest, no fixtures - plain asserts, so it runs anywhere with one command.
The offline half is the half that matters: it pins the guardrail and the
parsing edge cases, which is where a silent regression would do real damage.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.documents import Document

import rag

SAMPLES = Path(__file__).parent / "samples"
_checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _checks
    _checks += 1
    if not condition:
        raise AssertionError(f"{label}" + (f" -- {detail}" if detail else ""))
    print(f"  ok  {label}")


class FakeLLM:
    """A model whose reply we control, so the guardrail can be tested offline."""

    def __init__(self, payload: str):
        self.payload = payload

    def invoke(self, _prompt):
        class Reply:
            content = self.payload

        return Reply()


# ---------------------------------------------------------------------------


def test_parsing() -> None:
    print("\n[parsing]")
    txt = (SAMPLES / "acme_security_policy.txt").read_bytes()
    pages = rag.parse_document(txt, "policy.txt")
    check("txt parses to one unpaginated page", len(pages) == 1 and pages[0][0] is None)

    pdf_path = SAMPLES / "acme_pentest_assessment.pdf"
    pages = rag.parse_document(pdf_path.read_bytes(), "pentest.pdf")
    check("pdf yields >1 page", len(pages) > 1, f"got {len(pages)}")
    check("pdf pages are 1-indexed", [p for p, _ in pages] == list(range(1, len(pages) + 1)))
    # Page attribution is what makes a "p.2" citation trustworthy: content
    # from page 2 must not be reported as page 1.
    p1 = next(t for p, t in pages if p == 1)
    p2 = next(t for p, t in pages if p == 2)
    check("page 1 holds the report title", "INDEPENDENT SECURITY ASSESSMENT" in p1)
    check("page 2 content is attributed to page 2",
          "METHODOLOGY" in p2 and "METHODOLOGY" not in p1)

    for data, name, exc in [
        (b"hello", "notes.xlsx", rag.UnsupportedFile),
        (b"   \n\t ", "blank.txt", rag.NoExtractableText),
        (b"%PDF-1.4 truncated", "corrupt.pdf", rag.UnsupportedFile),
        (b"", "empty.pdf", rag.UnsupportedFile),
    ]:
        try:
            rag.parse_document(data, name)
            raise AssertionError(f"{name} should have raised {exc.__name__}")
        except exc:
            pass
    check("bad inputs raise our own exception types, never a library's", True)

    messy = "Section  1\xa0\xa0 title\n   \n\n\n\n  indented   text  \n"
    clean = rag._normalise_whitespace(messy)
    check("whitespace collapses", "  " not in clean and "\n\n\n" not in clean, repr(clean))
    check("paragraph breaks survive", "\n\n" in rag._normalise_whitespace("a\n\n\n\n\nb"))


def test_chunking() -> None:
    print("\n[chunking]")
    pages = [(1, "A" * 2400), (2, "B" * 900)]
    docs = rag.chunk_pages(pages, "doc.pdf", chunk_size=1000, chunk_overlap=150)
    check("produces multiple chunks", len(docs) > 3, f"got {len(docs)}")
    check("no chunk exceeds chunk_size", all(len(d.page_content) <= 1000 for d in docs))
    # A chunk spanning two pages could not be cited unambiguously.
    check("no chunk mixes content from two pages",
          all(set(d.page_content) <= {"A"} or set(d.page_content) <= {"B"} for d in docs))
    check("every chunk carries source + page", all(
        d.metadata.get("source") == "doc.pdf" and "page" in d.metadata for d in docs))
    check("unpaginated pages record page 0",
          rag.chunk_pages([(None, "x" * 50)], "n.txt")[0].metadata["page"] == 0)


def test_question_extraction() -> None:
    print("\n[question extraction]")
    text = (SAMPLES / "vendor_security_questionnaire.txt").read_text()
    qs = rag.extract_questions_heuristic(text)
    check("all 12 questions extracted", len(qs) == 12, f"got {len(qs)}")
    # Regression guard: these have the shape "Is X? State Y." and were dropped
    # by an endswith('?') test. A dropped question silently hides a gap.
    check("mid-line '?' questions are kept",
          any("State the algorithm" in q for q in qs)
          and any("coverage limit" in q for q in qs))
    check("numbering stripped from question text", not qs[0].startswith("1."))
    check("no duplicates", len(qs) == len(set(qs)))
    for name, preset in rag.PRESETS.items():
        check(f"preset non-empty: {name[:34]}", len(preset) >= 8)


def test_guardrail() -> None:
    """
    The core safety property: a claim of coverage survives only if a verbatim
    supporting quote is found in the retrieved text.
    """
    print("\n[guardrail]")
    hits = [(Document(page_content="Logs are retained for 12 months.",
                      metadata={"source": "policy.txt", "page": 0}), 0.4)]
    q = "How long are logs kept?"

    cases = [
        ("fabricated quote", '{"verdict":"COVERED","answer":"7 years.","quotes":["Logs are retained for 7 years."]}', rag.NOT_FOUND),
        ("paraphrased quote", '{"verdict":"COVERED","answer":"12 months.","quotes":["Log retention is twelve months"]}', rag.NOT_FOUND),
        ("no quote supplied", '{"verdict":"COVERED","answer":"Yes.","quotes":[]}', rag.NOT_FOUND),
        ("reply is not JSON", 'Probably yes, I think.', rag.NOT_FOUND),
        ("bogus verdict string", '{"verdict":"DEFINITELY","answer":"x","quotes":[]}', rag.NOT_FOUND),
        ("genuine quote", '{"verdict":"COVERED","answer":"12 months.","quotes":["Logs are retained for 12 months."]}', rag.COVERED),
        ("genuine quote, whitespace differs", '{"verdict":"COVERED","answer":"12 months.","quotes":["Logs   are retained\\nfor 12 months."]}', rag.COVERED),
        ("json wrapped in a code fence", '```json\n{"verdict":"COVERED","answer":"12 months.","quotes":["Logs are retained for 12 months."]}\n```', rag.COVERED),
    ]
    for label, payload, expected in cases:
        f = rag.answer_question(FakeLLM(payload), q, hits)
        check(f"{label} -> {expected}", f.verdict == expected,
              f"got {f.verdict} ({f.note})")

    f = rag.answer_question(FakeLLM(cases[0][1]), q, hits)
    check("downgrade drops the unverified answer text", f.answer == "")
    check("downgrade explains itself", "Downgraded" in f.note)

    f = rag.answer_question(FakeLLM('{"verdict":"COVERED","answer":"a","quotes":["Logs are retained for 12 months."]}'), q, hits)
    check("verified citation keeps source + distance",
          f.citations[0].source == "policy.txt" and f.citations[0].distance == 0.4)

    f = rag.answer_question(FakeLLM('{"verdict":"COVERED","answer":"x","quotes":["short"]}'), q, hits)
    check("quotes too short to be evidence are ignored", f.verdict == rag.NOT_FOUND)

    check("empty retrieval never reports coverage",
          rag.answer_question(FakeLLM('{"verdict":"COVERED","answer":"x","quotes":["y"]}'), q, []).verdict
          == rag.NOT_FOUND)


def test_helpers() -> None:
    print("\n[helpers]")
    class R:
        content = [{"type": "text", "text": "READY"}, {"type": "other", "x": 1}]
    check("get_text unwraps Gemini content blocks", rag.get_text(R()) == "READY")

    class S:
        content = "plain"
    check("get_text passes strings through", rag.get_text(S()) == "plain")
    check("_extract_json survives a preamble",
          rag._extract_json('Sure!\n```json\n{"a":1}\n```')["a"] == 1)
    check("icons map to every verdict",
          {rag.Finding("q", v, "").icon for v in (rag.COVERED, rag.PARTIAL, rag.NOT_FOUND)}
          == {"✅", "⚠️", "❌"})


def test_sections() -> None:
    print("\n[structure-aware chunking]")
    text = (SAMPLES / "acme_security_policy.txt").read_text()
    text = rag._normalise_whitespace(text)
    sections = rag.split_into_sections(text)
    headings = [h for h, _ in sections if h]
    check("top-level headings found", len(headings) >= 7, str(headings))
    check("clause numbers are NOT treated as headings",
          not any(h.startswith("2.1") or h.startswith("3.1") for h in headings),
          "splitting on every clause would shred the doc into single sentences")
    check("a doc with no headings returns one section",
          rag.split_into_sections("just some prose, no headings here") ==
          [(None, "just some prose, no headings here")])

    docs = rag.chunk_pages_structured([(None, text)], "policy.txt")
    check("one chunk per section", len(docs) == len(sections), f"{len(docs)} vs {len(sections)}")
    check("section recorded in metadata", any(d.metadata["section"] for d in docs))
    logging_chunk = next(d for d in docs if "12 months" in d.page_content)
    check("log-retention chunk is the LOGGING section, not the password one",
          "LOGGING" in logging_chunk.metadata["section"],
          logging_chunk.metadata["section"])

    # An oversized section must still carry its heading into every fragment,
    # or a mid-section chunk retrieves with no idea what it is about.
    big = [(1, "5. LOGGING AND MONITORING\n" + ("word " * 800))]
    frags = rag.chunk_pages_structured(big, "big.pdf", chunk_size=500, chunk_overlap=50)
    check("oversized section splits into several chunks", len(frags) > 2)
    check("every fragment keeps its heading",
          all(f.page_content.startswith("5. LOGGING") for f in frags))


def test_retry() -> None:
    print("\n[rate-limit retry]")
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 Resource has been exhausted (quota)")
        return "ok"

    check("retries through a 429 and succeeds",
          rag._with_retry(flaky, attempts=4, base_delay=0.01) == "ok" and calls["n"] == 3)

    def broken():
        raise ValueError("malformed request - not transient")

    try:
        rag._with_retry(broken, attempts=3, base_delay=0.01)
        raise AssertionError("a non-transient error must not be retried")
    except ValueError:
        pass
    check("a non-rate-limit error is raised immediately, not retried", True)


def test_export() -> None:
    print("\n[export + summary]")
    findings = [
        rag.Finding("Q1", rag.COVERED, "Yes, AES-256.",
                    [rag.Citation("policy.pdf", 4, "encrypted with AES-256", 0.31)]),
        rag.Finding("Q2", rag.NOT_FOUND, "", [], "nothing found"),
        rag.Finding("Q3, with a comma", rag.PARTIAL, 'He said "maybe"', []),
    ]
    counts = rag.summarise(findings)
    check("summary counts each verdict",
          counts == {rag.COVERED: 1, rag.PARTIAL: 1, rag.NOT_FOUND: 1, "total": 3}, str(counts))

    import csv, io
    rows = list(csv.reader(io.StringIO(rag.findings_to_csv(findings))))
    check("csv has a header plus one row per finding", len(rows) == 4, str(len(rows)))
    # Commas and quotes in questions/answers must not break the CSV a
    # respondent pastes straight back into a client's spreadsheet.
    check("commas and quotes survive csv quoting",
          rows[3][1] == "Q3, with a comma" and rows[3][3] == 'He said "maybe"', str(rows[3]))
    check("evidence quote is exported", "AES-256" in rows[1][5])


def test_extraction_fallback() -> None:
    print("\n[extraction fallback]")
    text = (SAMPLES / "vendor_security_questionnaire.txt").read_text()
    check("no llm falls back to the heuristic",
          len(rag.extract_questions(text, llm=None)) == 12)
    check("an unusable llm reply falls back rather than returning nothing",
          len(rag.extract_questions(text, llm=FakeLLM("I cannot help with that."))) == 12)
    check("an empty llm list falls back too",
          len(rag.extract_questions(text, llm=FakeLLM("[]"))) == 12)
    good = FakeLLM('["Only question one?", "Only question two?"]')
    check("a good llm reply is used", rag.extract_questions(text, llm=good) ==
          ["Only question one?", "Only question two?"])


def test_chat() -> None:
    print("\n[chat mode]")
    answer, cites = rag.chat_answer(FakeLLM("unused"), "anything?", [])
    check("empty retrieval says so instead of inventing", "could not find" in answer.lower())
    check("empty retrieval cites nothing", cites == [])

    hits = [(Document(page_content="TLS 1.3 is enforced.",
                      metadata={"source": "arch.pdf", "page": 2}), 0.2)]
    answer, cites = rag.chat_answer(
        FakeLLM("TLS 1.3 [arch.pdf, p.2]"), "which tls?", hits,
        history=[("hi", "hello")])
    check("answer returned with citations", "TLS 1.3" in answer and len(cites) == 1)
    check("citation label includes the page", cites[0].label() == "arch.pdf, p.2")


def test_prune_offline() -> None:
    print("\n[collection pruning - offline]")
    check("pruning a directory that does not exist is a no-op",
          rag.prune_collections("/tmp/definitely_not_a_chroma_dir_xyz") == 0)


def test_live() -> None:
    """Hits the API. Verifies the two things only a real call can show."""
    print("\n[live api]")
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        print("  SKIP - no GOOGLE_API_KEY"); return

    from langchain_google_genai import ChatGoogleGenerativeAI
    chat_model, embed_model = rag.resolve_models(key)
    check(f"models resolved ({chat_model} / {embed_model})", bool(chat_model and embed_model))

    docs = []
    for name in ("acme_security_policy.txt", "acme_architecture.txt"):
        docs += rag.chunk_pages(rag.parse_document((SAMPLES / name).read_bytes(), name), name)
    store = rag.build_index(docs, embed_model, persist_directory="/tmp/chroma_test", api_key=key)
    check("all chunks indexed", store._collection.count() == len(docs))

    hits = rag.retrieve(store, "How long are security audit logs retained?",
                        embed_model, k=5, api_key=key)
    check("retrieval returns k hits", len(hits) == 5)
    # Chroma's method says "relevance_scores" but returns DISTANCES. If this
    # ever flips, every citation ranking in the UI silently inverts.
    dists = [d for _, d in hits]
    check("distances ascend (lower == closer)", dists == sorted(dists), str(dists))
    check("nearest chunk actually answers the question",
          "12 months" in hits[0][0].page_content)

    llm = ChatGoogleGenerativeAI(model=chat_model, temperature=0, google_api_key=key)
    f = rag.answer_question(llm, "What is the capital of France?", hits)
    check("world knowledge is refused", f.verdict == rag.NOT_FOUND, f.answer)

    f = rag.answer_question(llm, "How long are security audit logs retained?", hits)
    check("a real question is answered with a citation",
          f.verdict == rag.COVERED and f.citations, f"{f.verdict} {f.note}")
    check("citation quote is genuinely in the chunk",
          rag._normalise_for_match(f.citations[0].quote)
          in rag._normalise_for_match(" ".join(d.page_content for d, _ in hits)))

# Reload recovery depends on reopening a collection without re-embedding.
    summary = rag.collection_summary(store)
    check("collection_summary rebuilds the per-document chunk counts",
          dict(summary) and sum(n for _, n in summary) == len(docs), str(summary))

    reopened = rag.open_index(embed_model, store._collection.name,
                              persist_directory="/tmp/chroma_test", api_key=key)
    check("an existing collection reopens with its chunks intact",
          reopened._collection.count() == len(docs))
    hits2 = rag.retrieve(reopened, "How long are security audit logs retained?",
                         embed_model, k=3, api_key=key)
    check("the reopened collection still answers", "12 months" in hits2[0][0].page_content)

    # A .txt has no pages, so its label carries none. The model used to append
    # a plausible "p.1" anyway - a fabricated citation in an app whose whole
    # promise is trustworthy citations.
    import re
    answer, _ = rag.chat_answer(llm, "What TLS version is enforced?", hits)
    check("no page number is invented for a paginationless source",
          not re.findall(r"\[[^\]]*\.txt,\s*p\.\d+\]", answer), answer[:160])


if __name__ == "__main__":
    live = "--live" in sys.argv
    test_parsing()
    test_chunking()
    test_sections()
    test_question_extraction()
    test_extraction_fallback()
    test_guardrail()
    test_retry()
    test_export()
    test_chat()
    test_prune_offline()
    test_helpers()
    if live:
        test_live()
    else:
        print("\n[live api]\n  SKIP - pass --live to include API checks")
    print(f"\nall {_checks} checks passed")
