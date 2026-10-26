"""
Offline checks for the three assignment pipelines - no API calls.

    .venv/bin/python tests/test_assignments.py

Covers each document's structure handling against the real PDFs, the
retrieval stack (local models, an in-memory index, a fake LLM), the scoring
rules, and the evaluation set itself: every expected fact must really be on
the page it is expected on, or a "failure" would be the test's own mistake.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402


def doc(text: str, meta: dict) -> Document:
    return Document(page_content=text, metadata=meta)

from rag import core, evaluation, retrieval, structured, vision, visual  # noqa: E402
from rag.assignments import CHROMA_DIR, EVAL_SET, VISION_DIR, registry  # noqa: E402

_checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _checks
    _checks += 1
    if not condition:
        raise AssertionError(f"{label}" + (f" -- {detail}" if detail else ""))
    print(f"  ok  {label}")


class FakeLLM:
    def __init__(self, payload: str):
        self.payload, self.prompts = payload, []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        class Reply:
            content = self.payload

        return Reply()


# ---------------------------------------------------------------------------


def test_cleaning() -> None:
    print("\n[cleaning]")
    header = "U.S. GOVERNMENT PUBLISHING OFFICE | 2018 ANNUAL REPORT"
    # Furniture is matched with digits ignored, so the body lines here differ
    # in words, as real body text does.
    pages = [(i, f"{i}{header}\nBody text about topic {chr(64 + i)}.\n2023 2024 2025e 2026f\n{i}")
             for i in range(1, 11)]
    cleaned = dict(core.strip_page_furniture(pages))
    check("a running header is stripped, even with the page number glued on",
          all(header not in t for t in cleaned.values()), cleaned[3])
    check("bare page numbers are stripped", all(not t.endswith(f"\n{p}") for p, t in cleaned.items()))
    check("body text survives", all(f"topic {chr(64 + p)}." in t for p, t in cleaned.items()))
    check("a table's year header is content, not furniture",
          all("2023 2024 2025e 2026f" in t for t in cleaned.values()))
    check("drop caps are repaired in both variants",
          structured._DROP_CAP.sub("T", "7T_he debt. 7'_he box.") == "The debt. The box.")


def test_bm25() -> None:
    print("\n[keyword search]")
    bm = retrieval.BM25(["Growth in Iraq is forecast to contract.",
                         "Growth in East Asia is projected to moderate.",
                         "Passports produced 15,901,600"])
    scores = bm.scores("Iraq growth forecast")
    check("BM25 ranks the exact-term document first", scores.index(max(scores)) == 0, str(scores))
    check("numbers with separators are one searchable term", bm.scores("15901600")[2] > 0)
    check("stop words alone match nothing", max(bm.scores("what is the")) == 0)


def _tiny_index():
    import chromadb

    docs = retrieval.finalise([
        doc("South Asia growth is expected to soften to 6.3 percent in 2026.",
                 {"source": "t", "page": 96, "region": "South Asia", "section": "South Asia"}),
        doc("East Asia and Pacific growth is projected to moderate to 4.2 percent in 2026.",
                 {"source": "t", "page": 81, "region": "East Asia and Pacific", "section": "East Asia and Pacific"}),
        doc("iMac uses 58 percent less energy than the requirement for ENERGY STAR.[^18]",
                 {"source": "t", "page": 7, "section": "Use", "footnotes": "18"}),
        doc("Footnote 18: Energy values are based on ENERGY STAR requirements; iMac is tested "
                 "with the 143W Power Adapter.",
                 {"source": "t", "page": 12, "section": "Endnotes", "footnote": "18", "kind": "footnote"}),
    ], "test")
    client = chromadb.EphemeralClient()
    retrieval.build_collection(client, "tiny", docs)
    return retrieval.Index.load(client, "tiny")


def test_retrieval() -> None:
    print("\n[retrieval - local models, in-memory index]")
    index = _tiny_index()
    s = retrieval.Settings(k=1)
    top = retrieval.search(index, "What is the growth outlook for South Asia?", s)
    check("hybrid + rerank finds the South Asia passage", top[0].doc.metadata["page"] == 96,
          top[0].doc.page_content)
    routed = retrieval.search(index, "growth outlook", s, where={"region": "East Asia and Pacific"})
    check("a route filter keeps the search inside one region",
          routed and routed[0].doc.metadata["region"] == "East Asia and Pacific")
    empty = retrieval.search(index, "growth", s, where={"region": "Atlantis"})
    check("a route that matches nothing falls back to the whole index", bool(empty))

    hit = retrieval.search(index, "How much less energy than ENERGY STAR?", retrieval.Settings(k=1))
    linked = retrieval.expand(index, hit, retrieval.Settings(k=1, footnotes=True))
    check("a retrieved claim brings its footnote with it",
          [h.why for h in linked] == ["retrieved", "footnote 18"], str([h.why for h in linked]))
    plain = retrieval.expand(index, hit, retrieval.Settings(k=1))
    check("without footnote links the claim arrives alone", len(plain) == 1)

    check("consecutive chunks are joined without repeating their overlap",
          retrieval._join("alpha beta gamma delta epsilon zeta", "delta epsilon zeta eta theta")
          == "alpha beta gamma delta epsilon zeta eta theta")
    run = [retrieval.Hit(doc(t, {"order": i, "page": 5, "section": ""}), why=w) for i, (t, w) in enumerate(
        [("GPO staff visited 166 depository", "neighbour"),
         ("visited 166 depository libraries in 25 states.", "retrieved")])]
    merged = retrieval._merge_runs(run)
    check("a hit and its neighbour become one continuous passage, so a sentence across "
          "the boundary can be quoted", len(merged) == 1
          and "staff visited 166 depository libraries in 25 states" in merged[0].doc.page_content
          and merged[0].why == "retrieved + neighbours", str([(m.doc.page_content, m.why) for m in merged]))

    llm = FakeLLM('{"verdict":"COVERED","answer":"58 percent less, tested with the 143W adapter.",'
                  '"quotes":["58 percent less energy than the requirement for ENERGY STAR",'
                  '"tested with the 143W Power Adapter"]}')
    answer = retrieval.ask(index, "How much less energy?", llm,
                           retrieval.Settings(k=1, footnotes=True))
    check("ask() verifies quotes from both the claim and its footnote",
          answer.finding.verdict == core.COVERED and len(answer.finding.citations) == 2,
          answer.finding.note)
    check("the prompt labels passages with page and section", "p.12 · Endnotes" in llm.prompts[0])


def test_gep_structure() -> None:
    print("\n[World Bank - structure, against the real PDF]")
    from pypdf import PdfReader

    raw = dict(core.parse_document(structured.PDF.read_bytes(), structured.PDF.name))
    pages = structured.section_map(PdfReader(str(structured.PDF)), raw)
    span = lambda key, value: [p for p, m in pages.items() if m[key] == value]
    check("South Asia is snapped from its bookmark (p.95) to its heading (p.96)",
          min(span("region", "South Asia")) == 96 and max(span("region", "South Asia")) == 98)
    check("Box 1.1 ends where its '(continued)' headers stop: p.30-35",
          (min(span("box", "1.1")), max(span("box", "1.1"))) == (30, 35))
    check("Box 1.2 is found even when its header sits mid-page: p.44-50",
          (min(span("box", "1.2")), max(span("box", "1.2"))) == (44, 50))
    check("the executive summary is p.19-20, its abbreviations page is separate",
          {19, 20} <= set(span("chapter", "Executive Summary")) and pages[21]["chapter"] == "Abbreviations")
    check("Chapter 3 is p.107-142", span("chapter_no", "3") == list(range(107, 143)))
    check("printed page numbers are read from the running header",
          structured.printed_page(raw[96]) == "74" and structured.printed_page(raw[19]) == "xvii")
    vision_pages = structured.vision_pages(raw, pages)
    check("table pages and the annex go to vision", 95 in vision_pages and 187 in vision_pages
          and 96 not in vision_pages, str(vision_pages))

    routes = {
        "What is the growth outlook for South Asia in this report?": {"region": "South Asia"},
        "What does Chapter 3 say about sovereign debt?": {"chapter_no": "3"},
        "According to the executive summary, what are the main risks?": {"chapter": "Executive Summary"},
        "According to Box 1.1, how could AI lift growth?": {"box": "1.1"},
        "What is the growth forecast for East Asia and Pacific in 2026?": {"region": "East Asia and Pacific"},
        "How did the conflict in the Middle East affect energy prices?": None,
        "According to the statistical appendix, what was euro area growth?": {"chapter": "Statistical Annex"},
    }
    for q, want in routes.items():
        check(f"route: {q[:58]}", structured.route(q)[0] == want, str(structured.route(q)))


def test_tables() -> None:
    print("\n[tables]")
    table = "\n".join(["| Country | 2025e | 2026f |", "|---|---|---|"]
                      + [f"| Country {i} [^1] | {i}.1 | {i}.2 |" for i in range(60)])
    pieces = structured.split_table(table, "TABLE 2.8 Forecasts", limit=400)
    check("a long table is split into several pieces", len(pieces) > 3, str(len(pieces)))
    check("every piece repeats the title and the header row",
          all(p.startswith("TABLE 2.8 Forecasts\n| Country | 2025e | 2026f |\n|---|---|---|") for p in pieces))
    rows = [r for p in pieces for r in p.splitlines()[3:]]
    check("no row is lost or cut", len(rows) == 60 and all(r.endswith("|") for r in rows))
    labelled = structured.label_repeated_columns(
        "| | 2025e | 2026f | 2026f |\n|---|---|---|---|\n| EAP | 5.0 | 4.2 | -0.2 |", "change vs January")
    check("a repeated header column is labelled with its meaning, the first is left alone",
          labelled.splitlines()[0] == "| | 2025e | 2026f | 2026f (change vs January) |", labelled)
    blocks = structured.markdown_blocks("Intro text.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nAfter.")
    check("markdown splits into text / table / text", [k for k, _ in blocks] == ["text", "table", "text"])


def test_footnotes() -> None:
    print("\n[footnotes - iMac]")
    body, notes = visual.split_footnotes(
        "## Use\niMac uses 58 percent less energy.[^18]\n\n[^17]: Wood fibers include\nbamboo.\n"
        "[^18]: Tested with the 143W Power Adapter.")
    check("footnote definitions are parsed, continuation lines included",
          notes == {"17": "Wood fibers include bamboo.", "18": "Tested with the 143W Power Adapter."}, str(notes))
    check("definitions are removed from the body", "[^17]:" not in body and "[^18]" in body)
    blocks = visual.heading_blocks("# A\ntext a\n## Chart: B\n- x: 1\n### C\n| t |")
    check("a new block starts at every heading", [h for h, _ in blocks] == ["A", "Chart: B", "C"])
    check("vision fidelity flags an invented number and ignores footnote markers",
          vision.unconfirmed_numbers("Baseline 445 kg[^7], reduced 99 kg, total 777",
                                     "445 kg Baseline emissions7 –99 kg") == ["777"])
    check("a footnote marker glued onto a year in the text layer still confirms the year",
          vision.unconfirmed_numbers("remove plastics by 2025.[^3]", "from packaging by 2025.3") == [])
    check("a number the text layer split with a space ('3. 020') still confirms, "
          "without gluing its neighbours together",
          vision.unconfirmed_numbers("| 3.020 | 1.864 | Myanmar [^3,4] |", "spread 3. 020 1.864") == []
          and vision.unconfirmed_numbers("3.0201.864", "3. 020 1.864") == ["3.0201.864"])


def test_scoring() -> None:
    print("\n[scoring]")

    def answer(text, verdict=core.COVERED, pages=(14,)):
        finding = core.Finding("q", verdict, text, [core.Citation("s", p, "quote", 0.0) for p in pages])
        return retrieval.Answer(finding, [retrieval.Hit(doc("x", {"page": p})) for p in pages])

    spec = {"expect": {"all": ["15901600"]}, "pages": [14]}
    check("thousands separators are normalised", evaluation.score(spec, answer("15,901,600 passports."))["passed"])
    check("a unicode minus matches", evaluation.score({"expect": {"all": ["-8.9"]}, "pages": [14]},
                                                      answer("contract by −8.9 percent"))["passed"])
    check("the right fact from the wrong page fails",
          not evaluation.score(spec, answer("15,901,600 passports.", pages=(15,)))["passed"])
    check("an ungrounded answer fails even with the right words",
          not evaluation.score(spec, answer("15,901,600", verdict=core.NOT_FOUND))["passed"])
    groups = {"expect": {"any": [["a"], ["b"], ["c"]], "min_any": 2}, "pages": [14]}
    check("min_any needs that many groups", evaluation.score(groups, answer("a and c"))["passed"]
          and not evaluation.score(groups, answer("only a"))["passed"])
    check("a refusal question passes only on NOT_FOUND",
          evaluation.score({"refuse": True}, answer("", verdict=core.NOT_FOUND))["passed"]
          and not evaluation.score({"refuse": True}, answer("made up"))["passed"])


def test_eval_set() -> None:
    print("\n[the evaluation set is itself correct]")
    data = json.loads(EVAL_SET.read_text())
    strategies = registry()
    for doc_id, specs in ((k, v) for k, v in data.items() if not k.startswith("_")):
        strategy = strategies[doc_id]
        text = dict(core.parse_document(strategy.PDF.read_bytes(), strategy.PDF.name))
        for md in (VISION_DIR / doc_id).glob("*.md"):
            text[int(md.stem)] = text.get(int(md.stem), "") + "\n" + md.read_text()
        check(f"{doc_id}: has the three official questions",
              sum(q["kind"] == "official" for q in specs) == 3)
        for q in specs:
            if q.get("refuse"):
                continue
            source = evaluation.normalise(" ".join(text.get(p, "") for p in q["pages"]))
            source = source.replace("aluminium", "aluminum")
            missing = [f for f in q["expect"].get("all", []) if evaluation.normalise(f) not in source]
            groups = q["expect"].get("any", [])
            met = sum(any(evaluation.normalise(a) in source for a in g) for g in groups)
            check(f"{q['id']}: every expected fact is on an expected page",
                  not missing and met >= q["expect"].get("min_any", len(groups)),
                  f"missing {missing}, groups met {met}")


def test_built_database() -> None:
    print("\n[the committed database]")
    if not CHROMA_DIR.exists():
        print("  SKIP - run ingest.py first")
        return
    client = retrieval.open_client(CHROMA_DIR)
    for doc_id in registry():
        for variant in ("naive", "advanced"):
            index = retrieval.Index.load(client, f"{doc_id}__{variant}")
            check(f"{doc_id}__{variant}: {len(index.docs)} chunks, numbered in order",
                  index.docs and [d.metadata["order"] for d in index.docs] == list(range(len(index.docs))))
    gep = retrieval.Index.load(client, "gep__advanced")
    check("every World Bank chunk carries its chapter", all(d.metadata["chapter"] for d in gep.docs))
    check("World Bank tables came through vision as table chunks",
          sum(d.metadata["kind"] == "table" and d.metadata["origin"] == "vision" for d in gep.docs) > 30)
    imac = retrieval.Index.load(client, "imac__advanced")
    check("iMac endnotes are their own linked chunks", "18" in imac.footnotes and "10" in imac.footnotes)
    check("a stray footnote definition does not replace the real endnote",
          "business-as-usual" in imac.footnotes["7"].page_content, imac.footnotes["7"].page_content)
    check("every footnote number has exactly one chunk",
          sum(d.metadata["kind"] == "footnote" for d in imac.docs) == len(imac.footnotes))
    invented = {p: n for p, n in visual.fidelity().items() if n}
    check("no iMac vision page contains a number missing from the PDF", not invented, str(invented))


def test_pages() -> None:
    """Every page renders inside the real navigation; the Ask form answers with a fake LLM."""
    print("\n[pages - rendered offline with a fake LLM]")
    if not CHROMA_DIR.exists():
        print("  SKIP - run ingest.py first")
        return
    import os

    from streamlit.testing.v1 import AppTest

    import ui

    os.environ.setdefault("GOOGLE_API_KEY", "offline-test")
    llm = ui.get_llm("gemini-3.1-flash-lite", "offline-test")
    check("the app's answer model falls back down the whole preference list",
          [f.model for f in llm.fallbacks] == [m for m in core.CHAT_PREFERENCE if m != "gemini-3.1-flash-lite"]
          and llm.runnable.model == "gemini-3.1-flash-lite")
    ui.cached_models = lambda key: ("fake-model", "fake-embed")
    ui.get_llm = lambda model, key, temperature=0.0: FakeLLM(
        '{"verdict":"COVERED","answer":"GPO staff visited 166 libraries.",'
        '"quotes":["GPO staff visited 166 depository libraries in 25 states"]}')

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=300)
    at.run()
    check("overview renders its scorecard", not at.exception and bool(at.dataframe), str(at.exception))
    for page, doc_id in (("views/gpo.py", "gpo"), ("views/world_bank.py", "gep"), ("views/imac.py", "imac")):
        at.switch_page(page).run()
        check(f"{doc_id}: page renders all three tabs", not at.exception and len(at.tabs) >= 3,
              str(at.exception))
    at.switch_page("views/world_bank.py").run()
    frames = [df.value for df in at.dataframe]
    check("the Evaluation tab shows every repeated run's score",
          any("passed in each run" in f.columns for f in frames))
    check("the Evaluation tab lists every variant, question by question",
          any("variant" in f.columns and "advanced" in list(f["variant"]) for f in frames)
          and any("question" in f.columns and len(f) == 8 for f in frames))
    at.switch_page("views/gpo.py").run()
    at.text_input(key="gpo_typed").set_value("How many depository libraries did GPO staff visit in FY 2018?")
    next(b for b in at.button if b.label == "Ask").click().run()
    text = " ".join(m.value for m in at.markdown)
    check("asking renders a verified answer card", not at.exception and "Answered from the document" in text
          and "166 depository libraries" in text, str(at.exception) or text[:300])
    check("the naive comparison renders beside it", "Naive RAG" in text)
    next(b for b in at.button if b.label.startswith("Re-run naive")).click().run()
    live = [df.value for df in at.dataframe if "live" in df.value.columns]
    check("'Re-run live' reproduces naive and strategy scores beside the saved ones",
          not at.exception and live and list(live[0]["variant"]) == ["naive", "advanced"],
          str(at.exception))


if __name__ == "__main__":
    test_cleaning()
    test_bm25()
    test_tables()
    test_footnotes()
    test_scoring()
    test_gep_structure()
    test_retrieval()
    test_eval_set()
    test_built_database()
    test_pages()
    print(f"\nall {_checks} checks passed")
