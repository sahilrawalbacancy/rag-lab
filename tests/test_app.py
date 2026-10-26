"""
UI-level checks for app.py, using Streamlit's own AppTest harness.

Separate from test_rag.py because these drive the real app end to end and hit
the API, so they are slower. They exist because the upload path - the primary
user journey - is glue code that unit tests cannot reach: parse/chunk are
tested directly in test_rag.py, but the wiring from an uploaded file through
st.file_uploader into the index is only exercised here.

    .venv/bin/python tests/test_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from streamlit.testing.v1 import AppTest  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
# The page runs standalone here; in the deployed app streamlit_app.py's
# st.navigation runs the same file.
APP = str(ROOT / "views" / "compliance_copilot.py")
_checks = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _checks
    _checks += 1
    if not condition:
        raise AssertionError(f"{label}" + (f" -- {detail}" if detail else ""))
    print(f"  ok  {label}")


def upload(name: str, mime: str = "text/plain") -> tuple[str, bytes, str]:
    return (name, (SAMPLES / name).read_bytes(), mime)


def fresh() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    check("app loads without an exception", not at.exception, str(at.exception))
    return at


def test_sample_button() -> None:
    print("\n[sample documents -> audit]")
    at = fresh()
    next(b for b in at.button if "sample" in b.label.lower()).click().run()
    check("samples indexed", any("passages indexed" in s.value for s in at.success),
          str([s.value for s in at.success]))

    next(b for b in at.button if b.label.startswith("Audit")).click().run()
    check("audit completes without an exception", not at.exception, str(at.exception))
    metrics = {m.label: m.value for m in at.metric}
    check("all four metric tiles render", len(metrics) == 4, str(metrics))
    check("every question got a verdict",
          int(metrics["Questions"]) ==
          sum(int(metrics[k]) for k in metrics if k != "Questions"), str(metrics))
    check("the gap callout is shown", bool(at.warning))
    check("csv export offered", bool(at.download_button))


def test_upload_knowledge_base() -> None:
    print("\n[upload knowledge base]")
    at = fresh()
    at.file_uploader[0].set_value([
        upload("acme_security_policy.txt"),
        upload("acme_pentest_assessment.pdf", "application/pdf"),
    ]).run()
    next(b for b in at.button if b.label == "Add documents").click().run()
    check("mixed txt + pdf upload indexes", not at.exception, str(at.exception))
    check("both documents reported indexed",
          any("2 document(s)" in s.value for s in at.success),
          str([s.value for s in at.success]))


def test_bad_upload_is_reported_not_crashed() -> None:
    print("\n[unusable upload]")
    at = fresh()
    at.file_uploader[0].set_value([("blank.txt", b"  \n \n ", "text/plain")]).run()
    next(b for b in at.button if b.label == "Add documents").click().run()
    check("no exception leaks to the user", not at.exception, str(at.exception))
    errors = " ".join(e.value for e in at.error)
    check("the reason is explained", "no text could be extracted" in errors, errors[:160])
    check("and it does not claim to have indexed anything",
          "Nothing could be indexed" in errors)


def test_upload_questionnaire() -> None:
    print("\n[upload questionnaire -> audit]")
    at = fresh()
    next(b for b in at.button if "sample" in b.label.lower()).click().run()
    at.radio[0].set_value("Upload a questionnaire").run()
    at.file_uploader[1].set_value([upload("vendor_security_questionnaire.txt")]).run()
    check("questions extracted from the upload", not at.exception, str(at.exception))
    check("all 12 found", any("Found 12 questions" in s.value for s in at.success),
          str([s.value for s in at.success]))
    check("questions are editable before running",
          bool(at.text_area) and len(at.text_area[0].value.splitlines()) == 12)

    next(b for b in at.button if b.label.startswith("Audit")).click().run()
    check("audit runs from the uploaded questionnaire", not at.exception, str(at.exception))
    check("results rendered", bool(at.metric))


def test_chat() -> None:
    print("\n[chat]")
    at = fresh()
    next(b for b in at.button if "sample" in b.label.lower()).click().run()
    at.chat_input[0].set_value("What TLS version is enforced?").run()
    check("chat replies without an exception", not at.exception, str(at.exception))
    reply = " ".join(m.value for msg in at.chat_message for m in msg.markdown)
    check("the answer comes from the documents", "TLS 1.3" in reply, reply[:200])


def test_session_isolation() -> None:
    """
    Two users must not share a vector store.

    Regression test for a real defect: every session used the collection name
    "knowledge_base" in the same directory, so a second user clicking "Build
    index" called reset_collection() and destroyed the first user's data. The
    first user's handle then pointed at a deleted collection id and their
    session raised chromadb NotFoundError - or, worse timing, served them the
    other user's documents.
    """
    print("\n[session isolation]")
    alice = AppTest.from_file(APP, default_timeout=300); alice.run()
    bob = AppTest.from_file(APP, default_timeout=300); bob.run()

    alice.file_uploader[0].set_value([upload("acme_security_policy.txt")]).run()
    next(b for b in alice.button if b.label == "Add documents").click().run()

    # The click that used to wipe Alice's collection.
    bob.file_uploader[0].set_value(
        [upload("acme_pentest_assessment.pdf", "application/pdf")]).run()
    next(b for b in bob.button if b.label == "Add documents").click().run()

    check("each session gets its own collection",
          alice.session_state["collection"] != bob.session_state["collection"])
    check("alice's index actually built",
          alice.session_state["store"] is not None,
          "errors=" + str([e.value[:200] for e in alice.error]))

    alice.chat_input[0].set_value("What algorithm encrypts data at rest?").run()
    check("alice's session survives bob's build", not alice.exception, str(alice.exception))
    a_reply = " ".join(m.value for msg in alice.chat_message for m in msg.markdown)
    check("alice still reaches her own document", "AES-256" in a_reply, a_reply[:160])
    check("alice cannot see bob's document",
          "Cobalt" not in a_reply and "penetration" not in a_reply.lower(), a_reply[:160])

    bob.chat_input[0].set_value("How many findings did the penetration test raise?").run()
    b_reply = " ".join(m.value for msg in bob.chat_message for m in msg.markdown)
    check("bob cannot see alice's document", "AES-256" not in b_reply, b_reply[:160])


def test_reload_recovery() -> None:
    """
    A browser reload must keep the index, and must not leak it to anyone else.

    session_state is wiped on reload, so the collection name lives in the URL.
    Before this, every reload silently orphaned a collection on disk holding
    the user's document text, and forced a full re-embed to carry on.
    """
    print("\n[reload recovery]")
    at = fresh()
    at.file_uploader[0].set_value([upload("acme_security_policy.txt")]).run()
    next(b for b in at.button if b.label == "Add documents").click().run()
    name = at.query_params["s"]
    name = name[0] if isinstance(name, (list, tuple)) else name
    check("the collection name is published in the url", str(name).startswith("kb_"))

    reloaded = AppTest.from_file(APP, default_timeout=300)
    reloaded.query_params["s"] = name
    reloaded.run()
    check("reload does not raise", not reloaded.exception, str(reloaded.exception))
    check("reload keeps the index", reloaded.session_state["store"] is not None)
    check("reload restores the document list",
          reloaded.session_state["indexed"] == [("acme_security_policy.txt", 8)],
          str(reloaded.session_state["indexed"]))

    reloaded.chat_input[0].set_value("What algorithm encrypts data at rest?").run()
    reply = " ".join(m.value for msg in reloaded.chat_message for m in msg.markdown)
    check("the recovered index still answers", "AES-256" in reply, reply[:140])

    stranger = AppTest.from_file(APP, default_timeout=200); stranger.run()
    check("a visitor without the url inherits nothing",
          stranger.session_state["store"] is None)

    for label, bad in [("junk", "not-a-collection"), ("stale", "kb_deadbeef"), ("empty", "")]:
        broken = AppTest.from_file(APP, default_timeout=200)
        broken.query_params["s"] = bad
        broken.run()
        check(f"a {label} url degrades gracefully",
              not broken.exception and broken.session_state["store"] is None)


def test_assignment_pages() -> None:
    """Each assignment page answers an official question live, with a verified quote."""
    print("\n[assignment pages - live]")
    cases = (
        ("gpo.py", "gpo", "How many depository libraries did GPO staff visit in FY 2018?", "166"),
        ("world_bank.py", "gep", "What is the growth outlook for South Asia in this report?", "6.3"),
        ("imac.py", "imac", "How much less energy does iMac use compared to the ENERGY STAR requirement?", "58"),
    )
    for page, doc_id, question, fact in cases:
        at = AppTest.from_file(str(ROOT / "views" / page), default_timeout=600)
        at.run()
        check(f"{doc_id}: page loads with its three tabs", not at.exception and len(at.tabs) >= 3,
              str(at.exception))
        at.text_input(key=f"{doc_id}_typed").set_value(question)
        next(b for b in at.button if b.label == "Ask").click().run()
        text = " ".join(m.value for m in at.markdown)
        # Judge the strategy's own column: the naive column beside it can say
        # "Answered" too, and a page-wide check once hid a strategy failure.
        strategy = text[text.find("This document's strategy"):text.find("Naive RAG")]
        check(f"{doc_id}: the STRATEGY answers the official question from the document",
              not at.exception and "Answered from the document" in strategy and fact in strategy,
              str(at.exception) or strategy[:300])


if __name__ == "__main__":
    if not (os.environ.get("GOOGLE_API_KEY") or "").strip():
        print("SKIP - no GOOGLE_API_KEY in .env"); raise SystemExit(0)
    test_sample_button()
    test_upload_knowledge_base()
    test_bad_upload_is_reported_not_crashed()
    test_upload_questionnaire()
    test_chat()
    test_session_isolation()
    test_reload_recovery()
    test_assignment_pages()
    print(f"\nall {_checks} UI checks passed")
