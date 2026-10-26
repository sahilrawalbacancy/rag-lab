"""
Compliance Copilot - the questionnaire-specific half of the pipeline.

Everything here is about auditing a whole questionnaire: preset checklists,
pulling questions out of an uploaded file, answering them all concurrently and
exporting the result. The shared RAG stages it calls (retrieve, the
quote-verified answer_question) live in rag.core.
"""

from __future__ import annotations

import re

from rag.core import (
    COVERED, ERROR, NOT_FOUND, PARTIAL, TOP_K, Finding, answer_question,
    get_text, retrieve, with_retry,
)

# ---------------------------------------------------------------------------
# 1. THE CHECKLIST (questions come from a preset, or from an uploaded file)
# ---------------------------------------------------------------------------

# Presets exist so the app is demoable in one click without hunting for a
# questionnaire PDF. Upload mode is the general-purpose path.
PRESETS: dict[str, list[str]] = {
    "Vendor Security Assessment (12 questions)": [
        "Is customer data encrypted at rest? State the algorithm used.",
        "Is data encrypted in transit? State the minimum TLS version supported.",
        "Do you enforce multi-factor authentication for administrative access?",
        "What is the Recovery Time Objective (RTO) in a disaster recovery scenario?",
        "How long are security audit logs retained?",
        "What is the process for revoking access when an employee leaves?",
        "Is the organisation certified against ISO 27001?",
        "Is independent third-party penetration testing performed at least annually?",
        "In which geographic regions is customer data stored?",
        "Is cyber liability insurance carried? State the coverage limit.",
        "Is production data ever used in non-production environments?",
        "How are application secrets and API keys managed?",
    ],
    "GDPR Readiness (8 questions)": [
        "Is a Data Protection Officer appointed, and how can they be contacted?",
        "What is the lawful basis for processing personal data?",
        "How can a data subject exercise their right of access?",
        "How can a data subject request erasure of their personal data?",
        "What is the personal data breach notification timeline to the supervisory authority?",
        "Is a record of processing activities maintained?",
        "Is personal data transferred outside the EEA, and under what safeguard?",
        "How long is personal data retained, and what triggers deletion?",
    ],
    "SaaS Due Diligence (10 questions)": [
        "What uptime or availability level is committed to in the SLA?",
        "What is the Recovery Point Objective (RPO) for customer data?",
        "How frequently are backups taken and how long are they retained?",
        "Is the platform single-tenant or multi-tenant, and how is tenant data isolated?",
        "Which third-party subprocessors have access to customer data?",
        "What is the notice period for terminating the contract?",
        "What happens to customer data on contract termination?",
        "Is there a documented incident response plan?",
        "What authentication methods are supported for end users?",
        "Is there an audited compliance certification such as SOC 2 or ISO 27001?",
    ],
}

# A questionnaire line is treated as a question if it looks like one. Used as a
# cheap pre-filter before the LLM extraction pass, and as the fallback if that
# pass fails.
_QUESTION_LINE = re.compile(
    r"""^\s*
        # "1.1" must be tried before "1." - otherwise "1.1 Is..." keeps "1 Is..."
        (?:\d+\.\d+|\(?\d+[.)]|[-*•]|[a-zA-Z][.)])?   # optional bullet/number
        \s*(?P<body>\S.*?)\s*$
    """,
    re.VERBOSE,
)


def extract_questions_heuristic(text: str, min_len: int = 15) -> list[str]:
    """
    Pull questions out of raw questionnaire text without calling the LLM.

    Keeps lines that either end in '?' or start with a numbered/bulleted
    marker and read as an instruction ("Describe...", "State..."). This is the
    fallback path — extract_questions() prefers the LLM, which handles
    questions that wrap across several lines.
    """
    imperative = (
        "describe", "state", "explain", "list", "confirm", "provide",
        "detail", "specify", "outline", "identify",
    )
    out: list[str] = []
    for raw in text.splitlines():
        m = _QUESTION_LINE.match(raw)
        if not m:
            continue
        body = m.group("body").strip()
        if len(body) < min_len:
            continue
        # A '?' ANYWHERE in the line, not just at the end. Real
        # questionnaires constantly write "Is data encrypted at rest? State
        # the algorithm used." — testing only endswith('?') drops every one of
        # those, and silently dropping a question means silently hiding a gap.
        looks_like_question = "?" in body or body.lower().startswith(imperative)
        if looks_like_question:
            out.append(body)
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# 2. QUESTION EXTRACTION FROM AN UPLOADED QUESTIONNAIRE
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """Below is the text of a questionnaire, assessment or \
checklist.

Extract every question or information request a respondent is expected to \
answer. Reply with ONLY a JSON array of strings, no prose and no code fence:
["first question", "second question"]

Rules:
- Copy each question's wording; do not rewrite or summarise it.
- Drop the numbering or bullet marker.
- A question split across several lines becomes ONE entry.
- Keep any trailing instruction that is part of the request, e.g. \
"Is data encrypted at rest? State the algorithm used."
- Ignore headings, section titles, instructions to the assessor, and \
signature or metadata blocks.

Questionnaire text:
{text}
"""

# Questionnaires are usually short; this cap keeps one oversized upload from
# blowing the context window. Text beyond it falls back to the heuristic.
_EXTRACT_CHAR_LIMIT = 24_000


def extract_questions(text: str, llm=None) -> list[str]:
    """
    Pull the question list out of raw questionnaire text.

    Prefers the LLM, which handles wrapped lines, tables and inconsistent
    numbering. Falls back to the regex heuristic when there is no LLM, the
    text is too long, or the model returns something unusable - the fallback
    matters because a dropped question silently becomes a hidden gap.
    """
    if llm is None or len(text) > _EXTRACT_CHAR_LIMIT:
        return extract_questions_heuristic(text)

    try:
        import json

        raw = get_text(llm.invoke(EXTRACT_PROMPT.format(text=text)))
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("no JSON array in reply")
        items = json.loads(raw[start : end + 1])
        questions = [
            str(q).strip() for q in items if isinstance(q, (str, int, float)) and str(q).strip()
        ]
        questions = list(dict.fromkeys(questions))
        if not questions:
            raise ValueError("model returned an empty list")
        return questions
    except Exception:
        return extract_questions_heuristic(text)


# ---------------------------------------------------------------------------
# 3. BATCH AUDIT
# ---------------------------------------------------------------------------

# Free-tier Gemini keys are rate limited per minute. Four concurrent workers
# cuts a 12-question audit from ~48s to ~15s while staying well inside the
# limit; with_retry absorbs the 429s that do slip through. Without the retry
# a long audit dies half-way and reports phantom gaps for every question that
# never ran - which would be indistinguishable from a genuine finding.
MAX_WORKERS = 4


def run_audit(
    questions: list[str],
    store,
    llm,
    embed_model: str,
    api_key: str | None = None,
    k: int = TOP_K,
    max_workers: int = MAX_WORKERS,
    on_result=None,
) -> list[Finding]:
    """
    Answer every question against the knowledge base.

    Questions run concurrently but results are returned in the original order,
    so the report always matches the questionnaire. A question that fails
    outright becomes an ERROR finding carrying the error rather than
    vanishing - a missing row would read as "not asked" instead of "broken",
    and a NOT_FOUND row would claim the documents were checked when they
    never were.

    on_result(index, finding) is called as each one lands, for UI progress.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[Finding | None] = [None] * len(questions)

    def one(index: int, question: str) -> tuple[int, Finding]:
        try:
            # retrieve() already retries internally.
            hits = retrieve(store, question, embed_model, k=k, api_key=api_key)
            return index, with_retry(lambda: answer_question(llm, question, hits))
        except Exception as exc:
            return index, Finding(
                question=question,
                verdict=ERROR,
                answer="",
                note=f"Failed to process: {type(exc).__name__}: {exc}",
            )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(one, i, q) for i, q in enumerate(questions)]
        for future in as_completed(futures):
            index, finding = future.result()
            results[index] = finding
            if on_result:
                on_result(index, finding)

    return [f for f in results if f is not None]


# ---------------------------------------------------------------------------
# 4. EXPORT
# ---------------------------------------------------------------------------


def _csv_safe(value: str) -> str:
    """
    Neutralise spreadsheet formulas.

    Questions come from a client's questionnaire - outside, untrusted text. A
    cell starting with = + - @ runs as a formula when the CSV is opened in
    Excel or Sheets; the leading apostrophe makes it plain text.
    """
    return f"'{value}" if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def findings_to_csv(findings: list[Finding]) -> str:
    """Render findings as CSV - the format a real questionnaire gets returned in."""
    import csv
    import io as _io

    buf = _io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["#", "Question", "Verdict", "Answer", "Sources", "Evidence", "Note"])
    for i, f in enumerate(findings, start=1):
        writer.writerow([i] + [_csv_safe(cell) for cell in (
            f.question,
            f.verdict,
            f.answer,
            " | ".join(c.label() for c in f.citations),
            " | ".join(c.quote for c in f.citations),
            f.note,
        )])
    return buf.getvalue()


def summarise(findings: list[Finding]) -> dict[str, int]:
    """Counts per verdict, plus the coverage score shown in the UI."""
    counts = {COVERED: 0, PARTIAL: 0, NOT_FOUND: 0, ERROR: 0}
    for f in findings:
        counts[f.verdict] = counts.get(f.verdict, 0) + 1
    counts["total"] = len(findings)
    return counts
