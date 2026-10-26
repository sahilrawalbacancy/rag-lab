"""
Evaluation: the same questions through every pipeline variant, scored by code.

A question PASSES only when all three hold:

1. facts      - the answer states every expected fact (numbers normalised:
                "15,901,600" == "15901600", "−8.9" == "-8.9");
2. grounded   - the verdict is COVERED/PARTIAL with at least one quote the
                guardrail verified in the retrieved text;
3. evidence   - a verified quote comes from an expected page - the right
                region, chapter or table, not a page that merely repeats it.

Refusal questions pass only with NOT_FOUND. No LLM judges the answers:
every pass/fail can be explained by pointing at a string or a page number.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict

from rag import core
from rag.assignments import EVAL_SET
from rag.retrieval import Answer, Index, Settings, ask


def questions(doc_id: str) -> list[dict]:
    return json.loads(EVAL_SET.read_text())[doc_id]


def normalise(text: str) -> str:
    text = text.lower().replace("−", "-").replace("–", "-")
    text = text.replace("aluminium", "aluminum")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)       # 15,901,600 -> 15901600
    return re.sub(r"\s+", " ", text)


def score(spec: dict, answer: Answer) -> dict:
    finding = answer.finding
    text = normalise(finding.answer)
    retrieved_pages = sorted({h.doc.metadata["page"] for h in answer.context})
    cited_pages = sorted({c.page for c in finding.citations})

    if spec.get("refuse"):
        refused = finding.verdict == core.NOT_FOUND
        return {"passed": refused, "facts": refused, "grounded": refused, "evidence": refused,
                "retrieved_expected": None, "missing": [] if refused else ["a refusal"],
                "retrieved_pages": retrieved_pages, "cited_pages": cited_pages}

    expect = spec.get("expect", {})
    missing = [f for f in expect.get("all", []) if normalise(f) not in text]
    groups = expect.get("any", [])
    met = sum(any(normalise(alt) in text for alt in group) for group in groups)
    needed = expect.get("min_any", len(groups))
    if met < needed:
        missing.append(f"{needed - met} more of: " + " / ".join("|".join(g) for g in groups))

    pages = set(spec.get("pages", []))
    grounded = finding.verdict in (core.COVERED, core.PARTIAL) and bool(finding.citations)
    evidence = not pages or bool(pages & set(cited_pages))
    return {
        "passed": not missing and grounded and evidence,
        "facts": not missing,
        "grounded": grounded,
        "evidence": evidence,
        "retrieved_expected": (not pages) or bool(pages & set(retrieved_pages)),
        "missing": missing,
        "retrieved_pages": retrieved_pages,
        "cited_pages": cited_pages,
    }


def _record(spec: dict, answer: Answer | None, error: str = "") -> dict:
    if answer is None:
        return {"id": spec["id"], "question": spec["question"], "kind": spec["kind"],
                "tests": spec["tests"], "passed": False, "error": error}
    f = answer.finding
    return {
        "id": spec["id"], "question": spec["question"], "kind": spec["kind"], "tests": spec["tests"],
        **score(spec, answer),
        "verdict": f.verdict, "answer": f.answer, "note": f.note, "route": answer.route,
        "seconds": answer.seconds,
        "citations": [{"label": c.label(), "quote": c.quote, "page": c.page} for c in f.citations],
        "context": [
            {"page": h.doc.metadata["page"], "section": h.doc.metadata.get("section", ""),
             "why": h.why, "rerank": h.rerank}
            for h in answer.context
        ],
    }


def run_variant(doc_id: str, index: Index, settings: Settings, llm, route_fn=None,
                specs: list[dict] | None = None) -> dict:
    """Every question through one variant. A crash on one question is recorded, not raised."""
    records = []
    for spec in specs or questions(doc_id):
        try:
            records.append(_record(spec, ask(index, spec["question"], llm, settings, route_fn)))
        except Exception as exc:
            records.append(_record(spec, None, f"{type(exc).__name__}: {exc}"))
    return {"settings": asdict(settings), "description": settings.describe(),
            "index": index.name, "summary": summarise(records), "questions": records}


def summarise(records: list[dict]) -> dict:
    n = len(records)
    count = lambda key: sum(1 for r in records if r.get(key))
    scored = [r for r in records if r.get("retrieved_expected") is not None]
    return {
        "total": n,
        "passed": count("passed"),
        "facts": count("facts"),
        "grounded": count("grounded"),
        "evidence": count("evidence"),
        "retrieved_expected": sum(1 for r in scored if r["retrieved_expected"]),
        "retrieval_scored": len(scored),
        "errors": count("error"),
        "avg_seconds": round(sum(r.get("seconds", 0) for r in records) / n, 2) if n else 0,
        "when": time.strftime("%Y-%m-%d %H:%M"),
    }
