"""
Assignment 1 - plain narrative text (GPO 2018 Annual Report).

The baseline shape: facts sit inside ordinary sentences, so the job is clean
text and precise retrieval rather than structure. Versus the naive pipeline:

- page furniture is stripped - the running header repeats on 14 pages and
  was being embedded into every chunk on them;
- smaller chunks with their neighbours attached ("sentence window"): each
  chunk is matched on one idea, then handed to the model with the text
  around it, so precision does not cost context;
- hybrid search (vectors + BM25) and a cross-encoder rerank.

The measurements behind each choice are in docs/1-gpo.md.
"""

from __future__ import annotations

from dataclasses import replace

from rag import core
from rag.assignments import DOCS
from rag.retrieval import Settings, finalise

DOC_ID = "gpo"
TITLE = "GPO 2018 Annual Report"
SOURCE = "GPO Annual Report 2018"
PDF = DOCS / "gpo_annual_report_2018.pdf"

# Chosen by the chunk-size experiment in evaluate.py (results in docs/1-gpo.md).
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

SETTINGS = Settings(window=1)
ABLATIONS = {
    "no neighbours": replace(SETTINGS, window=0),
    "no rerank": replace(SETTINGS, rerank=False),
    "dense only": replace(SETTINGS, hybrid=False, rerank=False),
}


def build(api_key: str | None = None, chunk_size: int = CHUNK_SIZE) -> list:
    pages = core.strip_page_furniture(core.parse_document(PDF.read_bytes(), PDF.name))
    overlap = min(CHUNK_OVERLAP, chunk_size // 5)
    return finalise(
        core.chunk_pages(pages, SOURCE, chunk_size=chunk_size, chunk_overlap=overlap), DOC_ID
    )


def route(question: str):
    """A narrative report has no sections worth routing to."""
    return None, ""
