"""
Assignment 3 - a short, visual document (Apple iMac Product Environmental Report).

Twelve pages where facts live in charts, icon call-outs, small tables and
endnotes rather than paragraphs. Versus the naive pipeline:

- every page is transcribed by Gemini vision into Markdown: chart values stay
  attached to their labels ("Baseline emissions: 445 kg" - the text layer
  jumbles them), tables keep their header row, and footnote markers become
  [^n] instead of digits glued onto words ("STAR.18", "content1");
- each transcription is checked against the PDF's own text layer: a number
  vision produced that is not in the PDF is flagged as unconfirmed;
- one chunk per heading block - a chart, a table, a call-out group - so a
  number never travels without its caption;
- footnotes are their own chunks, LINKED to the chunks that cite them: the
  "58% less energy[^18]" claim is on p.7, its test conditions on p.12, and
  retrieval brings them together;
- hybrid search, so exact tokens ("230V", "kg CO2e") match by keyword.
"""

from __future__ import annotations

import re
from dataclasses import replace

from langchain_core.documents import Document

from rag import core, vision
from rag.assignments import DOCS, VISION_DIR
from rag.retrieval import Settings, finalise
from rag.structured import _splitter, markdown_blocks

DOC_ID = "imac"
TITLE = "Apple iMac Product Environmental Report (October 2024)"
SOURCE = "iMac Environmental Report"
PDF = DOCS / "apple_imac_environmental_report_oct2024.pdf"

SETTINGS = Settings(footnotes=True)
ABLATIONS = {
    "no footnote links": replace(SETTINGS, footnotes=False),
    "no rerank": replace(SETTINGS, rerank=False),
    "dense only": replace(SETTINGS, hybrid=False, rerank=False),
}

_DEFINITION = re.compile(r"^\s*\[\^(\d+)\]:\s*(.*)")
_REFERENCE = re.compile(r"\[\^(\d+)\](?!:)")


def split_footnotes(markdown: str) -> tuple[str, dict[str, str]]:
    """Separate "[^n]: text" definitions (with continuation lines) from the body."""
    body, notes, current = [], {}, None
    for line in markdown.splitlines():
        m = _DEFINITION.match(line)
        if m:
            current = m.group(1)
            notes[current] = m.group(2).strip()
        elif current and line.strip() and not line.lstrip().startswith(("#", "|")):
            notes[current] += " " + line.strip()
        else:
            current = None
            body.append(line)
    return "\n".join(body).strip(), notes


def heading_blocks(markdown: str) -> list[tuple[str, str]]:
    """(heading, block) pairs: a new block starts at every Markdown heading."""
    blocks, heading, buffer = [], "", []
    for line in markdown.splitlines():
        if line.lstrip().startswith("#"):
            if "\n".join(buffer).strip():
                blocks.append((heading, "\n".join(buffer).strip()))
            heading, buffer = line.lstrip("#").strip(), [line]
        else:
            buffer.append(line)
    if "\n".join(buffer).strip():
        blocks.append((heading, "\n".join(buffer).strip()))
    return blocks


def _pieces(block: str, limit: int = 1200) -> list[tuple[str, str]]:
    """A block whole when it fits; otherwise split, never cutting a table."""
    if len(block) <= limit:
        return [("table" if "\n|" in f"\n{block}" else "text", block)]
    out = []
    for kind, text in markdown_blocks(block):
        out += [(kind, text)] if kind == "table" else [("text", p) for p in _splitter.split_text(text)]
    return out


def build(api_key: str | None = None) -> list[Document]:
    from pypdf import PdfReader

    page_count = len(PdfReader(str(PDF)).pages)
    transcribed = vision.transcribe_pages(PDF, list(range(1, page_count + 1)), VISION_DIR / DOC_ID, api_key)

    # A footnote can come back defined on more than one page: the model once
    # wrote "[^7]: GHG reductions achieved" under a table on p.10, while the
    # real endnote 7 is on p.11. The longest definition is the real one.
    split = {page: split_footnotes(md) for page, md in transcribed.items()}
    best: dict[str, tuple[int, str]] = {}
    for page, (_, notes) in split.items():
        for number, text in notes.items():
            if len(text) > len(best.get(number, (0, ""))[1]):
                best[number] = (page, text)

    docs: list[Document] = []
    for page in sorted(transcribed):
        body, notes = split[page]
        for number, text in notes.items():
            if best[number] != (page, text):
                continue
            docs.append(Document(
                page_content=f"Footnote {number}: {text}",
                metadata={"source": SOURCE, "page": page, "section": "Endnotes",
                          "kind": "footnote", "footnote": number, "origin": "vision"},
            ))
        for heading, block in heading_blocks(body):
            for kind, piece in _pieces(block):
                kind = "chart" if heading.lower().startswith("chart") else kind
                refs = sorted(set(_REFERENCE.findall(piece)), key=int)
                docs.append(Document(
                    page_content=piece,
                    metadata={"source": SOURCE, "page": page, "section": heading,
                              "kind": kind, "footnotes": ",".join(refs), "origin": "vision"},
                ))
    return finalise(docs, DOC_ID)


def fidelity() -> dict[int, list[str]]:
    """{page: numbers in the vision transcription that the PDF's text layer lacks}."""
    text_layer = dict(core.parse_document(PDF.read_bytes(), PDF.name))
    out = {}
    for md in sorted((VISION_DIR / DOC_ID).glob("*.md")):
        page = int(md.stem)
        out[page] = vision.unconfirmed_numbers(md.read_text(), text_layer.get(page, ""))
    return out


def route(question: str):
    """Twelve pages, no sections worth routing to."""
    return None, ""
