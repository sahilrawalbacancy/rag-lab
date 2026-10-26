"""
Assignment 2 - a long, structured report (World Bank GEP, June 2026).

200 pages in which every question's topic is scattered: "South Asia" appears
on 16 pages but the South Asia section is p.96-98. The job is to find the
right SECTION, then the right passage inside it. Versus the naive pipeline:

- a section map built from the PDF's bookmarks - sorted, each snapped to the
  page where its heading really appears (they are sometimes a page early),
  with boxed articles ending where their "(continued)" headers stop;
- every chunk carries its place: chapter / region / box metadata, and a
  breadcrumb line ("Chapter 2 · Regional Outlooks › South Asia") that
  travels into the embedding, the keyword index and the citation;
- table pages and the statistical annex are transcribed by Gemini vision:
  pypdf tears their columns apart (p.98's 2028 column ends up below the
  table) and footnote markers shift every value ("Iraq 1 0.5 -1.5 ...");
- tables are split into row groups that each repeat the header, so a row
  never loses the years its numbers belong to;
- a question that names a region, chapter, box or the executive summary is
  searched only inside it, and always sees that section's opening.
"""

from __future__ import annotations

import re
from dataclasses import replace

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import core, vision
from rag.assignments import DOCS, VISION_DIR
from rag.retrieval import Settings, finalise

DOC_ID = "gep"
TITLE = "World Bank Global Economic Prospects, June 2026"
SOURCE = "GEP June 2026"
PDF = DOCS / "world_bank_gep_june_2026.pdf"

SETTINGS = Settings(window=1, route=True, lead=True)
ABLATIONS = {
    "no routing": replace(SETTINGS, route=False, lead=False),
    "no rerank": replace(SETTINGS, rerank=False),
    "dense only": replace(SETTINGS, hybrid=False, rerank=False),
}

# Canonical region name -> how questions refer to it. Bare "Middle East" is
# deliberately absent: half the report discusses "the conflict in the Middle
# East", and routing every such question to one region would be wrong.
REGIONS = {
    "East Asia and Pacific": ["east asia and pacific", "east asia and the pacific", "eap"],
    "Europe and Central Asia": ["europe and central asia", "eca"],
    "Latin America and the Caribbean": ["latin america and the caribbean", "latin america", "lac"],
    "Middle East, North Africa, Afghanistan, and Pakistan": [
        "middle east, north africa, afghanistan, and pakistan",
        "middle east and north africa", "north africa", "mena", "menaap", "mna",
    ],
    "South Asia": ["south asia", "sar"],
    "Sub-Saharan Africa": ["sub-saharan africa", "ssa"],
}

_TABLE_HEADING = re.compile(r"^TABLE [A-Z]*\d", re.M)
_DROP_CAP = re.compile(r"7['’]?T?_")       # "7T_he" / "7'_he" -> "The"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# The section map
# ---------------------------------------------------------------------------


def _outline(reader) -> list[tuple[str, int]]:
    out = []

    def walk(items):
        for item in items:
            if isinstance(item, list):
                walk(item)
                continue
            number = reader.get_destination_page_number(item)
            if number is not None:
                out.append((" ".join(item.title.replace("--", " ").split()), number + 1))

    walk(reader.outline)
    return sorted(out, key=lambda entry: entry[1])


def _snap(title: str, page: int, raw: dict[int, str]) -> int:
    """The page whose opening lines carry the heading - bookmarks drift."""
    want = _norm(title)[:18]
    for candidate in (page, page + 1, page - 1, page + 2):
        lines = raw.get(candidate, "").splitlines()[:12]
        if any(_norm(line).startswith(want) for line in lines):
            return candidate
    return page


def _chapter_name(title: str) -> str:
    m = re.match(r"Ch\.\s*(\d+)\s+(.*)", title)
    if m:
        return f"Chapter {m.group(1)} · {m.group(2)}"
    return "Contents" if "contents" in title.lower() else title


def section_map(reader, raw: dict[int, str]) -> dict[int, dict]:
    """
    {page: {chapter, section, chapter_no, region, box}} for every page.

    Three kinds of bookmark: chapters (plus the executive summary and the
    annex), regions (inside Chapter 2), and boxed articles. A box ends at the
    last page within reach whose header says "BOX n.n ... (continued)";
    without that, Box 1.1 would swallow p.30-43 instead of p.30-35.
    """
    entries = [(t, _snap(t, p, raw)) for t, p in _outline(reader)]
    chapters = [(t, p) for t, p in entries if not t.lower().startswith("box ") and t not in REGIONS]
    regions = sorted((t, p) for t, p in entries if t in REGIONS)
    regions.sort(key=lambda e: e[1])
    boxes = [(t, p) for t, p in entries if t.lower().startswith("box ")]
    last_page = len(reader.pages)

    def chapter_at(page: int) -> str:
        started = [t for t, p in chapters if p <= page]
        return _chapter_name(started[-1]) if started else "Front matter"

    region_at: dict[int, str] = {}
    for i, (name, start) in enumerate(regions):
        later = [p for _, p in regions[i + 1:]] + [p for _, p in chapters if p > start]
        for page in range(start, (min(later) if later else last_page + 1)):
            region_at[page] = name

    box_at: dict[int, tuple[str, str]] = {}
    for title, start in boxes:
        number = re.match(r"Box\s+([\d.]+)", title).group(1)
        marker = f"BOX {number} "
        # The "(continued)" header can land anywhere in a page's extracted
        # text - column order, not visual order - so the whole page is read.
        end = max(
            [p for p in range(start, min(start + 16, last_page + 1))
             if any(line.startswith(marker) for line in raw.get(p, "").splitlines())],
            default=start,
        )
        for page in range(start, end + 1):
            box_at[page] = (title, number)

    pages = {}
    for page in range(1, last_page + 1):
        chapter = chapter_at(page)
        if any(line.strip() == "Abbreviations" for line in raw.get(page, "").splitlines()):
            chapter = "Abbreviations"
        number = re.match(r"Chapter (\d+)", chapter)
        region = region_at.get(page, "") if chapter.startswith("Chapter 2") else ""
        box_title, box_no = box_at.get(page, ("", ""))
        pages[page] = {
            "chapter": chapter,
            "section": box_title or region or chapter,
            "chapter_no": number.group(1) if number else "",
            "region": region,
            "box": box_no,
        }
    return pages


def printed_page(raw_text: str) -> str:
    """The page number printed in the running header ("... J U N E 2 0 2 6 74", "xvii")."""
    first = raw_text.splitlines()[0] if raw_text else ""
    m = re.search(r"(\d{1,3}|[ivxlc]{1,7})\s*$", first)
    return m.group(1) if m else ""


def breadcrumb(meta: dict) -> str:
    return meta["chapter"] if meta["section"] == meta["chapter"] else f"{meta['chapter']} › {meta['section']}"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=core.CHUNK_SIZE, chunk_overlap=core.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def markdown_blocks(markdown: str) -> list[tuple[str, str]]:
    """Split Markdown into ("text", ...) and ("table", ...) runs, in order."""
    blocks, buffer, lines, i = [], [], markdown.splitlines(), 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            if buffer:
                blocks.append(("text", "\n".join(buffer).strip()))
                buffer = []
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            blocks.append(("table", "\n".join(lines[i:j])))
            i = j
        else:
            buffer.append(lines[i])
            i += 1
    if buffer:
        blocks.append(("text", "\n".join(buffer).strip()))
    return [(kind, text) for kind, text in blocks if text]


def split_table(table: str, title: str, limit: int = 1200) -> list[str]:
    """
    Row groups of at most `limit` characters, each repeating title + header.

    A row on its own ("| Iraq [^1] | 0.5 | -1.5 | -2.2 | -8.9 |") means
    nothing without the years above it, and the header is far from most rows
    once a long table is split - so every piece carries it.
    """
    rows = table.splitlines()
    header_len = 2 if len(rows) > 1 and set(rows[1].replace("|", "").strip()) <= set("-: ") else 1
    header, body = rows[:header_len], rows[header_len:]
    pieces, current = [], []
    for row in body:
        if current and len("\n".join([title, *header, *current, row])) > limit:
            pieces.append(current)
            current = []
        current.append(row)
    if current or not pieces:
        pieces.append(current)
    return ["\n".join(filter(None, [title, *header, *piece])) for piece in pieces]


def label_repeated_columns(table: str, meaning: str) -> str:
    """
    Give repeated header cells their meaning.

    GEP forecast tables end with a second "2026f | 2027f" pair that holds
    percentage-point CHANGES from the January forecast, not growth rates.
    Asked for "2026", a model can read the change (-0.2) instead of the
    forecast (4.2); spelling the meaning into the header removes the trap.
    """
    rows = table.splitlines()
    cells = rows[0].split("|")
    seen: set[str] = set()
    for i, cell in enumerate(cells):
        label = cell.strip()
        if label and label in seen:
            cells[i] = f" {label} ({meaning}) "
        seen.add(label)
    return "\n".join(["|".join(cells), *rows[1:]])


def _vision_chunks(markdown: str, page_text: str = "") -> list[tuple[str, str]]:
    """(kind, text) pieces from one transcribed page."""
    out, previous_text = [], ""
    # The spanning header is sometimes dropped by vision, never by the text
    # layer - where it is split across lines, hence \s+.
    spanning = re.search(r"differences\s+from\s+January", f"{markdown}\n{page_text}")
    meaning = "change vs January 2026 forecast, percentage points" if spanning else "repeated column"
    for kind, text in markdown_blocks(markdown):
        if kind == "table":
            title_lines = [line for line in previous_text.splitlines() if line.strip()][-2:]
            text = label_repeated_columns(text, meaning)
            out += [("table", piece) for piece in split_table(text, "\n".join(title_lines))]
        else:
            out += [("text", piece) for piece in _splitter.split_text(text)]
            previous_text = text
    return out


def vision_pages(raw: dict[int, str], pages: dict[int, dict]) -> list[int]:
    """Pages whose tables plain extraction mangles: TABLE headings, and the annex."""
    return sorted(
        p for p, text in raw.items()
        if _TABLE_HEADING.search(text) or pages[p]["chapter"] == "Statistical Annex"
    )


def build(api_key: str | None = None) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(PDF))
    parsed = core.parse_document(PDF.read_bytes(), PDF.name)
    raw = dict(parsed)
    pages = section_map(reader, raw)
    cleaned = dict(core.strip_page_furniture(parsed))
    to_transcribe = vision_pages(raw, pages)
    transcribed = vision.transcribe_pages(PDF, to_transcribe, VISION_DIR / DOC_ID, api_key)

    docs: list[Document] = []
    for page in sorted(raw):
        meta = {
            "source": SOURCE, "page": page, "printed_page": printed_page(raw[page]),
            **pages[page],
        }
        crumb = breadcrumb(meta)
        if page in transcribed:
            pieces, origin = _vision_chunks(transcribed[page], raw[page]), "vision"
        else:
            text = _DROP_CAP.sub("T", cleaned.get(page, ""))
            pieces, origin = [("text", p) for p in _splitter.split_text(text)], "text layer"
        for kind, piece in pieces:
            if piece.strip():
                docs.append(Document(
                    page_content=f"{crumb}\n{piece.strip()}",
                    metadata={**meta, "kind": kind, "origin": origin},
                ))
    return finalise(docs, DOC_ID)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _says(question: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", question) is not None


def route(question: str) -> tuple[dict | None, str]:
    """
    The section a question explicitly names, as a metadata filter.

    Most specific wins: a box, then regions, then a chapter / the executive
    summary / the annex. Only explicit names route - a question that names
    nothing searches the whole report, because a wrong filter is worse than
    no filter: it hides the answer completely.
    """
    q = question.lower()
    box = re.search(r"\bbox\s+(\d+\.\d+)", q)
    if box:
        return {"box": box.group(1)}, f"Box {box.group(1)}"
    named = [name for name, aliases in REGIONS.items() if any(_says(q, a) for a in aliases)]
    if named:
        return {"region": named if len(named) > 1 else named[0]}, " + ".join(named)
    chapter = re.search(r"\b(?:chapter|ch\.?)\s*(\d)\b", q)
    if chapter:
        return {"chapter_no": chapter.group(1)}, f"Chapter {chapter.group(1)}"
    if "executive summary" in q:
        return {"chapter": "Executive Summary"}, "Executive Summary"
    if re.search(r"statistical (annex|appendix)", q):
        return {"chapter": "Statistical Annex"}, "Statistical Annex"
    return None, ""
