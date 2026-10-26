"""
Gemini vision: one PDF page in, Markdown out - for content plain text
extraction cannot represent (tables whose columns come apart, charts whose
labels lose their values, footnote markers glued to words).

Runs at ingestion time only, never while the app serves a question. Every
page's Markdown is cached under data/vision/<doc>/ and committed, so the
index can be rebuilt without calling the API again and anyone can read
exactly what the model extracted.
"""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path

from rag.core import with_retry

logging.getLogger("pypdf").setLevel(logging.ERROR)

# Best first. Availability comes and goes - "503: high demand", "404: no
# longer available to new users" for a model the key can still list, and on
# the free tier only 20 requests per model PER DAY - so every page falls back
# down this list. Flash-Lite models are the last resort; their output is held
# to the same number cross-check as everyone else's (unconfirmed_numbers).
VISION_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
]
_exhausted: set[str] = set()   # models whose daily quota ran out during this run

PROMPT = """Transcribe this PDF page into Markdown so that every fact on it is \
searchable as text.

- Keep all body text verbatim, in reading order. Do not summarise.
- Skip running page headers, footers and page numbers.
- Tables: one Markdown table per table, with its title above it and the full \
header row. A footnote marker attached to a row label (for example "Iraq 1") \
is written as a footnote reference: "Iraq [^1]". Never shift a value into \
another column.
- Charts and diagrams: a heading "Chart: <title>", then one bullet per bar, \
segment or data point with its label and its value exactly as printed. Read \
values ONLY from labels printed on the page; never estimate a value from a \
bar's height. If no values are printed, list the labels and say so.
- Icon call-outs: one bullet each, with the figure and its caption.
- Footnote reference numbers in the text: write them as [^n]. Footnote and \
endnote definitions: write each as [^n]: text.
Output only the Markdown, no commentary."""


def one_page_pdf(pdf_path: str | Path, page: int) -> bytes:
    """Cut page `page` (1-indexed) out of the PDF as its own one-page PDF."""
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    writer.add_page(PdfReader(str(pdf_path)).pages[page - 1])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _transcribe(pdf_bytes: bytes, api_key: str) -> tuple[str, str]:
    """Try each model in turn; return (markdown, model_used)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    errors = []
    for model in VISION_MODELS:
        if model in _exhausted:
            continue
        try:
            reply = with_retry(
                lambda: client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                        PROMPT,
                    ],
                    config=types.GenerateContentConfig(temperature=0),
                ),
                attempts=3,
                base_delay=5.0,
            )
            text = (reply.text or "").strip()
            if text:
                return _unfence(text), model
            errors.append(f"{model}: empty reply")
        except Exception as exc:  # 503 / 404 / quota - move down the list
            if "PerDay" in str(exc):
                _exhausted.add(model)
            errors.append(f"{model}: {str(exc)[:80]}")
    raise RuntimeError("every vision model failed: " + " | ".join(errors))


def _unfence(text: str) -> str:
    """Models sometimes wrap the whole answer in ```markdown fences."""
    match = re.fullmatch(r"```(?:markdown|md)?\s*\n(.*?)\n```", text, re.S)
    return match.group(1).strip() if match else text


def page_markdown(pdf_path: str | Path, page: int, cache_dir: str | Path, api_key: str | None) -> str:
    """
    Markdown for one page, from the cache when present.

    The cache is the source of truth once written: re-running ingestion never
    re-calls the model, so the index is reproducible and the API is only paid
    for once. Delete a page's .md file to force a fresh transcription.
    """
    cache_dir = Path(cache_dir)
    cached = cache_dir / f"{page:03d}.md"
    if cached.exists():
        return cached.read_text()
    if not api_key:
        raise RuntimeError(f"no cached vision output for page {page} and no API key")

    markdown, model = _transcribe(one_page_pdf(pdf_path, page), api_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(markdown + "\n")
    log = cache_dir / "models.json"
    used = json.loads(log.read_text()) if log.exists() else {}
    used[str(page)] = model
    log.write_text(json.dumps(dict(sorted(used.items(), key=lambda kv: int(kv[0]))), indent=1) + "\n")
    return markdown


def transcribe_pages(
    pdf_path: str | Path, pages: list[int], cache_dir: str | Path, api_key: str | None,
    workers: int = 3,
) -> dict[int, str]:
    """Markdown for several pages, transcribing the uncached ones concurrently."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda p: page_markdown(pdf_path, p, cache_dir, api_key), pages)
        return dict(zip(pages, results))


# ---------------------------------------------------------------------------
# Fidelity: does the transcription invent numbers?
# ---------------------------------------------------------------------------

_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _number_set(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(text)}


def unconfirmed_numbers(markdown: str, text_layer: str) -> list[str]:
    """
    Numbers in the vision Markdown that never appear in the PDF's own text layer.

    Vision output is model-generated text, so it is checked against the one
    thing the model cannot change: the characters actually stored in the PDF.
    Every printed number is in the text layer - just jumbled or misplaced - so
    a number that is NOT there was invented. Footnote reference numbers
    ([^n]) are ignored; they are formatting, not data.
    """
    data = re.sub(r"\[\^[\d,\s]+\]:?", " ", markdown)      # [^7], [^3,4]
    # In the text layer a footnote marker glued to a number reads as a
    # decimal ("by 2025.3" is the year 2025 plus marker 3), so its parts
    # count as present too.
    known = _number_set(text_layer)
    known |= {part for n in known for part in n.split(".")}
    # The text layer also splits numbers with a stray space ("3. 020",
    # "-7.8 44*" on the GEP regression tables). Each adjacent PAIR of
    # fragments is rejoined - never a whole run, which would glue separate
    # numbers together.
    known |= {a + b for a, b in re.findall(r"(\d[\d.]*) (?=(\d[\d.]*))", text_layer)}
    return sorted(_number_set(data) - known, key=lambda n: (len(n), n))
