"""
Compliance Copilot — the RAG pipeline.

Deliberately free of Streamlit imports so the whole pipeline can be exercised
from a plain script (see test_rag.py). app.py is only the UI layer.

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
    """
    text = text.replace("\xa0", " ")           # non-breaking space -> space
    text = re.sub(r"[ \t]{2,}", " ", text)      # runs of spaces/tabs -> one
    text = re.sub(r" *\n *", "\n", text)        # strip space around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)      # 3+ blank lines -> paragraph gap
    return text.strip()


def _parse_pdf(data: bytes) -> list[tuple[int, str]]:
    from pypdf import PdfReader
    from pypdf.errors import DependencyError, PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = list(enumerate(reader.pages, start=1))
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
    import docx2txt

    # docx2txt only accepts a path, not bytes, so the upload has to touch disk.
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return docx2txt.process(tmp_path) or ""
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
# 3. THE CHECKLIST (questions come from a preset, or from an uploaded file)
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
        (?:\(?\d+[.)]|\d+\.\d+|[-*•]|[a-zA-Z][.)])?   # optional bullet/number
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
# 4. RESULT TYPES
# ---------------------------------------------------------------------------

# Verdicts, worst-to-best. The whole point of the app is the NOT_FOUND row, so
# it is a first-class verdict rather than an error case.
COVERED = "COVERED"
PARTIAL = "PARTIAL"
NOT_FOUND = "NOT_FOUND"


@dataclass
class Citation:
    source: str
    page: int          # 0 means "document has no pages" (docx/txt)
    quote: str         # verbatim span from the chunk, verified to exist
    # Cosine DISTANCE, not a similarity score: LOWER means a closer match.
    # Chroma's method is named "..._with_relevance_scores" but returns raw
    # distances, so anything that sorts or thresholds on this must not assume
    # the usual higher-is-better convention.
    distance: float

    def label(self) -> str:
        return f"{self.source}, p.{self.page}" if self.page else self.source


@dataclass
class Finding:
    question: str
    verdict: str                              # COVERED | PARTIAL | NOT_FOUND
    answer: str
    citations: list[Citation] = field(default_factory=list)
    note: str = ""                            # why a verdict was downgraded

    @property
    def icon(self) -> str:
        return {COVERED: "✅", PARTIAL: "⚠️", NOT_FOUND: "❌"}[self.verdict]


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
    "gemini-2.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]
EMBED_PREFERENCE = [
    "gemini-embedding-2",
    "gemini-embedding-001",
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
    chat = next((m for m in CHAT_PREFERENCE if m in gen), None)
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
# 6. EMBEDDING + VECTOR STORAGE
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
        _with_retry(lambda: store.add_documents(docs))
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
# 7. RETRIEVAL
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
    store's document embedder, then matched by vector. Scores are LangChain
    relevance scores: higher is more similar.
    """
    # Retried here rather than only in run_audit, so the chat path and any
    # direct caller get the same protection.
    query_vec = _with_retry(
        lambda: make_embeddings(embed_model, TASK_QUERY, api_key).embed_query(question)
    )
    return store.similarity_search_by_vector_with_relevance_scores(query_vec, k=k)


# ---------------------------------------------------------------------------
# 8. GROUNDED GENERATION + THE QUOTE-VERIFICATION GUARDRAIL
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
- Do not infer an answer from a related topic. "We encrypt in transit" does \
not answer a question about encryption at rest.

Context:
{context}

Question: {question}
"""


def _normalise_for_match(text: str) -> str:
    """Fold away the differences a model harmlessly introduces when quoting."""
    return re.sub(r"\s+", " ", text).strip().lower()


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


def answer_question(
    llm,
    question: str,
    hits: list[tuple[Document, float]],
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
    are a request; this is an check.
    """
    if not hits:
        return Finding(
            question=question,
            verdict=NOT_FOUND,
            answer="",
            note="No chunks retrieved - is the knowledge base empty?",
        )

    context = "\n\n".join(
        f"[{i}] ({doc.metadata.get('source', '?')}"
        + (f", p.{doc.metadata['page']}" if doc.metadata.get("page") else "")
        + f")\n{doc.page_content}"
        for i, (doc, _) in enumerate(hits, start=1)
    )

    raw = get_text(llm.invoke(AUDIT_PROMPT.format(context=context, question=question)))

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
    answer = str(payload.get("answer", "")).strip()
    claimed = payload.get("quotes") or []
    if isinstance(claimed, str):
        claimed = [claimed]

    # --- verification: every claimed quote must exist in a retrieved chunk ---
    citations: list[Citation] = []
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
                    )
                )
                break
        else:
            unverified.append(quote)

    note = ""
    if verdict in (COVERED, PARTIAL) and not citations:
        note = (
            "Downgraded to NOT_FOUND: the model claimed coverage but no "
            "supporting quote could be found in the retrieved text"
            + (f" ({len(unverified)} quote(s) did not match)." if unverified else ".")
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
# 9. STRUCTURE-AWARE CHUNKING
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
_SECTION_HEADING = re.compile(
    r"^\s*(\d+)\.?\s+([A-Z][A-Z0-9 ,&/()'-]{3,})\s*$",
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
# 10. QUESTION EXTRACTION FROM AN UPLOADED QUESTIONNAIRE
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
# 11. BATCH AUDIT
# ---------------------------------------------------------------------------

# Free-tier Gemini keys are rate limited per minute. Four concurrent workers
# cuts a 12-question audit from ~48s to ~15s while staying well inside the
# limit; _with_retry absorbs the 429s that do slip through. Without the retry
# a long audit dies half-way and reports phantom gaps for every question that
# never ran - which would be indistinguishable from a genuine finding.
MAX_WORKERS = 4

_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "resourceexhausted",
                       "quota", "rate limit", "too many requests")


def _with_retry(fn, attempts: int = 4, base_delay: float = 2.0):
    """Retry a call through rate limiting with exponential backoff."""
    import time

    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            message = f"{type(exc).__name__} {exc}".lower()
            transient = any(m in message for m in _RATE_LIMIT_MARKERS)
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError("unreachable")


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
    outright becomes a NOT_FOUND finding carrying the error rather than
    vanishing - a missing row would read as "not asked" instead of "broken".

    on_result(index, finding) is called as each one lands, for UI progress.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[Finding | None] = [None] * len(questions)

    def one(index: int, question: str) -> tuple[int, Finding]:
        try:
            hits = _with_retry(
                lambda: retrieve(store, question, embed_model, k=k, api_key=api_key)
            )
            return index, _with_retry(lambda: answer_question(llm, question, hits))
        except Exception as exc:
            return index, Finding(
                question=question,
                verdict=NOT_FOUND,
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
# 12. EXPORT
# ---------------------------------------------------------------------------


def findings_to_csv(findings: list[Finding]) -> str:
    """Render findings as CSV - the format a real questionnaire gets returned in."""
    import csv
    import io as _io

    buf = _io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["#", "Question", "Verdict", "Answer", "Sources", "Evidence", "Note"])
    for i, f in enumerate(findings, start=1):
        writer.writerow([
            i,
            f.question,
            f.verdict,
            f.answer,
            " | ".join(c.label() for c in f.citations),
            " | ".join(c.quote for c in f.citations),
            f.note,
        ])
    return buf.getvalue()


def summarise(findings: list[Finding]) -> dict[str, int]:
    """Counts per verdict, plus the coverage score shown in the UI."""
    counts = {COVERED: 0, PARTIAL: 0, NOT_FOUND: 0}
    for f in findings:
        counts[f.verdict] = counts.get(f.verdict, 0) + 1
    counts["total"] = len(findings)
    return counts


# ---------------------------------------------------------------------------
# 13. CONVERSATIONAL MODE
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

    answer = get_text(
        llm.invoke(
            CHAT_PROMPT.format(
                history_block=history_block, context=context, question=question
            )
        )
    ).strip()

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
