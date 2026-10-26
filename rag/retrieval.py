"""
The retrieval pipeline shared by the three assignments: everything between
"a question arrives" and "the model writes an answer".

    question
      -> route    (optional) restrict the search to the section the question names
      -> dense    local bi-encoder; nearest chunks in Chroma
      -> sparse   BM25 keyword scores over the same chunks
      -> fuse     reciprocal rank fusion of the two rankings
      -> rerank   local cross-encoder scores each (question, chunk) pair
      -> expand   neighbouring chunks, the section's opening, linked footnotes
      -> answer   core.answer_question - quotes verified, numbers checked

Every step is switched by a Settings flag, so the naive baseline and every
ablation in the evaluation run through this same code - a difference in the
scores is a difference in one setting, nothing else.

Everything here runs inside the Streamlit process: embeddings and reranking
are small ONNX models on CPU (no API quota, no GPU) and Chroma is embedded.
Only the final answer calls Gemini.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from rag import core

# ---------------------------------------------------------------------------
# 1. LOCAL MODELS
# ---------------------------------------------------------------------------

# Why local: the Gemini free tier counts every text in an embedding batch as
# one request - 100 a minute, ~1,000 a day. The three documents need ~2,400
# chunk embeddings, which would take days. Small local models make indexing
# free and repeatable, and keep the app inside Streamlit Cloud's memory:
# both models together add ~415 MB (measured).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"           # 384 dims, 67 MB
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-12-v2"  # cross-encoder, 120 MB

_models: dict = {}
_lock = threading.Lock()


def _model(kind: str):
    """Load each model once per process, on first use."""
    with _lock:
        if kind not in _models:
            if kind == "embed":
                from fastembed import TextEmbedding

                _models[kind] = TextEmbedding(EMBED_MODEL)
            else:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                _models[kind] = TextCrossEncoder(RERANK_MODEL)
        return _models[kind]


# Small batches on purpose: fastembed's default batch of 256 long chunks
# allocates attention buffers for all of them at once - 5.4 GB measured.
def embed_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model("embed").passage_embed(texts, batch_size=16)]


def embed_query(text: str) -> list[float]:
    # query_embed and passage_embed are the local equivalent of Gemini's
    # RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT task types: a question and the
    # passage answering it are not phrased alike.
    return next(_model("embed").query_embed(text)).tolist()


def rerank_scores(question: str, texts: list[str]) -> list[float]:
    return [float(s) for s in _model("rerank").rerank(question, texts, batch_size=4)]


# ---------------------------------------------------------------------------
# 2. KEYWORD SEARCH (BM25)
# ---------------------------------------------------------------------------

# Numbers keep their decimal point and drop thousands separators, so "8.9"
# and "15,901,600" are single searchable terms.
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOP = frozenset(
    "a an and are as at be by can did do does for from had has have how in is "
    "it its of on or so than that the their this to was were what when where "
    "which who will with".split()
)


def _terms(text: str) -> list[str]:
    return [t.replace(",", "") for t in _WORD.findall(text.lower()) if t not in _STOP]


class BM25:
    """
    Okapi BM25 - twenty lines, so no dependency.

    Dense vectors match meaning; BM25 matches the exact tokens a dense model
    blurs: "Iraq", "ENERGY STAR", "kg CO2e", "2026f".
    """

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(_terms(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = set(_terms(query))
        out = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
            out.append(sum(
                self.idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + norm)
                for t in terms if t in doc
            ))
        return out


# ---------------------------------------------------------------------------
# 3. THE INDEX (Chroma, embedded)
# ---------------------------------------------------------------------------


def finalise(docs: list[Document], doc_id: str) -> list[Document]:
    """
    Number chunks in reading order and fill the metadata every chunk carries.

    `order` is what neighbour expansion walks; Chroma accepts only scalar
    metadata, so missing fields become "" rather than None.
    """
    for i, d in enumerate(docs):
        meta = d.metadata
        meta.pop("chunk_index", None)
        for key in ("section", "chapter", "printed_page", "footnotes", "footnote"):
            meta.setdefault(key, "")
        meta.setdefault("kind", "text")
        meta.setdefault("origin", "text layer")
        meta["order"] = i
        meta["doc"] = doc_id
    return docs


def naive_chunks(pdf_path: str | Path, source: str, doc_id: str) -> list[Document]:
    """The initial chatbot's pipeline: text layer, fixed 1,000-character chunks."""
    pages = core.parse_document(Path(pdf_path).read_bytes(), Path(pdf_path).name)
    return finalise(core.chunk_pages(pages, source), doc_id)


def build_collection(client, name: str, docs: list[Document], description: str = ""):
    """Embed `docs` locally and store them as a fresh cosine-space collection."""
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(
        name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embed_model": EMBED_MODEL, "chunks": len(docs), "description": description},
        embedding_function=None,
    )
    vectors = embed_passages([d.page_content for d in docs])
    for i in range(0, len(docs), 500):
        batch = docs[i : i + 500]
        collection.add(
            ids=[f"{name}-{d.metadata['order']}" for d in batch],
            embeddings=vectors[i : i + 500],
            documents=[d.page_content for d in batch],
            metadatas=[d.metadata for d in batch],
        )
    return collection


@dataclass
class Index:
    """One collection plus the in-memory structures retrieval needs."""

    name: str
    collection: object
    docs: list[Document]                       # docs[i].metadata["order"] == i
    bm25: BM25
    footnotes: dict[str, Document] = field(default_factory=dict)

    @classmethod
    def load(cls, client, name: str) -> "Index":
        collection = client.get_collection(name, embedding_function=None)
        data = collection.get(include=["documents", "metadatas"])
        docs = sorted(
            (Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])),
            key=lambda d: d.metadata["order"],
        )
        footnotes = {d.metadata["footnote"]: d for d in docs if d.metadata.get("footnote")}
        return cls(name, collection, docs, BM25([d.page_content for d in docs]), footnotes)


def open_client(path: str | Path, copy: bool = True):
    """
    Open the committed database. By default a temporary COPY is opened:
    Chroma writes to its files even when only reading, and the copy keeps
    data/chroma/ byte-identical in git while the app or the tests run.
    """
    import shutil
    import tempfile

    import chromadb

    path = Path(path)
    if copy:
        target = Path(tempfile.mkdtemp(prefix="rag-lab-chroma-")) / "chroma"
        shutil.copytree(path, target)
        path = target
    return chromadb.PersistentClient(path=str(path))


# ---------------------------------------------------------------------------
# 4. SEARCH
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    k: int = 5                # chunks kept after ranking
    candidates: int = 20      # pulled from each retriever before fusion / rerank
    hybrid: bool = True       # add BM25 to the dense ranking
    rerank: bool = True       # cross-encoder over the fused candidates
    window: int = 0           # neighbouring chunks added each side of a hit
    route: bool = False       # restrict to a section the question names
    lead: bool = False        # add the routed section's opening chunk
    footnotes: bool = False   # add the footnotes a retrieved chunk cites
    max_context: int = 10     # hard cap on passages handed to the model

    def describe(self) -> str:
        parts = ["dense + BM25" if self.hybrid else "dense only"]
        if self.rerank:
            parts.append("rerank")
        if self.window:
            parts.append(f"±{self.window} neighbours")
        if self.route:
            parts.append("section routing")
        if self.lead:
            parts.append("section opening")
        if self.footnotes:
            parts.append("linked footnotes")
        return f"top {self.k}, " + ", ".join(parts)


# The initial chatbot: nearest five chunks by vector, nothing else.
NAIVE = Settings(hybrid=False, rerank=False)


@dataclass
class Hit:
    doc: Document
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused: float = 0.0
    rerank: float | None = None
    why: str = "retrieved"     # retrieved | neighbour | section opening | footnote n

    @property
    def order(self) -> int:
        return self.doc.metadata["order"]


def _matches(meta: dict, where: dict | None) -> bool:
    for key, value in (where or {}).items():
        allowed = value if isinstance(value, list) else [value]
        if meta.get(key) not in allowed:
            return False
    return True


def _chroma_where(where: dict | None) -> dict | None:
    if not where:
        return None
    clauses = [{k: {"$in": v} if isinstance(v, list) else v} for k, v in where.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search(index: Index, question: str, settings: Settings, where: dict | None = None) -> list[Hit]:
    """Ranked hits for one question - dense, optionally fused with BM25 and reranked."""
    allowed = [d.metadata["order"] for d in index.docs if _matches(d.metadata, where)]
    if not allowed:            # the route matched nothing: never search an empty set
        allowed, where = [d.metadata["order"] for d in index.docs], None
    n = min(settings.candidates, len(allowed))

    found = index.collection.query(
        query_embeddings=[embed_query(question)], n_results=n,
        where=_chroma_where(where), include=["metadatas"],
    )
    hits: dict[int, Hit] = {}
    for rank, meta in enumerate(found["metadatas"][0], start=1):
        hits[meta["order"]] = Hit(index.docs[meta["order"]], dense_rank=rank)

    if settings.hybrid:
        scores = index.bm25.scores(question)
        top = sorted((o for o in allowed if scores[o] > 0), key=lambda o: -scores[o])[:n]
        for rank, order in enumerate(top, start=1):
            hits.setdefault(order, Hit(index.docs[order])).sparse_rank = rank

    # Reciprocal rank fusion: a chunk near the top of EITHER list rises; the
    # constant 60 is the standard damping that stops rank 1 dominating.
    for hit in hits.values():
        hit.fused = sum(1 / (60 + r) for r in (hit.dense_rank, hit.sparse_rank) if r)
    ranked = sorted(hits.values(), key=lambda h: -h.fused)[: settings.candidates]

    if settings.rerank and ranked:
        for hit, score in zip(ranked, rerank_scores(question, [h.doc.page_content for h in ranked])):
            hit.rerank = score
        ranked.sort(key=lambda h: -h.rerank)
    return ranked[: settings.k]


def expand(index: Index, hits: list[Hit], settings: Settings, where: dict | None = None) -> list[Hit]:
    """
    Turn ranked hits into the passages the model reads.

    - section opening: a report's sections usually open with their own
      summary (Chapter 3's first page is an abstract), so a question routed
      to a section always sees it;
    - neighbours: the chunk either side on the same page, so an answer cut by
      a chunk boundary arrives whole;
    - footnotes: the endnotes a chunk cites ("58 percent less energy[^18]")
      are pages away from the claim - linked, they arrive together.
    """
    out: list[Hit] = []
    seen: set[int] = set()

    def add(hit: Hit) -> None:
        if hit.order not in seen and len(out) < settings.max_context:
            seen.add(hit.order)
            out.append(hit)

    if settings.lead and where:
        opening = next(
            (d for d in index.docs if _matches(d.metadata, where) and len(d.page_content) > 200),
            None,
        )
        if opening is not None:
            add(Hit(opening, why="section opening"))

    for hit in hits:
        group = [hit]
        for step in range(1, settings.window + 1):
            for order in (hit.order - step, hit.order + step):
                if 0 <= order < len(index.docs):
                    d = index.docs[order]
                    if (d.metadata["page"], d.metadata["section"]) == (
                        hit.doc.metadata["page"], hit.doc.metadata["section"]
                    ):
                        group.append(Hit(d, why="neighbour"))
        for member in sorted(group, key=lambda h: h.order):   # reading order
            add(member)
        if settings.footnotes:
            for number in filter(None, hit.doc.metadata.get("footnotes", "").split(",")):
                note = index.footnotes.get(number)
                if note is not None:
                    add(Hit(note, why=f"footnote {number}"))
    return _merge_runs(out)


def _join(a: str, b: str) -> str:
    """Concatenate two consecutive chunks, dropping the overlap they share."""
    for k in range(min(len(a), len(b), 600), 15, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return f"{a}\n{b}"


def _merge_runs(hits: list[Hit]) -> list[Hit]:
    """
    Fuse consecutive chunks of the same page and section into one passage.

    Neighbours handed over as separate, overlapping fragments let the model
    quote a sentence that runs across a chunk boundary - a genuine quote the
    guardrail then rejects, because it checks each passage on its own. One
    continuous passage is also what "sentence window" retrieval means.
    """
    runs: list[list[Hit]] = []
    for hit in hits:
        last = runs[-1][-1] if runs else None
        if last is not None and hit.order == last.order + 1 and (
            hit.doc.metadata["page"], hit.doc.metadata["section"]
        ) == (last.doc.metadata["page"], last.doc.metadata["section"]):
            runs[-1].append(hit)
        else:
            runs.append([hit])

    out = []
    for run in runs:
        if len(run) == 1:
            out.append(run[0])
            continue
        text = run[0].doc.page_content
        for hit in run[1:]:
            text = _join(text, hit.doc.page_content)
        lead = next((h for h in run if h.why == "retrieved"), run[0])
        merged = Hit(Document(page_content=text, metadata=dict(run[0].doc.metadata)),
                     lead.dense_rank, lead.sparse_rank, lead.fused, lead.rerank,
                     "retrieved + neighbours" if lead.why == "retrieved" else lead.why)
        out.append(merged)
    return out


# ---------------------------------------------------------------------------
# 5. ANSWER
# ---------------------------------------------------------------------------

QA_PROMPT = """You answer questions about one document using ONLY the \
numbered passages below. You have no other knowledge. If the passages do not \
answer the question, say so - that is a valid and useful result.

Reply with ONLY a JSON object, no prose and no code fence:
{{
  "verdict": "COVERED" | "PARTIAL" | "NOT_FOUND",
  "answer": "<at most 4 sentences; give every figure with its unit and the period it refers to. Empty string if NOT_FOUND>",
  "quotes": ["<a span copied character-for-character from one passage>"]
}}

Rules:
- COVERED: the passages fully answer the question. PARTIAL: they answer part \
of it - say which part is missing. NOT_FOUND: they do not address it.
- Each passage is labelled with its page and, where known, its section. When \
the question names a section, chapter, region, product configuration or year, \
use only passages that match it: a figure for a different region or \
configuration is a wrong answer, not a partial one.
- Every quote MUST be copied character-for-character from one passage - a \
sentence, or a table row exactly as it appears. Never paraphrase, merge or \
reformat a quote.
- COVERED or PARTIAL requires at least one quote. Every number in your answer \
must appear inside one of your quotes: quote each sentence or table row you \
take a figure from, and when a figure comes from a table also quote the \
header row that names its column (a product configuration, a year). Report \
figures as printed; do not calculate new ones.
- Do not infer an answer from a related topic.

Passages:
{context}

Question: {question}
"""


@dataclass
class Answer:
    finding: core.Finding
    context: list[Hit]
    route: str = ""          # the section the search was restricted to, if any
    seconds: float = 0.0


def ask(index: Index, question: str, llm, settings: Settings, route_fn=None) -> Answer:
    """Run the whole pipeline for one question."""
    started = time.time()
    where, route = (route_fn(question) if (settings.route and route_fn) else (None, ""))
    context = expand(index, search(index, question, settings, where), settings, where)
    finding = core.with_retry(
        lambda: core.answer_question(
            llm, question, [(h.doc, 0.0) for h in context], prompt=QA_PROMPT, repair=True
        ),
        attempts=6, base_delay=5.0,
    )
    return Answer(finding, context, route, round(time.time() - started, 2))
