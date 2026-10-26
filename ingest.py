"""
Build the vector database the assignment pages read: data/chroma/.

    .venv/bin/python ingest.py

Two collections per document:
    <doc>__naive      the initial chatbot's pipeline: text layer, fixed
                      1,000-character chunks, nothing else
    <doc>__advanced   that document's strategy (rag/narrative.py,
                      rag/structured.py, rag/visual.py)

Embedding runs locally, so indexing costs no API quota. Vision pages are read
from the cache in data/vision/; only an uncached page calls Gemini. The
database is built in a temporary folder and swapped in only once complete,
so a failed run never leaves a half-built index behind.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from rag.assignments import CHROMA_DIR, ROOT, registry
from rag.retrieval import Index, build_collection, naive_chunks


def main() -> None:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GOOGLE_API_KEY")
    staging = Path(tempfile.mkdtemp(prefix="rag-lab-ingest-")) / "chroma"
    client = chromadb.PersistentClient(path=str(staging))

    for doc_id, strategy in registry().items():
        for variant, build, description in (
            ("naive", lambda: naive_chunks(strategy.PDF, strategy.SOURCE, doc_id),
             "text layer, fixed 1,000-character chunks"),
            ("advanced", lambda: strategy.build(key), strategy.SETTINGS.describe()),
        ):
            started = time.time()
            docs = build()
            build_collection(client, f"{doc_id}__{variant}", docs, description)
            kinds = sorted({d.metadata["kind"] for d in docs})
            print(f"{doc_id:5} {variant:9} {len(docs):5} chunks  {', '.join(kinds):30} "
                  f"{time.time() - started:6.1f}s")

    # Flush and close Chroma before copying its files.
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    shutil.copytree(staging, CHROMA_DIR)
    shutil.rmtree(staging.parent, ignore_errors=True)

    check = chromadb.PersistentClient(path=str(CHROMA_DIR))
    for collection in check.list_collections():
        assert Index.load(check, collection.name).docs, f"{collection.name} is empty"
    size = sum(f.stat().st_size for f in CHROMA_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"\nwrote {CHROMA_DIR.relative_to(ROOT)}/ - {len(check.list_collections())} collections, {size:.1f} MB")


if __name__ == "__main__":
    main()
