"""Where everything lives, and the three assignments by id."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = DATA / "docs"
CHROMA_DIR = DATA / "chroma"       # built by ingest.py, read by the app
VISION_DIR = DATA / "vision"       # cached Gemini page transcriptions
RESULTS_DIR = DATA / "results"     # written by evaluate.py
EVAL_SET = DATA / "eval_set.json"


def registry() -> dict:
    """{doc_id: strategy module}. Imported lazily: each pulls in its parser."""
    from rag import narrative, structured, visual

    return {m.DOC_ID: m for m in (narrative, structured, visual)}
