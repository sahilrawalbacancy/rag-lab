"""
Score every pipeline variant on data/eval_set.json -> data/results/<doc>.json.

    .venv/bin/python evaluate.py                    # all three documents
    .venv/bin/python evaluate.py gep                # one document
    .venv/bin/python evaluate.py gpo --chunk-sizes 300,500,1000

Variants per document: naive (the initial chatbot), advanced (the document's
strategy), and each ablation (advanced minus one technique) - so every
technique's contribution is measured rather than asserted. The chunk-size
experiment indexes each size in memory, so it never touches data/chroma/.
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from rag import core, evaluation
from rag.assignments import CHROMA_DIR, RESULTS_DIR, ROOT, registry
from rag.retrieval import EMBED_MODEL, NAIVE, RERANK_MODEL, Index, build_collection, open_client


def make_llm(key: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    chat_model, _ = core.resolve_models(key)
    return chat_model, ChatGoogleGenerativeAI(model=chat_model, temperature=0, google_api_key=key)


def chunk_size_experiment(strategy, sizes: list[int], llm) -> dict:
    import chromadb

    client = chromadb.EphemeralClient()
    out = {}
    for size in sizes:
        name = f"{strategy.DOC_ID}__size{size}"
        build_collection(client, name, strategy.build(chunk_size=size))
        result = evaluation.run_variant(strategy.DOC_ID, Index.load(client, name), strategy.SETTINGS, llm)
        out[str(size)] = {"chunks": len(Index.load(client, name).docs), "summary": result["summary"],
                          "passed": [q["id"] for q in result["questions"] if q["passed"]]}
        print(f"  chunk size {size:5}: {result['summary']['passed']}/{result['summary']['total']}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docs", nargs="*", help="gpo, gep, imac (default: all)")
    parser.add_argument("--variants", help="comma-separated subset, e.g. naive,advanced")
    parser.add_argument("--chunk-sizes", help="comma-separated sizes for the chunk-size experiment")
    parser.add_argument("--model", help="pin the answer model (sets GEMINI_CHAT_MODEL)")
    parser.add_argument("--repeats", type=int, default=0,
                        help="re-run naive and advanced N more times to measure run-to-run spread")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if args.model:
        os.environ["GEMINI_CHAT_MODEL"] = args.model
    key = os.environ["GOOGLE_API_KEY"]
    chat_model, llm = make_llm(key)
    client = open_client(CHROMA_DIR)
    strategies = registry()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for doc_id in args.docs or list(strategies):
        strategy = strategies[doc_id]
        variants = {"naive": (f"{doc_id}__naive", NAIVE),
                    "advanced": (f"{doc_id}__advanced", strategy.SETTINGS)}
        variants.update({name: (f"{doc_id}__advanced", s) for name, s in strategy.ABLATIONS.items()})
        key_variants = {k: variants[k] for k in ("naive", "advanced")}
        if args.variants:
            variants = {k: v for k, v in variants.items() if k in args.variants.split(",")}

        path = RESULTS_DIR / f"{doc_id}.json"
        results = json.loads(path.read_text()) if path.exists() else {}
        results.update({"doc": doc_id, "title": strategy.TITLE, "chat_model": chat_model,
                        "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL})
        results.setdefault("variants", {})
        print(f"\n== {strategy.TITLE}")
        for name, (collection, settings) in variants.items():
            result = evaluation.run_variant(doc_id, Index.load(client, collection), settings, llm,
                                            strategy.route)
            results["variants"][name] = result
            s = result["summary"]
            print(f"  {name:18} {s['passed']}/{s['total']} passed   "
                  f"retrieved {s['retrieved_expected']}/{s['retrieval_scored']}   "
                  f"grounded {s['grounded']}   errors {s['errors']}   {s['avg_seconds']}s/q")
            path.write_text(json.dumps(results, indent=1) + "\n")
        if args.chunk_sizes:
            sizes = [int(s) for s in args.chunk_sizes.split(",")]
            results.setdefault("experiments", {})["chunk_size"] = chunk_size_experiment(strategy, sizes, llm)
            path.write_text(json.dumps(results, indent=1) + "\n")
        if args.repeats:
            # Eight to ten questions and an answer model that is not fully
            # deterministic: one run can flip by a question either way. The
            # saved run plus N repeats gives the spread a fair comparison needs.
            stability = {}
            for name, (collection, settings) in key_variants.items():
                runs = [results["variants"][name]] if name in results["variants"] else []
                for _ in range(args.repeats):
                    runs.append(evaluation.run_variant(doc_id, Index.load(client, collection), settings,
                                                       llm, strategy.route))
                stability[name] = {
                    "passed": [r["summary"]["passed"] for r in runs],
                    "total": runs[0]["summary"]["total"],
                    "errors": sum(r["summary"]["errors"] for r in runs),
                    "per_question": {q["id"]: sum(x["passed"] for r in runs for x in r["questions"]
                                                  if x["id"] == q["id"]) for q in runs[0]["questions"]},
                }
                print(f"  {name:18} across {len(runs)} runs: {stability[name]['passed']}")
            results["stability"] = stability
            path.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
