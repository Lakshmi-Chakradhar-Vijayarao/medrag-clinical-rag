# scripts/eval_retrieval.py

import sys
import json
from pathlib import Path
from collections import defaultdict

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from app.core.config import RAG_CONFIG, get_active_encoder_name, get_hybrid_alpha
from app.db.bm25_index import BM25Index
from app.db.chroma_store import ChromaStore
from app.rag.retrieval import HybridRetriever
from app.rag.rerank import CrossEncoderReranker

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def load_chunks():
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def run_eval_for_encoder(label: str, use_biomed: bool, chunks):
    """
    Run retrieval evaluation for a specific encoder config.

    Args:
        label: human-readable label (e.g. "MiniLM (general)", "BioClinical (biomed)").
        use_biomed: True -> use biomedical encoder from config, False -> main encoder.
        chunks: preloaded list of chunk dicts.
    """
    # Flip encoder choice in global config
    RAG_CONFIG.setdefault("retrieval", {})
    RAG_CONFIG["retrieval"]["use_biomedical_encoder"] = use_biomed

    encoder_name = get_active_encoder_name()
    print(f"\n[eval_retrieval] ===== Encoder: {label} ({encoder_name}) =====")

    bm25 = BM25Index(chunks)
    print(f"[BM25Index] Initialized over {len(chunks)} chunks")

    chroma = ChromaStore(str(PROJECT_ROOT / "chroma_db"))

    alpha = get_hybrid_alpha()
    print(f"[eval_retrieval] Initializing retrievers (hybrid_alpha={alpha})...")

    # Plain hybrid (no rerank) and hybrid + cross-encoder rerank
    hybrid_plain = HybridRetriever(chroma, bm25, alpha=alpha, use_rerank=False)
    reranker = CrossEncoderReranker()
    hybrid_rerank = HybridRetriever(
        chroma,
        bm25,
        alpha=alpha,
        use_rerank=True,
        reranker=reranker,
    )

    # Gold map: query -> expected source
    eval_queries = [
        {
            "query": "What is the first-line pharmacologic treatment for Type 2 diabetes?",
            "gold_source": "diabetes_overview",
        },
        {
            "query": "What are the first-line drug classes for treating hypertension?",
            "gold_source": "hypertension_guideline",
        },
    ]

    k_values = [1, 3, 5]
    stats = {
        "bm25": defaultdict(int),
        "dense": defaultdict(int),
        "hybrid": defaultdict(int),
        "hybrid_rerank": defaultdict(int),
    }

    for item in eval_queries:
        q = item["query"]
        gold_source = item["gold_source"]

        print(f"\n[eval_retrieval] Query: {q}")
        print(f"[eval_retrieval] Gold source: {gold_source}")

        max_k = max(k_values)

        # BM25 only
        bm25_results = bm25.search(q, top_k=max_k)
        # Dense only
        dense_results = chroma.dense_search(q, n_results=max_k)
        # Hybrid
        hybrid_results = hybrid_plain.retrieve(q, top_k=max_k)
        # Hybrid + rerank
        hybrid_rr_results = hybrid_rerank.retrieve(q, top_k=max_k)

        def sources_from_bm25(results):
            return [r.get("metadata", {}).get("source", "") for r in results]

        def sources_from_dense(dense):
            metas = dense.get("metadatas", [[]])[0] if dense.get("metadatas") else []
            return [m.get("source", "") for m in metas]

        def sources_from_hybrid(results):
            return [r.get("metadata", {}).get("source", "") for r in results]

        bm25_sources = sources_from_bm25(bm25_results)
        dense_sources = sources_from_dense(dense_results)
        hybrid_sources = sources_from_hybrid(hybrid_results)
        hybrid_rr_sources = sources_from_hybrid(hybrid_rr_results)

        for k in k_values:
            if gold_source in bm25_sources[:k]:
                stats["bm25"][k] += 1
            if gold_source in dense_sources[:k]:
                stats["dense"][k] += 1
            if gold_source in hybrid_sources[:k]:
                stats["hybrid"][k] += 1
            if gold_source in hybrid_rr_sources[:k]:
                stats["hybrid_rerank"][k] += 1

    print("\n=== Retrieval Recall@k (Encoder: {} / {}) ===".format(label, encoder_name))
    total = len(eval_queries)
    for name in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
        print(f"\n{name.upper()}:")
        for k in k_values:
            hits = stats[name][k]
            recall = hits / total if total > 0 else 0.0
            print(f"  Recall@{k}: {hits}/{total} = {recall:.2f}")


def main():
    print("[eval_retrieval] Loading chunks...")
    chunks = load_chunks()
    print(f"[eval_retrieval] Loaded {len(chunks)} chunks")

    # 1) Evaluate with main general-purpose encoder
    run_eval_for_encoder(label="MiniLM (general)", use_biomed=False, chunks=chunks)

    # 2) (Optional) Evaluate with biomedical encoder
    #    If the biomedical model is not available locally, this block may fail;
    #    you can comment it out if needed.
    try:
        run_eval_for_encoder(label="BioClinical (biomed)", use_biomed=True, chunks=chunks)
    except Exception as e:
        print("\n[eval_retrieval] Skipping biomedical encoder eval due to error:")
        print(f"  {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
