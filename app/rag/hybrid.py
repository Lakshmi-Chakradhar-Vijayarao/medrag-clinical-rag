# app/rag/hybrid.py
from typing import List, Dict, Any


def hybrid_fuse(dense_results: Dict[str, Any],
                bm25_results: List[Dict[str, Any]],
                alpha: float = 0.6) -> List[Dict[str, Any]]:
    """
    Fuse dense (Chroma) and sparse (BM25) scores.

    Args:
        dense_results: Chroma query result dict
        bm25_results: list of {id, text, score, metadata}
        alpha: weight for dense score (0–1). Higher = trust dense more.

    Returns:
        List of dicts with: id, text, metadata, dense_score, bm25_score, score
    """
    combined: Dict[str, Dict[str, Any]] = {}

    # Handle case where Chroma has no hits
    dense_ids_list = dense_results.get("ids", [])
    if dense_ids_list and dense_ids_list[0]:
        dense_ids = dense_results["ids"][0]
        dense_docs = dense_results["documents"][0]
        dense_metas = dense_results["metadatas"][0]
        dense_dists = dense_results["distances"][0]

        # Convert distance → similarity
        for d_id, d_doc, d_meta, d_dist in zip(dense_ids, dense_docs, dense_metas, dense_dists):
            sim = 1.0 / (1.0 + d_dist)
            combined[d_id] = {
                "id": d_id,
                "text": d_doc,
                "metadata": d_meta or {},
                "dense_score": float(sim),
                "bm25_score": 0.0,
            }

    # Normalize BM25 scores
    max_bm25 = max((r["score"] for r in bm25_results), default=1.0) or 1.0

    for item in bm25_results:
        doc_id = item["id"]
        bm25_norm = item["score"] / max_bm25
        meta = item.get("metadata", {}) or {}
        if doc_id in combined:
            combined[doc_id]["bm25_score"] = float(bm25_norm)
            # If dense metadata was missing 'source', prefer non-empty BM25 meta
            if not combined[doc_id].get("metadata"):
                combined[doc_id]["metadata"] = meta
        else:
            combined[doc_id] = {
                "id": doc_id,
                "text": item["text"],
                "metadata": meta,
                "dense_score": 0.0,
                "bm25_score": float(bm25_norm),
            }

    fused: List[Dict[str, Any]] = []
    for c in combined.values():
        score = alpha * c["dense_score"] + (1 - alpha) * c["bm25_score"]
        c["score"] = float(score)
        fused.append(c)

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused
