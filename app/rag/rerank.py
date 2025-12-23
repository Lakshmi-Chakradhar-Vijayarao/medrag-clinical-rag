# app/rag/rerank.py
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Optional cross-encoder reranker (e.g., MS MARCO MiniLM).

    Given a query and candidate docs, re-scores them with a CrossEncoder and
    returns them sorted by that score. This is slower, so we use it only
    in eval / "research" mode.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[CrossEncoderReranker] Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return candidates

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
