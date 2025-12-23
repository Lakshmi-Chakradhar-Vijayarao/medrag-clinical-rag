# app/rag/retrieval.py
from typing import Optional, List, Dict, Any

from app.rag.hybrid import hybrid_fuse


class HybridRetriever:
    """
    Hybrid retriever over:
      - Dense semantic search (Chroma)
      - Sparse lexical search (BM25)

    Optional:
      - Cross-encoder reranking (research / eval mode).
    """

    def __init__(
        self,
        chroma_store,
        bm25_index,
        alpha: float = 0.6,
        use_rerank: bool = False,
        reranker=None,
    ):
        """
        Args:
            chroma_store: ChromaStore instance
            bm25_index: BM25Index instance
            alpha: dense vs bm25 fusion weight
            use_rerank: whether to apply cross-encoder reranking
            reranker: CrossEncoderReranker instance (optional)
        """
        self.chroma = chroma_store
        self.bm25 = bm25_index
        self.alpha = alpha
        self.use_rerank = use_rerank
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        disease_area: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k evidence chunks.

        Args:
            query: user question
            top_k: number of chunks to return
            disease_area: optional metadata filter (e.g. "diabetes")

        Returns:
            List of chunk dicts sorted by relevance.
        """
        # If reranking is enabled, grab more candidates first
        base_k = top_k * 2 if (self.use_rerank and self.reranker) else top_k
        where = {"disease_area": disease_area} if disease_area else None

        dense = self.chroma.dense_search(query, n_results=base_k, where=where)
        sparse = self.bm25.search(query, top_k=base_k)

        fused = hybrid_fuse(dense, sparse, alpha=self.alpha)

        if self.use_rerank and self.reranker and fused:
            fused = self.reranker.rerank(query, fused)

        return fused[:top_k]
