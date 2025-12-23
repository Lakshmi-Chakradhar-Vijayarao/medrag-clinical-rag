# app/db/chroma_store.py

from __future__ import annotations

from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import get_active_encoder_name


class ChromaStore:
    """
    Thin wrapper around Chroma + SentenceTransformer embeddings.

    - Uses PersistentClient so the index survives restarts.
    - Embedding model is chosen via app.core.config.get_active_encoder_name().
    - Exposes:
        * add_chunks() with batched encoding (scale-friendly)
        * dense_search() for raw Chroma queries
        * similarity_search() for quick inspection (distance -> similarity)
    """

    def __init__(self, persist_dir: str = "./chroma_db", encoder_name: Optional[str] = None):
        # Persistent Chroma client
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("medrag_docs")

        count = self.collection.count()
        print(f"[ChromaStore] Initialized collection 'medrag_docs' with {count} docs")

        # Decide which encoder to use (config-driven, but can be overridden)
        if encoder_name is None:
            encoder_name = get_active_encoder_name()
        self.encoder_name = encoder_name

        print(f"[ChromaStore] Loading sentence encoder: {self.encoder_name}")
        self.encoder = SentenceTransformer(self.encoder_name)

        try:
            dim = self.encoder.get_sentence_embedding_dimension()
            print(f"[ChromaStore] Encoder '{self.encoder_name}' embedding dim = {dim}")
        except Exception:
            # Some models may not expose this, don't crash if so.
            print(f"[ChromaStore] Encoder '{self.encoder_name}' loaded (dimension unavailable)")

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64) -> None:
        """
        Add a list of chunk dicts to Chroma with freshly computed embeddings.
        Each chunk is expected to have: id, text, metadata.

        Embeddings are computed in batches to support larger corpora.
        """
        if not chunks:
            print("[ChromaStore] No chunks to add.")
            return

        ids = [c["id"] for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [c.get("metadata", {}) for c in chunks]

        print(f"[ChromaStore] Encoding {len(docs)} chunks for embeddings (batch_size={batch_size})...")

        all_embs: List[List[float]] = []
        for start in range(0, len(docs), batch_size):
            end = min(len(docs), start + batch_size)
            batch_docs = docs[start:end]
            batch_embs = self.encoder.encode(batch_docs, show_progress_bar=False)
            all_embs.extend(batch_embs.tolist())

        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=all_embs,
        )
        new_count = self.collection.count()
        print(f"[ChromaStore] Added {len(docs)} chunks. New count: {new_count}")

    def dense_search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dense semantic search over the Chroma collection.

        Args:
            query: user query string
            n_results: number of candidates to return
            where: optional Chroma metadata filter, e.g. {"disease_area": "diabetes"}

        Returns:
            Chroma query dict with keys: ids, documents, metadatas, distances.
        """
        q_emb = self.encoder.encode([query])
        return self.collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=n_results,
            where=where,
        )

    def similarity_search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience helper: wraps dense_search and converts distances to similarities.

        Returns:
            List of dicts:
                {
                  "id": str,
                  "text": str,
                  "metadata": dict,
                  "distance": float,
                  "similarity": float,  # 1 / (1 + distance)
                }
        """
        raw = self.dense_search(query, n_results=n_results, where=where)
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        out: List[Dict[str, Any]] = []
        for doc_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
            dist_f = float(dist)
            sim = 1.0 / (1.0 + dist_f)
            out.append(
                {
                    "id": doc_id,
                    "text": doc_text,
                    "metadata": meta or {},
                    "distance": dist_f,
                    "similarity": sim,
                }
            )
        return out
