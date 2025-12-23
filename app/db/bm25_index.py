# app/db/bm25_index.py
from rank_bm25 import BM25Okapi


class BM25Index:
    """
    Simple BM25 wrapper over the ingested chunks.

    Stores:
      - texts
      - ids
      - metadata (so we can recover 'source' during eval and fusion).
    """

    def __init__(self, chunks):
        self.texts = [c["text"] for c in chunks]
        self.ids = [c["id"] for c in chunks]
        self.metas = [c.get("metadata", {}) for c in chunks]

        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        print(f"[BM25Index] Initialized over {len(self.texts)} chunks")

    def search(self, query: str, top_k: int = 10):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.ids, self.texts, self.metas, scores),
            key=lambda x: x[3],
            reverse=True,
        )[:top_k]

        return [
            {
                "id": id_,
                "text": text,
                "score": float(score),
                "metadata": meta or {},
            }
            for id_, text, meta, score in ranked
        ]
