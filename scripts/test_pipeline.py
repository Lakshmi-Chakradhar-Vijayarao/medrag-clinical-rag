# scripts/test_pipeline.py

import sys
import json
from pathlib import Path

# Ensure project root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from app.db.chroma_store import ChromaStore
from app.db.bm25_index import BM25Index
from app.rag.retrieval import HybridRetriever
from app.llm.client import LocalLLMClient
from app.rag.pipeline import MedRAGPipeline

CHUNKS_PATH = PROJECT_ROOT / "data/processed/chunks.jsonl"


def load_chunks():
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    print("[test_pipeline] Loading chunks...")
    chunks = load_chunks()
    print(f"[test_pipeline] Loaded {len(chunks)} chunks")

    print("[test_pipeline] Initializing Chroma...")
    chroma = ChromaStore(str(PROJECT_ROOT / "chroma_db"))

    print("[test_pipeline] Initializing BM25...")
    bm25 = BM25Index(chunks)

    retriever = HybridRetriever(chroma, bm25, alpha=0.6)

    print("[test_pipeline] Loading LLM...")
    llm = LocalLLMClient()

    pipeline = MedRAGPipeline(retriever, llm)

    query = "What is the first-line pharmacologic treatment for Type 2 diabetes?"
    print(f"\n[test_pipeline] Running query: {query}\n")

    result = pipeline.run(query)

    print("=== ANSWER ===")
    print(result["answer"])
    print("\n=== SUPPORTED? ===")
    print(result["is_supported"])
    print("\n=== ISSUES ===")
    for issue in result["issues"]:
        print(f"- {issue}")

    print("\n=== EVIDENCE IDS ===")
    for i, c in enumerate(result["evidence"]):
        print(f"[{i}] id={c.get('id')} source={c.get('metadata', {}).get('source', '')}")


if __name__ == "__main__":
    main()
