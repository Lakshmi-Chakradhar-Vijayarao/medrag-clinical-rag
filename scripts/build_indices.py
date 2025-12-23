# scripts/build_indices.py
import sys
import json
from pathlib import Path

# Resolve project root (the folder that contains "app", "data", etc.)
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from app.db.chroma_store import ChromaStore  # now Python can find "app"

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
    chunks = load_chunks()
    print(f"[build_indices] Loaded {len(chunks)} chunks")

    chroma = ChromaStore("./chroma_db")
    chroma.add_chunks(chunks)
    print("[build_indices] Chroma index built and persisted")

if __name__ == "__main__":
    main()
