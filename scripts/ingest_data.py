# scripts/ingest_data.py

import json
import re
from pathlib import Path
from typing import List, Dict, Any

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# Optional mapping from filename stem -> disease_area label
DISEASE_AREA_MAP = {
    "diabetes_overview": "diabetes",
    "hypertension_guideline": "hypertension",
    "heart_failure_overview": "heart_failure",
    # add more here as you grow your library
}


def read_raw_files() -> List[Path]:
    """
    Return a sorted list of all .txt files in data/raw.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"[ingest_data] RAW_DIR not found: {RAW_DIR}")

    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"[ingest_data] No .txt files in {RAW_DIR}")
    print(f"[ingest_data] Found {len(files)} raw guideline files in {RAW_DIR}")
    return files


def split_into_sections(text: str) -> List[str]:
    """
    Split a guideline into 'sections' using double newlines / blank lines.

    This is simple but robust for most plain-text guidelines.
    """
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split on blank lines
    raw_sections = re.split(r"\n\s*\n", text)
    sections = [s.strip() for s in raw_sections if s.strip()]
    return sections


def split_section_into_chunks(section_text: str, max_chars: int = 600) -> List[str]:
    """
    Split a section into sentence-based chunks of ~max_chars characters.

    This is char-based for simplicity but respects sentence boundaries.
    """
    # Rough sentence split
    sentences = re.split(r"(?<=[.!?])\s+", section_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [section_text.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) + 1 > max_chars and current:
            chunks.append(" ".join(current).strip())
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def make_section_title(section_text: str, max_len: int = 120) -> str:
    """
    Create a human-friendly section title from the first line or first sentence.
    """
    # Prefer first line
    first_line = section_text.strip().split("\n", 1)[0].strip()
    if not first_line:
        first_line = section_text.strip()

    if len(first_line) > max_len:
        return first_line[: max_len - 3] + "..."
    return first_line


def ingest_file(path: Path, global_start_index: int) -> List[Dict[str, Any]]:
    """
    Ingest a single guideline file into chunk dicts.

    Args:
        path: Path to raw .txt file
        global_start_index: starting global index for chunks from this file

    Returns:
        list of chunk dicts, each with id / text / source / metadata
    """
    print(f"[ingest_data] Ingesting {path.name}...")
    text = path.read_text(encoding="utf-8", errors="ignore")
    sections = split_into_sections(text)
    print(f"[ingest_data]  -> {len(sections)} sections detected in {path.name}")

    source = path.stem  # e.g., "diabetes_overview"
    disease_area = DISEASE_AREA_MAP.get(source, source)

    chunks: List[Dict[str, Any]] = []
    global_index = global_start_index

    for sec_idx, sec_text in enumerate(sections):
        section_title = make_section_title(sec_text)
        sec_chunks = split_section_into_chunks(sec_text, max_chars=600)

        for chunk_idx, chunk_text in enumerate(sec_chunks):
            chunk_id = f"{source}_sec{sec_idx}_chunk{chunk_idx}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "source": source,
                    "metadata": {
                        "global_index": global_index,
                        "source": source,
                        "disease_area": disease_area,
                        "section_index": sec_idx,
                        "section_title": section_title,
                    },
                }
            )
            global_index += 1

    print(f"[ingest_data]  -> Produced {len(chunks)} chunks from {path.name}")
    return chunks


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = read_raw_files()
    all_chunks: List[Dict[str, Any]] = []

    global_index = 0
    for path in raw_files:
        file_chunks = ingest_file(path, global_start_index=global_index)
        all_chunks.extend(file_chunks)
        global_index += len(file_chunks)

    # Write to JSONL
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[ingest_data] Wrote {len(all_chunks)} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
