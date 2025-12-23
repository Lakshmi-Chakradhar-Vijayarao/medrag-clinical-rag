# server.py
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from app.db.chroma_store import ChromaStore
from app.db.bm25_index import BM25Index
from app.rag.retrieval import HybridRetriever
from app.llm.client import LocalLLMClient
from app.rag.pipeline import MedRAGPipeline
from app.rag.plan_check import PlanChecker  # NEW

PROJECT_ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = PROJECT_ROOT / "data/processed/chunks.jsonl"


def load_chunks() -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# Initialize global components once (GPU/CPU-friendly)
print("[server] Loading chunks...")
_chunks = load_chunks()
print(f"[server] Loaded {_chunks and len(_chunks)} chunks")

print("[server] Initializing Chroma...")
_chroma = ChromaStore(str(PROJECT_ROOT / "chroma_db"))

print("[server] Initializing BM25...")
_bm25 = BM25Index(_chunks)

print("[server] Initializing retriever...")
_retriever = HybridRetriever(_chroma, _bm25, alpha=0.6)

print("[server] Loading LLM...")
_llm = LocalLLMClient()

print("[server] Creating MedRAGPipeline...")
_pipeline = MedRAGPipeline(_retriever, _llm)

print("[server] Creating PlanChecker (NLI-based)...")
_plan_checker = PlanChecker(_retriever)

# Simple in-memory cache: query -> result
_cache: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="MedRAG Clinical RAG API",
    description="Hybrid dense+BM25 clinical retrieval with local TinyLlama generator.",
)


# ==========
# Pydantic models
# ==========

class EvidenceItem(BaseModel):
    id: str | None = None
    source: str = ""
    disease_area: str = ""
    section_title: str = ""
    score: float = 0.0
    text: str = ""


class QueryRequest(BaseModel):
    query: str
    mode: str = "fast"
    top_k: int = 4


class QueryResponse(BaseModel):
    query: str
    answer: str
    is_supported: bool
    issues: List[str]
    latency_seconds: float
    mode: str
    top_k: int
    confidence: str
    max_evidence_score: float
    avg_top2_evidence_score: float
    evidence: List[EvidenceItem]


class BatchQueryRequest(BaseModel):
    queries: List[str]
    mode: str = "fast"
    top_k: int = 4


class BatchQueryResponseItem(BaseModel):
    query: str
    answer: str
    is_supported: bool
    issues: List[str]
    latency_seconds: float
    mode: str
    top_k: int
    confidence: str
    max_evidence_score: float
    avg_top2_evidence_score: float
    evidence: List[EvidenceItem]


# ---- NEW models for /plan_check ----

class PlanCheckRequest(BaseModel):
    plan: str
    condition: str | None = None
    top_k: int = 4
    max_pairs: int = 64


class PlanSentenceAnalysis(BaseModel):
    text: str
    aligned: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]]


class PlanCheckResponse(BaseModel):
    plan: str
    condition: str | None
    mode: str
    top_k: int
    max_pairs: int
    latency_seconds: float
    evidence: List[EvidenceItem]
    plan_sentences: List[PlanSentenceAnalysis]
    notes: List[str]


# ==========
# Helpers
# ==========

def serialize_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert internal evidence chunks into a clean list of dicts for the API.
    """
    out: List[Dict[str, Any]] = []
    for c in evidence:
        meta = c.get("metadata", {}) or {}
        out.append(
            {
                "id": c.get("id"),
                "source": meta.get("source", ""),
                "disease_area": meta.get("disease_area", ""),
                "section_title": meta.get("section_title", ""),
                "score": float(c.get("score", 0.0)),
                "text": c.get("text", ""),
            }
        )
    return out


def compute_confidence(evidence: List[Dict[str, Any]], issues: List[str]) -> dict:
    """
    Simple heuristic confidence based on evidence scores + issues.

    Returns:
        {
          "confidence": "high" | "medium" | "low",
          "max_evidence_score": float,
          "avg_top2_evidence_score": float
        }
    """
    scores = [float(c.get("score", 0.0)) for c in evidence if "score" in c]
    if not scores:
        return {
            "confidence": "low",
            "max_evidence_score": 0.0,
            "avg_top2_evidence_score": 0.0,
        }

    scores_sorted = sorted(scores, reverse=True)
    max_score = scores_sorted[0]
    if len(scores_sorted) >= 2:
        avg_top2 = (scores_sorted[0] + scores_sorted[1]) / 2.0
    else:
        avg_top2 = scores_sorted[0]

    # Very simple heuristic
    has_issues = bool(issues)
    if max_score >= 0.75 and not has_issues:
        conf = "high"
    elif max_score >= 0.4:
        conf = "medium"
    else:
        conf = "low"

    return {
        "confidence": conf,
        "max_evidence_score": max_score,
        "avg_top2_evidence_score": avg_top2,
    }


def chunks_to_evidence_items(chunks: List[Dict[str, Any]]) -> List[EvidenceItem]:
    """
    Convert internal evidence chunks into EvidenceItem models.
    Reuses the same metadata fields as serialize_evidence.
    """
    serialized = serialize_evidence(chunks)
    return [EvidenceItem(**e) for e in serialized]


# ==========
# Endpoints
# ==========

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "MedRAG backend is running",
        "endpoints": ["/query", "/batch_query", "/plan_check"],
        "docs": "/docs",
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    key = f"{req.mode}::{req.top_k}::{req.query.strip()}"
    if key in _cache:
        cached = _cache[key]
        return QueryResponse(**cached)

    start = time.time()
    result = _pipeline.run(req.query, mode=req.mode, top_k=req.top_k)
    elapsed = time.time() - start

    meta = result.get("meta", {}) or {}
    evidence_internal = result.get("evidence", []) or []
    evidence_serialized = serialize_evidence(evidence_internal)

    conf_info = compute_confidence(evidence_serialized, result.get("issues", []))

    payload: Dict[str, Any] = {
        "query": result.get("query", req.query),
        "answer": result.get("answer", ""),
        "is_supported": result.get("is_supported", False),
        "issues": result.get("issues", []),
        "latency_seconds": elapsed,
        "mode": meta.get("mode", req.mode),
        "top_k": meta.get("top_k", req.top_k),
        "confidence": conf_info["confidence"],
        "max_evidence_score": conf_info["max_evidence_score"],
        "avg_top2_evidence_score": conf_info["avg_top2_evidence_score"],
        "evidence": evidence_serialized,
    }

    _cache[key] = payload
    return QueryResponse(**payload)


@app.post("/batch_query", response_model=List[BatchQueryResponseItem])
def batch_query_endpoint(req: BatchQueryRequest):
    responses: List[BatchQueryResponseItem] = []

    for q in req.queries:
        single = query_endpoint(QueryRequest(query=q, mode=req.mode, top_k=req.top_k))
        # single is a QueryResponse model; we reuse its data
        responses.append(BatchQueryResponseItem(**single.dict()))

    return responses


@app.post("/plan_check", response_model=PlanCheckResponse)
def plan_check_endpoint(req: PlanCheckRequest):
    """
    Compare a free-text clinical plan against guideline evidence.

    IMPORTANT:
    - This endpoint is for research and educational purposes only.
    - It does NOT provide clinical advice or treatment recommendations.
    """
    start = time.time()
    analysis = _plan_checker.check(
        plan_text=req.plan,
        condition=req.condition,
        top_k=req.top_k,
        max_pairs=req.max_pairs,
    )
    elapsed = time.time() - start

    evidence_chunks = analysis.get("evidence_chunks", []) or []
    evidence_items = chunks_to_evidence_items(evidence_chunks)

    plan_sentences_raw = analysis.get("plan_sentences", []) or []
    notes = analysis.get("notes", []) or []

    plan_sentences_typed: List[PlanSentenceAnalysis] = []
    for ps in plan_sentences_raw:
        plan_sentences_typed.append(
            PlanSentenceAnalysis(
                text=ps.get("text", ""),
                aligned=ps.get("aligned", []) or [],
                conflicts=ps.get("conflicts", []) or [],
            )
        )

    return PlanCheckResponse(
        plan=req.plan,
        condition=req.condition,
        mode="plan_check:v1",
        top_k=req.top_k,
        max_pairs=req.max_pairs,
        latency_seconds=elapsed,
        evidence=evidence_items,
        plan_sentences=plan_sentences_typed,
        notes=notes,
    )
