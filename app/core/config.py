# app/core/config.py

from __future__ import annotations
from typing import Literal, Dict, Any

# Simple global config dict. You can tweak these and re-run without code changes.

RAG_CONFIG: Dict[str, Any] = {
    "mode_default": "fast",      # "fast" | "full"
    "top_k_default": 4,
    "hybrid_alpha": 0.6,
    "use_rerank": True,

    "retrieval": {
        # Main general-purpose encoder (what you're using now)
        "encoder_name_main": "sentence-transformers/all-MiniLM-L6-v2",

        # Optional biomedical encoder (only used if you flip use_biomedical_encoder=True)
        # Make sure the model is available in sentence-transformers before enabling.
        "encoder_name_biomed": "pritamdeka/S-BioClinicalBERT-Sentence-Embeddings",

        # Global toggle: False → use encoder_name_main, True → use encoder_name_biomed
        "use_biomedical_encoder": False,
    },

    "cross_encoder": {
        # Already used in your reranker
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },

    "llm": {
        "backend": "tinyllama",   # Just a label for now
        "max_input_tokens": 1536,
        "max_new_tokens": 96,
    },
}


def get_active_encoder_name() -> str:
    """
    Helper for ChromaStore to decide which encoder to load.
    """
    retr_cfg = RAG_CONFIG["retrieval"]
    if retr_cfg.get("use_biomedical_encoder", False):
        return retr_cfg["encoder_name_biomed"]
    return retr_cfg["encoder_name_main"]


def is_biomedical_encoder_active() -> bool:
    return bool(RAG_CONFIG["retrieval"].get("use_biomedical_encoder", False))


def get_hybrid_alpha() -> float:
    return float(RAG_CONFIG.get("hybrid_alpha", 0.6))


def use_rerank() -> bool:
    return bool(RAG_CONFIG.get("use_rerank", True))
