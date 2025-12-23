# app/rag/pipeline.py
from typing import Dict, Any, List
import re

from app.rag.agents import generate_answer, verify_answer, simple_lexical_mismatch
from app.rag.safety import check_scope_warnings


class MedRAGPipeline:
    """
    MedRAGPipeline orchestrates retrieval + generation + (optional) verification.

    Modes:
        - "fast":  single LLM call (generator only) + lexical guardrails.
        - "full":  generator + verifier LLM + lexical guardrails (slower, for eval).
    """

    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm = llm_client

    def _strip_prompt_from_answer(self, text: str) -> str:
        marker = "Answer:"
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return text.strip()

    def _first_n_sentences(self, text: str, n: int = 2) -> str:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        parts = [p for p in parts if p.strip()]
        if not parts:
            return text.strip()
        return " ".join(parts[:n]).strip()

    def _compute_confidence(self, chunks: List[Dict[str, Any]], has_issues: bool) -> Dict[str, Any]:
        """
        Compute a simple confidence signal based on fused retrieval scores and issues.
        Returns both the numeric scores and a label.
        """
        scores = [
            float(c.get("score", 0.0))
            for c in chunks
            if isinstance(c.get("score", 0.0), (int, float))
        ]
        if not scores:
            return {
                "label": "unknown",
                "max_score": 0.0,
                "avg_top2": 0.0,
            }

        max_score = max(scores)
        top2 = scores[:2]
        avg_top2 = sum(top2) / len(top2)

        # Simple heuristic mapping → label
        if has_issues:
            # Any issues -> at most medium
            if max_score >= 0.7:
                label = "medium"
            elif max_score >= 0.4:
                label = "low"
            else:
                label = "low"
        else:
            if max_score >= 0.8:
                label = "high"
            elif max_score >= 0.5:
                label = "medium"
            else:
                label = "low"

        return {
            "label": label,
            "max_score": max_score,
            "avg_top2": avg_top2,
        }

    def run(self, query: str, mode: str = "fast", top_k: int = 4) -> Dict[str, Any]:
        """
        Run the full MedRAG pipeline.

        Args:
            query: user question.
            mode: "fast" or "full".
            top_k: number of evidence chunks to retrieve.

        Returns:
            dict with keys: query, answer, is_supported, issues, evidence, meta.
        """
        mode = mode.lower()
        if mode not in ("fast", "full"):
            raise ValueError(f"Unsupported mode: {mode}")

        # 0. Safety pre-checks based on the *query* itself
        scope_issues = check_scope_warnings(query)

        # 1. Retrieve evidence
        chunks = self.retriever.retrieve(query, top_k=top_k)

        # 2. Generate initial answer
        gen_raw = generate_answer(self.llm, query, chunks)
        gen_answer = self._strip_prompt_from_answer(gen_raw)

        # 3. Lightweight lexical sanity check
        lex_issues = simple_lexical_mismatch(gen_answer, chunks)

        verif_issues: List[str] = []
        is_supported = True
        final_answer = gen_answer

        if mode == "fast":
            # FAST MODE: no verifier LLM call
            final_answer = self._first_n_sentences(gen_answer, n=1)
            is_supported = not bool(lex_issues)
        else:
            # FULL MODE: call verifier LLM as well (slower, for offline eval)
            verif = verify_answer(self.llm, query, chunks, gen_answer)
            final_answer_raw = verif.get("final_answer", gen_answer)
            final_answer = self._strip_prompt_from_answer(final_answer_raw)
            is_supported = bool(verif.get("is_supported", True))
            verif_issues = verif.get("issues", []) or []

        # 4. Combine issues from safety + lexical + verifier
        issues: List[str] = list(scope_issues) + list(lex_issues) + list(verif_issues)

        # 5. If lexical issues exist, force is_supported to False and trim answer
        if lex_issues:
            is_supported = False
            final_answer = self._first_n_sentences(final_answer, n=1)

        # 6. Hard safety override for out-of-scope clinical queries
        if scope_issues:
            # For any scope violation, we *do not trust* the model answer at all.
            is_supported = False
            final_answer = (
                "I can't safely provide a specific clinical recommendation for this question. "
                "This assistant is not designed for dosing, pregnancy, paediatric, or emergency "
                "decision-making. Please consult up-to-date clinical guidelines and a licensed clinician."
            )

        # 7. Confidence based on retrieval scores + issues
        conf_info = self._compute_confidence(chunks, has_issues=bool(issues))

        # If scope warnings are present, force confidence to 'low' regardless of scores.
        if scope_issues:
            conf_info["label"] = "low"

        return {
            "query": query,
            "answer": final_answer,
            "is_supported": is_supported,
            "issues": issues,
            "evidence": chunks,
            "meta": {
                "mode": mode,
                "top_k": top_k,
                "confidence": conf_info["label"],
                "max_evidence_score": conf_info["max_score"],
                "avg_top2_evidence_score": conf_info["avg_top2"],
            },
        }
