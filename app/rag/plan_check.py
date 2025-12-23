# app/rag/plan_check.py

import re
from typing import List, Dict, Any, Optional

import torch
from sentence_transformers import CrossEncoder


class PlanChecker:
    """
    PlanChecker: analyse a free-text clinical plan against retrieved guideline evidence.

    - Uses the existing retriever to pull top-k guideline chunks.
    - Splits plan and guideline text into sentences.
    - Uses an NLI model to score (plan_sentence, guideline_sentence) pairs.
    - For each plan sentence, surfaces:
        * aligned evidence (high entailment probability)
        * potential conflicts (high contradiction probability)

    IMPORTANT:
    - This is strictly for research/educational purposes, NOT clinical decision-making.
    """

    def __init__(
        self,
        retriever,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: Optional[str] = None,
    ):
        self.retriever = retriever

        print(f"[PlanChecker] Loading NLI model: {nli_model_name}")
        # CrossEncoder will pick device automatically if None, otherwise use e.g. "cpu", "cuda"
        self.nli = CrossEncoder(nli_model_name, device=device)

    def _split_sentences(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def check(
        self,
        plan_text: str,
        condition: Optional[str] = None,
        top_k: int = 4,
        max_pairs: int = 64,
    ) -> Dict[str, Any]:
        """
        Run plan-vs-guideline analysis.

        Args:
            plan_text: free-text description of a proposed management plan.
            condition: optional disease/condition string to steer retrieval (e.g. "hypertension").
            top_k: how many chunks to retrieve from the guideline corpus.
            max_pairs: max (plan_sentence, guideline_sentence) pairs to score with NLI.

        Returns:
            dict with keys:
                - plan_sentences: list of {text, aligned, conflicts}
                - evidence_chunks: the retrieved chunks (same objects as retriever returns)
                - notes: high-level disclaimers / info
        """
        plan_text = (plan_text or "").strip()
        cond = (condition or "").strip()
        query = cond if cond else plan_text

        # 1. Retrieve guideline chunks using the existing hybrid retriever
        chunks = self.retriever.retrieve(query, top_k=top_k) or []

        plan_sents = self._split_sentences(plan_text)

        # Collect guideline sentences with metadata
        guideline_sents: List[Dict[str, Any]] = []
        for c in chunks:
            chunk_text = c.get("text", "") or ""
            meta = c.get("metadata", {}) or {}
            source = meta.get("source", "")
            section_title = meta.get("section_title", "")
            chunk_id = c.get("id", "")

            for s in self._split_sentences(chunk_text):
                guideline_sents.append(
                    {
                        "text": s,
                        "chunk_id": chunk_id,
                        "source": source,
                        "section_title": section_title,
                    }
                )

        # If no sentences or no evidence, return early with disclaimers
        if not plan_sents or not guideline_sents:
            return {
                "plan_sentences": [
                    {"text": s, "aligned": [], "conflicts": []} for s in plan_sents
                ],
                "evidence_chunks": chunks,
                "notes": [
                    "Insufficient plan or guideline sentences for analysis.",
                    "This tool is for research/education only and does not provide clinical advice.",
                ],
            }

        # 2. Build (plan_sentence, guideline_sentence) pairs, capped by max_pairs
        pairs: List[List[str]] = []
        pair_meta: List[Dict[str, Any]] = []

        for ps in plan_sents:
            for gs in guideline_sents:
                if len(pairs) >= max_pairs:
                    break
                pairs.append([ps, gs["text"]])
                pair_meta.append({"plan": ps, "guideline": gs})
            if len(pairs) >= max_pairs:
                break

        if not pairs:
            return {
                "plan_sentences": [
                    {"text": s, "aligned": [], "conflicts": []} for s in plan_sents
                ],
                "evidence_chunks": chunks,
                "notes": [
                    "No sentence pairs created for analysis.",
                    "This tool is for research/education only and does not provide clinical advice.",
                ],
            }

        # 3. Run NLI model on all pairs
        # CrossEncoder for NLI typically outputs 3 logits: [contradiction, neutral, entailment]
        logits = self.nli.predict(pairs)
        logits_tensor = torch.tensor(logits)
        probs = torch.softmax(logits_tensor, dim=1).tolist()

        # 4. Aggregate per plan sentence
        per_sentence: Dict[str, Dict[str, Any]] = {
            s: {"text": s, "aligned": [], "conflicts": []} for s in plan_sents
        }

        for meta, prob in zip(pair_meta, probs):
            plan_text_sent = meta["plan"]
            gl = meta["guideline"]

            # Assuming label order: [contradiction, neutral, entailment]
            p_contra, p_neutral, p_entail = prob

            rec = {
                "evidence_sentence": gl["text"],
                "evidence_id": gl["chunk_id"],
                "source": gl["source"],
                "section_title": gl["section_title"],
                "p_contradiction": float(p_contra),
                "p_entailment": float(p_entail),
            }

            entry = per_sentence[plan_text_sent]

            # Thresholds are heuristic; you can tune later based on eval
            if p_entail >= 0.6:
                entry["aligned"].append(rec)
            if p_contra >= 0.6:
                entry["conflicts"].append(rec)

        # 5. Keep only top-k evidence per category per plan sentence
        def _topn(lst: List[Dict[str, Any]], key: str, n: int = 3):
            return sorted(lst, key=lambda x: x.get(key, 0.0), reverse=True)[:n]

        for s, entry in per_sentence.items():
            entry["aligned"] = _topn(entry["aligned"], "p_entailment", n=3)
            entry["conflicts"] = _topn(entry["conflicts"], "p_contradiction", n=3)

        return {
            "plan_sentences": list(per_sentence.values()),
            "evidence_chunks": chunks,
            "notes": [
                "This is an automated guideline comparison tool for research and education only.",
                "It does NOT provide clinical advice, dosing, paediatric, pregnancy, or emergency recommendations.",
                "Any potential 'conflicts' are heuristic and must be reviewed by qualified clinicians.",
            ],
        }
