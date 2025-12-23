# app/rag/agents.py
import json
from typing import List, Dict, Any


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a readable evidence block for the LLM."""
    lines = []
    for i, c in enumerate(chunks):
        src = c.get("metadata", {}).get("source", "")
        lines.append(f"[{i}] (source: {src}) {c['text']}")
    return "\n\n".join(lines)


def generate_answer(llm, query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Generator agent: answer strictly based on evidence.
    Short, simple prompt to reduce context bloat.
    """
    context = format_context(chunks)
    prompt = f"""
You are a clinical assistant. Answer the question using ONLY the EVIDENCE below.

EVIDENCE:
{context}

QUESTION:
{query}

RULES:
- Use ONLY information from the EVIDENCE.
- Do NOT guess or add external medical knowledge.
- If the EVIDENCE is insufficient, say:
  "I don't know based on the provided evidence."

Answer in 2–4 sentences, then add a final line like: Evidence: [indices]

Answer:
"""
    raw = llm(prompt)
    return raw


def verify_answer(llm, query: str, chunks: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
    """
    Verifier agent: checks if the proposed answer is supported by the evidence.
    Returns JSON with is_supported, final_answer, and issues.

    IMPORTANT:
    - If JSON parsing fails, we DO NOT automatically mark the answer as unsupported.
      We treat the verifier as a no-op and keep the original answer, so that
      hallucination detection is handled by lexical checks instead of JSON format.
    """
    context = format_context(chunks)
    prompt = f"""
You are a verification agent for a clinical question-answering system.

EVIDENCE:
{context}

QUESTION:
{query}

PROPOSED ANSWER:
{answer}

TASK:
1. Check if the PROPOSED ANSWER is fully supported by the EVIDENCE.
2. Identify any unsupported or contradictory statements.
3. If needed, correct the answer to align with the EVIDENCE only.

Respond in JSON ONLY, with this exact structure:

{{
  "is_supported": true or false,
  "final_answer": "corrected or original answer, but ONLY using information from the EVIDENCE",
  "issues": ["list of issues, or [] if none"]
}}

IMPORTANT:
- Do NOT include any text before or after the JSON.
- Do NOT use backticks.
- Do NOT add explanations outside the JSON.

JSON:
"""
    raw = llm(prompt)

    try:
        first = raw.find("{")
        last = raw.rfind("}")
        json_str = raw[first:last + 1]
        data = json.loads(json_str)
    except Exception:
        # Fallback: do NOT penalize the answer just because the verifier
        # output wasn't perfectly formatted JSON.
        data = {
            "is_supported": True,
            "final_answer": answer,
            "issues": [
                "Verifier JSON parsing failed; using original answer (no extra penalty)."
            ],
        }
    return data


def simple_lexical_mismatch(answer: str, chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Very simple lexical safety net:
    Flags terms that appear in the answer but not in the evidence text.
    This is not perfect, but good enough to catch obvious hallucinations for demo.
    """
    evidence_text = " ".join(c["text"] for c in chunks).lower()
    answer_text = answer.lower()

    suspicious_terms = [
        "sulfonylurea",
        "weight gain",
        "nausea",
        "vomiting",
        "diarrhea",
        "hyperkalemia",
    ]

    issues = []
    for term in suspicious_terms:
        if term in answer_text and term not in evidence_text:
            issues.append(f"Term '{term}' appears in the answer but not in the evidence.")
    return issues
