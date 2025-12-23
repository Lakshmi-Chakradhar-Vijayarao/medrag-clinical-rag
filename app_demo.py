# app_demo.py
import json
import time
import traceback
from typing import Any, Dict, List

import requests
import gradio as gr

# --- Backend endpoints ---
BACKEND_BASE = "http://127.0.0.1:8000"
QUERY_URL = f"{BACKEND_BASE}/query"
PLAN_CHECK_URL = f"{BACKEND_BASE}/plan_check"


# ==========
# Helpers to call the FastAPI backend
# ==========

def call_query_api(query: str, mode: str = "fast", top_k: int = 4) -> Dict[str, Any]:
    payload = {
        "query": query,
        "mode": mode,
        "top_k": top_k,
    }
    resp = requests.post(QUERY_URL, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def call_plan_check_api(plan: str, condition: str | None, top_k: int = 4, max_pairs: int = 64) -> Dict[str, Any]:
    payload = {
        "plan": plan,
        "condition": condition or None,
        "top_k": top_k,
        "max_pairs": max_pairs,
    }
    resp = requests.post(PLAN_CHECK_URL, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


# ==========
# UI callbacks
# ==========

def medrag_qa_ui(query: str, mode: str, top_k: int):
    """
    Tab 1: Standard clinical Q&A using /query.
    Returns:
    - Answer markdown
    - Evidence markdown
    - Issues / warnings markdown
    """
    if not query.strip():
        return "⚠️ Please enter a question.", "", ""

    try:
        start = time.time()
        result = call_query_api(query=query, mode=mode, top_k=top_k)
        elapsed = time.time() - start

        is_supported = result.get("is_supported", False)
        confidence = result.get("confidence", "unknown")
        issues = result.get("issues", []) or []
        evidence = result.get("evidence", []) or []

        status = "✅ Supported" if is_supported else "⚠️ Not fully supported"
        header = (
            f"{status} | Confidence: **{confidence.title()}**\n\n"
            f"Mode: `{result.get('mode', mode)}` | Top-k: `{result.get('top_k', top_k)}` "
            f"| Latency: `{elapsed:.2f}s`\n\n"
            f"**Query:** {result.get('query', query)}\n"
        )

        answer_block = f"{header}\n\n### Answer\n\n{result.get('answer', '').strip() or '_No answer returned._'}"

        # Evidence formatting
        if evidence:
            ev_lines: List[str] = []
            for i, ev in enumerate(evidence):
                ev_lines.append(
                    f"**[{i}] id={ev.get('id','')}**\n\n"
                    f"- Source: `{ev.get('source','') or 'n/a'}`\n"
                    f"- Disease area: `{ev.get('disease_area','') or 'n/a'}`\n"
                    f"- Section: {ev.get('section_title','') or '`(no section title)`'}\n"
                    f"- Score: `{ev.get('score',0.0):.3f}`\n\n"
                    f"{ev.get('text','')[:600].replace(chr(10),' ')}..."
                )
            evidence_md = "\n\n---\n\n".join(ev_lines)
        else:
            evidence_md = "_No evidence returned._"

        # Issues / warnings
        if issues:
            issues_md = "### Warnings / Issues\n\n" + "\n".join(f"- {msg}" for msg in issues)
        else:
            issues_md = "_No issues reported._"

        return answer_block, evidence_md, issues_md

    except Exception as e:
        traceback.print_exc()
        err = f"❌ Error calling backend: `{type(e).__name__}` – {e}"
        return err, "_No evidence (error occurred)._", "_Check terminal for traceback._"


def medrag_plan_check_ui(plan: str, condition: str, top_k: int, max_pairs: int):
    """
    Tab 2: Plan check using /plan_check.
    Returns:
    - Summary markdown (what this tool is / is not)
    - Evidence markdown
    - Per-sentence alignment/conflict markdown
    """
    plan = (plan or "").strip()
    condition = (condition or "").strip()

    if not plan:
        return "⚠️ Please paste a plan / free-text note.", "", ""

    try:
        start = time.time()
        result = call_plan_check_api(plan=plan, condition=condition or None, top_k=top_k, max_pairs=max_pairs)
        elapsed = time.time() - start

        notes = result.get("notes", []) or []
        evidence = result.get("evidence", []) or []
        plan_sentences = result.get("plan_sentences", []) or []

        # Top summary block: strong disclaimers
        summary_lines = [
            "### Plan Check – Guideline Comparison (Research Only)\n",
            f"- Latency: `{elapsed:.2f}s`",
            f"- Condition hint: `{result.get('condition') or 'n/a'}`",
            f"- Top-k evidence: `{result.get('top_k', top_k)}`",
            f"- Max NLI pairs: `{result.get('max_pairs', max_pairs)}`",
            "",
            "**⚠️ This tool is for research and education only.**",
            "**It does NOT provide clinical advice, dosing, paediatric, pregnancy, or emergency recommendations.**",
        ]
        if notes:
            summary_lines.append("\n**System notes:**")
            summary_lines.extend([f"- {n}" for n in notes])

        summary_md = "\n".join(summary_lines)

        # Evidence block
        if evidence:
            ev_lines: List[str] = []
            for i, ev in enumerate(evidence):
                ev_lines.append(
                    f"**[{i}] id={ev.get('id','')}**\n\n"
                    f"- Source: `{ev.get('source','') or 'n/a'}`\n"
                    f"- Disease area: `{ev.get('disease_area','') or 'n/a'}`\n"
                    f"- Section: {ev.get('section_title','') or '`(no section title)`'}\n"
                    f"- Score: `{ev.get('score',0.0):.3f}`\n\n"
                    f"{ev.get('text','')[:600].replace(chr(10),' ')}..."
                )
            evidence_md = "\n\n---\n\n".join(ev_lines)
        else:
            evidence_md = "_No evidence retrieved._"

        # Plan sentence analysis block
        if plan_sentences:
            ps_lines: List[str] = ["### Per-sentence analysis\n"]
            for idx, ps in enumerate(plan_sentences):
                text = ps.get("text", "")
                aligned = ps.get("aligned", []) or []
                conflicts = ps.get("conflicts", []) or []

                ps_lines.append(f"**Sentence [{idx}]**: {text}\n")

                if aligned:
                    ps_lines.append("  - ✅ Guideline-aligned snippets:")
                    for a in aligned:
                        ps_lines.append(
                            f"    - From `{a.get('source','')}` / "
                            f"{a.get('section_title','') or '(no section title)'}, "
                            f"p_entail ≈ `{a.get('p_entailment',0.0):.3f}`\n"
                            f"      → _{a.get('evidence_sentence','')[:200].replace(chr(10),' ')}..._"
                        )
                else:
                    ps_lines.append("  - ✅ No strong guideline alignment found.")

                if conflicts:
                    ps_lines.append("  - ⚠️ Potential guideline conflicts:")
                    for c in conflicts:
                        ps_lines.append(
                            f"    - From `{c.get('source','')}` / "
                            f"{c.get('section_title','') or '(no section title)'}, "
                            f"p_contra ≈ `{c.get('p_contradiction',0.0):.3f}`\n"
                            f"      → _{c.get('evidence_sentence','')[:200].replace(chr(10),' ')}..._"
                        )
                else:
                    ps_lines.append("  - ⚠️ No high-confidence conflicts detected.")

                ps_lines.append("")  # blank line between sentences

            plan_analysis_md = "\n".join(ps_lines)
        else:
            plan_analysis_md = "_No per-sentence analysis available._"

        return summary_md, evidence_md, plan_analysis_md

    except Exception as e:
        traceback.print_exc()
        err = f"❌ Error calling /plan_check: `{type(e).__name__}` – {e}"
        return err, "_No evidence (error occurred)._", "_Check terminal for traceback._"


# ==========
# Build Gradio UI (two tabs)
# ==========

with gr.Blocks(title="MedRAG – Clinical RAG Assistant (Backend-powered)") as demo:
    gr.Markdown(
        """
# MedRAG – Clinical RAG Assistant (Backend-powered)

Frontend demo calling the FastAPI MedRAG backend at `/query` and `/plan_check`.

- **Tab 1 – Clinical Q&A:** Ask guideline-style questions (non-dosing, adult, non-emergency).
- **Tab 2 – Plan Check:** Paste a free-text plan and compare it to guideline snippets.

⚠️ This system is for research and educational use only, not for real clinical decision-making.
"""
    )

    with gr.Tab("Clinical Q&A"):
        with gr.Row():
            query_input = gr.Textbox(
                lines=3,
                label="Clinical-style question",
                placeholder="e.g., What is the first-line pharmacologic treatment for Type 2 diabetes?",
            )
        with gr.Row():
            mode_input = gr.Radio(
                choices=["fast", "full"],
                value="fast",
                label="Mode",
                info="fast = generator only; full = generator + verifier (slower)",
            )
            topk_input = gr.Slider(
                minimum=1,
                maximum=8,
                step=1,
                value=4,
                label="Top-k evidence",
            )

        qa_button = gr.Button("Submit")
        qa_clear = gr.Button("Clear")

        answer_md = gr.Markdown(label="Answer")
        evidence_md = gr.Markdown(label="Evidence")
        issues_md = gr.Markdown(label="Warnings / Issues")

        qa_button.click(
            fn=medrag_qa_ui,
            inputs=[query_input, mode_input, topk_input],
            outputs=[answer_md, evidence_md, issues_md],
        )

        qa_clear.click(
            lambda: ("", "", "",),
            inputs=None,
            outputs=[answer_md, evidence_md, issues_md],
        )

    with gr.Tab("Plan Check (Guideline comparison)"):
        gr.Markdown(
            """
Paste a short free-text **plan / note** and (optionally) a condition keyword, e.g.:

> Plan: Start metformin as first-line therapy for this adult with newly diagnosed type 2 diabetes and reinforce lifestyle modification.

This tool will retrieve relevant guideline snippets and run a small NLI model to highlight **aligned** and **potentially conflicting** statements.

⚠️ Research only – not clinical advice.
"""
        )

        plan_input = gr.Textbox(
            lines=4,
            label="Plan / clinical note (free text)",
            placeholder="e.g., Plan: Start metformin as first-line therapy for this adult with newly diagnosed type 2 diabetes and reinforce lifestyle modification.",
        )
        condition_input = gr.Textbox(
            lines=1,
            label="Condition hint (optional)",
            placeholder="e.g., type 2 diabetes, hypertension",
        )
        with gr.Row():
            plan_topk_input = gr.Slider(
                minimum=1,
                maximum=8,
                step=1,
                value=4,
                label="Top-k evidence",
            )
            plan_maxpairs_input = gr.Slider(
                minimum=8,
                maximum=128,
                step=8,
                value=64,
                label="Max NLI pairs (plan × evidence sentences)",
            )

        plan_button = gr.Button("Run Plan Check")
        plan_clear = gr.Button("Clear")

        plan_summary_md = gr.Markdown(label="Summary & Disclaimers")
        plan_evidence_md = gr.Markdown(label="Retrieved Evidence")
        plan_analysis_md = gr.Markdown(label="Plan Sentence Analysis")

        plan_button.click(
            fn=medrag_plan_check_ui,
            inputs=[plan_input, condition_input, plan_topk_input, plan_maxpairs_input],
            outputs=[plan_summary_md, plan_evidence_md, plan_analysis_md],
        )

        plan_clear.click(
            lambda: ("", "", "",),
            inputs=None,
            outputs=[plan_summary_md, plan_evidence_md, plan_analysis_md],
        )


if __name__ == "__main__":
    demo.launch()
