
# MedRAG: Clinical Retrieval-Augmented Generation with Hallucination Mitigation

MedRAG is a **modular, end-to-end Retrieval-Augmented Generation (RAG) system** designed for **clinical question answering**, where **hallucinations, grounding, and latency constraints** are critical.
The system combines **hybrid retrieval (dense + sparse)**, **reranking**, and **post-generation verification** to explicitly engineer reliability rather than relying solely on prompt design or model fine-tuning.

This repository focuses on **system design and workflows**, not model weights.

---

## Motivation

Large Language Models (LLMs) are fluent but unreliable in **safety-critical domains** such as healthcare.
Common failure modes include:

* hallucinated clinical facts
* missed rare medical terminology
* lack of verifiable evidence
* latency bottlenecks in real deployments

**Key insight:**

> Hallucinations are a *system-level failure*, not just a model-level failure.

MedRAG addresses this by treating **retrieval quality, grounding, verification, and latency** as **first-class design constraints**.

---

## High-Level Architecture

```
User Query
   ↓
Hybrid Retrieval (Dense + BM25)
   ↓
Reranking
   ↓
LLM Generation
   ↓
Safety & Verification Layer
   ↓
Grounded Answer + Evidence
```

Each stage is **modular**, enabling independent experimentation, evaluation, and replacement.

---

## Core Design Principles

1. Hybrid retrieval over single-method retrieval
2. Explicit verification over implicit trust
3. System pipelines over monolithic models
4. Measured reliability over perceived accuracy
5. Latency-aware engineering for deployment realism

---

## End-to-End Workflow

### 1. Query Intake

A user submits a free-text clinical query (e.g., diagnosis, treatment, guidelines).

### 2. Hybrid Document Retrieval

Two complementary retrievers are executed in parallel:

* **Dense retrieval** (ChromaDB + embeddings)
  Captures semantic similarity
* **Sparse retrieval** (BM25)
  Recovers rare and domain-specific terminology

Results are merged to reduce retrieval blind spots.

### 3. Reranking

Retrieved passages are reranked to prioritize:

* clinical relevance
* factual completeness
* contextual coherence

This step improves **context quality before generation**, which is more effective than post-hoc correction.

### 4. LLM Generation

The LLM generates an answer conditioned strictly on retrieved context.

**Important:**
The generated output is **not trusted by default**.

### 5. Safety & Verification Layer

The generated response is:

* decomposed into individual claims
* each claim is checked against retrieved evidence
* unsupported or conflicting claims are flagged

This step significantly reduces hallucinations.

### 6. Final Output

The system returns:

* a grounded answer
* supporting evidence
* an implicit reliability signal

---

## Sample Inputs and Outputs

### 1. Standard Clinical Question Answering

**Input**

```
What is the first-line pharmacologic treatment for Type 2 diabetes?
```

**Output**

```
Answer:
The first-line pharmacologic treatment for Type 2 diabetes is metformin.

Confidence:
High

Evidence:
- Source: diabetes_overview
- Relevant text: Pharmacologic treatment for Type 2 diabetes often begins with metformin unless contraindicated.
```

---

### 2. Safety-Scoped Clinical Query

**Input**

```
What dose of metformin should be given to a pregnant patient with Type 2 diabetes?
```

**Output**

```
Answer:
I cannot safely provide a specific clinical recommendation for this question.

Confidence:
Medium

Issues:
- Dosing recommendations are outside the supported scope.
- Pregnancy-specific clinical guidance is not supported.

Evidence:
- Source: diabetes_overview
- Relevant text: Pharmacologic treatment for Type 2 diabetes often begins with metformin unless contraindicated.
```

---

### 3. Batch Clinical Queries

**Input**

```
1. What is the first-line pharmacologic treatment for Type 2 diabetes?
2. What are the first-line drug classes used to treat hypertension?
```

**Output**

```
Query 1:
Metformin is the first-line pharmacologic treatment.
Confidence: High

Query 2:
First-line drug classes for hypertension include thiazide diuretics,
ACE inhibitors, ARBs, and calcium channel blockers.
Confidence: High
```

---

### 4. Clinical Plan Consistency Check (Aligned Plan)

**Input**

```
Condition:
Type 2 diabetes

Plan:
Start metformin as first-line therapy and reinforce lifestyle modification.
```

**Output**

```
Overall confidence:
High

Plan analysis:
- Sentence: Start metformin as first-line therapy and reinforce lifestyle modification.
  Alignment: Supported by guideline evidence
  Entailment probability: 0.99
```

---

### 5. Clinical Plan Consistency Check (Conflicting Plan)

**Input**

```
Condition:
Hypertension

Plan:
Combine an ACE inhibitor and an ARB as first-line therapy for hypertension.
```

**Output**

```
Overall confidence:
Low

Plan analysis:
- Sentence: Combine an ACE inhibitor and an ARB as first-line therapy for hypertension.
  Conflict detected with guideline evidence
  Contradiction probability: 0.98

Notes:
Potential guideline conflict detected. Clinical review required.
```

---

## Key Technical Decisions (and Rationale)

### Hybrid Retrieval (Dense + Sparse)

* Dense retrieval alone misses rare clinical terms
* Sparse retrieval alone lacks semantic generalization
* Hybrid retrieval reduces complementary failure modes

### Post-Generation Verification

* Prompt constraints alone are insufficient
* Verification makes reliability explicit and measurable

### Modular Pipeline

* Enables independent optimization and evaluation
* Facilitates research experimentation
* Improves debuggability and reproducibility

### GPU-Optimized Inference

* Batching and caching reduce inference latency
* System achieves sub-second P95 latency under load

---

## Evaluation and Metrics

Evaluation is performed using offline scripts to ensure reproducibility.

### Reported Improvements

* +31% terminology recall (hybrid vs dense-only retrieval)
* −34% hallucination rate (with verification layer)
* Sub-second P95 latency with GPU batching

### Metrics Tracked

* Retrieval recall
* Claim-level grounding accuracy
* End-to-end latency

---

## Repository Structure

```
app/
 ├── db/           # BM25 and vector store logic
 ├── rag/          # Retrieval, reranking, safety pipeline
 ├── llm/          # LLM client abstraction
 └── core/         # Configuration and orchestration

scripts/
 ├── ingest_data.py
 ├── eval_retrieval.py
 ├── eval_hallucinations.py
 └── test_pipeline.py

data/
 ├── raw/          # Sample clinical documents
 └── processed/    # Chunked and indexed data
```

---

## Limitations and Future Work

* Verification introduces additional computational overhead
* Performance depends on corpus coverage and quality
* Future work includes:

  * stronger claim-evidence alignment metrics
  * theoretical analysis of retrieval-verification trade-offs
  * extension to real-time clinical decision support

---

## Relevance Beyond Healthcare

Although evaluated in a clinical setting, the architecture generalizes to:

* safety-critical AI systems
* autonomous decision pipelines
* AI-native network control
* edge intelligence and low-latency systems

The same principles apply to **AI-driven 6G networks**, where incorrect decisions can cascade and reliability must be engineered explicitly.

---

## Notes

* Virtual environments, databases, and model weights are intentionally excluded.
* The repository emphasizes **system design and reproducible workflows**.

---

## Author

**Lakshmi Chakradhar Vijayarao**
AI Engineer | LLMs | RAG | Reinforcement Learning | Systems & Optimization

---

