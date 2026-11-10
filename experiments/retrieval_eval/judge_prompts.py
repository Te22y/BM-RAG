"""
judge_prompts.py
================

All LLM-judge prompts of the evaluation (phases 1 and 2).
The judge model is fixed at gpt-5.2 (not the GPT-4o generator, to avoid self-preference bias).

Every prompt asks for strict JSON so replies can be parsed.
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1. Faithfulness / Groundedness
#    Input: question + context_docs + generated_answer
#    Output: {"score": 0.0–1.0, "unsupported_claims": [...], "reasoning": "..."}
#
#    Meaning: whether each atomic claim in the answer is supported by the context.
#    score = supported_claims / total_claims
# ══════════════════════════════════════════════════════════════════════════════
FAITHFULNESS_PROMPT = """\
You are an expert grader for a scientific RAG benchmark in building environment and
materials science. Your task is to evaluate the **faithfulness** of a generated answer
with respect to the retrieved context documents.

DEFINITION
----------
Faithfulness measures whether every factual claim in the answer is explicitly supported
by the context. A claim is "supported" if it can be verified directly from the context
text — no inference, no outside knowledge. If a claim is partially supported (correct
direction but wrong magnitude), count it as UNSUPPORTED.

IMPORTANT RULES
---------------
- Domain-specific units: treat equivalent representations as identical
  (e.g., 0.81 GPa = 810 MPa, 1 W/m·K = 1 W/(m·K)).
- Do NOT penalise the answer for correct, well-known constants that are not in the
  context (e.g., Boltzmann's constant). Penalise only domain-specific factual claims.
- "I don't know" or explicit uncertainty statements should be scored as faithful (they
  make no unsupported claims).

SCORING
-------
1. List every distinct factual CLAIM in the generated answer (ignore filler phrases).
2. For each claim, mark it SUPPORTED (S) or UNSUPPORTED (U) based solely on the
   provided context.
3. score = (number of S claims) / (total claims). If there are 0 claims, score = 1.0.

OUTPUT FORMAT (strict JSON, no markdown fences, no commentary):
{{
  "claims": [
    {{"claim": "...", "supported": true | false, "evidence": "...or null if unsupported"}}
  ],
  "score": 0.0,
  "reasoning": "one short sentence"
}}

=== QUESTION ===
{question}

=== CONTEXT DOCUMENTS ===
{context}

=== GENERATED ANSWER ===
{answer}
"""

# ══════════════════════════════════════════════════════════════════════════════
# 2. Answer Correctness
#    Input: question + gold_answer + generated_answer
#    Output: {"score": 0 | 0.5 | 1, "reasoning": "..."}
#
#    0   = incorrect / contradicts gold
#    0.5 = partially correct (right direction, missing key detail or wrong value)
#    1   = fully correct (all key facts match)
# ══════════════════════════════════════════════════════════════════════════════
ANSWER_CORRECTNESS_PROMPT = """\
You are an expert grader for a scientific RAG benchmark in building environment and
materials science. Compare the GENERATED ANSWER to the GOLD ANSWER and assign a
correctness score.

SCORING RUBRIC
--------------
1 (Fully correct):
  - All key facts from the gold answer are present in the generated answer.
  - Numeric values match within ±5% AND same physical quantity (wt% ≠ vol%).
  - The generated answer may include additional (non-contradictory) detail.

0.5 (Partially correct):
  - The generated answer captures the right topic / direction but is missing one key
    detail, or has a numeric value outside ±5%, or uses a wrong but related unit.

0 (Incorrect):
  - The generated answer contradicts the gold answer, or is completely off-topic,
    or says "I don't know" when the gold answer exists.

UNIT EQUIVALENCE (count as same):
  0.81 GPa = 810 MPa | 1 W/m·K = 1 W/(m·K) | same value in different SI prefixes.

OUTPUT FORMAT (strict JSON, no markdown fences):
{{
  "score": 0 | 0.5 | 1,
  "reasoning": "one or two sentences explaining the score"
}}

=== QUESTION ===
{question}

=== GOLD ANSWER ===
{gold_answer}

=== GENERATED ANSWER ===
{generated_answer}
"""

# ══════════════════════════════════════════════════════════════════════════════
# 3. Context Precision (per retrieved doc)
#    Input: question + single context doc
#    Output: {"relevant": true | false, "score": 0.0–1.0, "reason": "..."}
#
#    Relevant = this doc would meaningfully help answer the question.
#    Used to compute Precision@k = fraction of retrieved docs that are relevant.
# ══════════════════════════════════════════════════════════════════════════════
CONTEXT_PRECISION_PROMPT = """\
You are a relevance assessor for a scientific retrieval benchmark in building
environment and materials science.

TASK
----
Given a QUESTION and one RETRIEVED DOCUMENT, decide whether the document is
RELEVANT to answering the question.

RELEVANT means: the document contains information that directly helps answer the
question. The information must be specific (not just background) and must relate to
the same material, property, method, or phenomenon as the question.

NOT RELEVANT: generic introductory statements, acknowledgements, bibliography text,
or content about a completely different material/topic.

OUTPUT FORMAT (strict JSON, no markdown fences):
{{
  "relevant": true | false,
  "score": 0.0,
  "reason": "one short sentence"
}}

For `score`: use 1.0 if fully relevant, 0.5 if partially relevant, 0.0 if not relevant.
Set `relevant` = true when score >= 0.5.

=== QUESTION ===
{question}

=== RETRIEVED DOCUMENT ===
Source: {source}
Type: {element_type}

{doc_text}
"""

# ══════════════════════════════════════════════════════════════════════════════
# 4. Context Recall (end-to-end)
#    Input: gold_answer + context_docs (everything retrieved)
#    Output: {"score": 0.0–1.0, "missing_info": [...], "reasoning": "..."}
#
#    Recall = fraction of gold_answer facts that appear in retrieved context.
#    High recall means the correct information WAS retrieved (even if not top-1).
# ══════════════════════════════════════════════════════════════════════════════
CONTEXT_RECALL_PROMPT = """\
You are an expert grader for a scientific RAG benchmark in building environment and
materials science.

TASK
----
Given a GOLD ANSWER and a set of RETRIEVED CONTEXT DOCUMENTS, determine what fraction
of the key facts in the gold answer can be found (directly or paraphrastically) in the
context.

PROCEDURE
---------
1. Identify each distinct factual CLAIM in the gold answer.
2. For each claim, check if it appears in any of the retrieved documents.
3. score = claims_found_in_context / total_claims.
   If gold has 0 claims, score = 1.0.

OUTPUT FORMAT (strict JSON, no markdown fences):
{{
  "claims": [
    {{"claim": "...", "found": true | false}}
  ],
  "score": 0.0,
  "missing_info": ["claim that was NOT found", ...],
  "reasoning": "one short sentence"
}}

=== GOLD ANSWER ===
{gold_answer}

=== RETRIEVED CONTEXT DOCUMENTS ===
{context}
"""
