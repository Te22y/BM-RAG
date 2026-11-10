"""
qa_gen_prompts.py
=================

LLM prompts used to build the test set:
  1. QA_GEN_PROMPT          for GPT-4o: a self-contained QA pair from a chunk
  2. REVERSE_ANSWER_PROMPT  for gpt-5.2: answer without context (filters out easy questions)
  3. ANSWER_MATCH_PROMPT    for gpt-5.2: compare a candidate answer with gold_answer

Every prompt asks for strict JSON so replies can be parsed.
"""

QA_GEN_PROMPT = """\
You are a meticulous test-set author for a scientific RAG benchmark in the domain of
**building environment, building materials, thermal insulation, phase change materials,
and concrete technology**.

You will be given ONE chunk of text extracted from a research paper. Your task is to
construct **1 to 2 high-quality question/answer pairs** that satisfy ALL of the
following constraints:

CONSTRAINTS
-----------
1. The QUESTION must be **stand-alone**: a reader who has never seen this chunk must
   be able to understand it. Do NOT use phrases such as "in this paper", "as shown in
   the table above", "the authors", "the figure", or any other reference that requires
   external context. Always name the material, method, or concept explicitly.

2. The ANSWER must be **fully derivable from the chunk alone** — no outside knowledge
   needed. The answer should be concise (numeric value + unit, a short phrase, or 1–3
   sentences for mechanism questions) and **must not** contain hedges or speculation.

3. The QUESTION must be **non-trivial**: it should require RAG to answer correctly.
   Avoid questions that a general-knowledge LLM could answer without consulting the
   paper (e.g., "What is the chemical formula of CaCO3?", "What does PCM stand for?").
   Prefer questions tied to **specific experimental results, material compositions,
   measured values, or mechanisms** described in the chunk.

4. The QUESTION must NOT contain the answer verbatim. Do not leak the numeric value
   or key phrase into the question.

5. Also output:
   - `evidence_span`: the exact substring of the chunk (≤ 300 chars) that supports
     the answer. Copy verbatim — no paraphrase.
   - `question_type`: one of {{factoid, numeric, comparison, mechanism, definition}}.
   - `category`: a short topic tag chosen from this open list (you may also propose
     one new tag if none fits): {{thermal_insulation, envelope_performance,
     phase_change_material, thermal_energy_storage, concrete_technology,
     mechanical_property, durability, mixture_design, microstructure, mechanism,
     measurement_method, simulation, other}}.

REJECTION
---------
If the chunk does not contain enough specific, factual content to build a non-trivial
stand-alone QA pair (e.g., it is a generic background paragraph, an acknowledgement,
a journal header, or pure boilerplate), output an EMPTY JSON array `[]` and stop.

OUTPUT FORMAT (STRICT)
----------------------
Output ONLY a JSON array. No markdown fences, no commentary. Each element:

  {{
    "question": "...",
    "gold_answer": "...",
    "evidence_span": "...",
    "question_type": "numeric",
    "category": "thermal_insulation"
  }}

Generate **at most 2** items. Quality over quantity — one excellent pair beats two
mediocre ones.

CHUNK
-----
Paper filename: {paper_filename}
Page number:    {page_number}
Element type:   {element_type}

----- CHUNK CONTENT START -----
{chunk_text}
----- CHUNK CONTENT END -----
"""


QA_GEN_SUMMARY_PROMPT = """\
You are a meticulous test-set author for a scientific RAG benchmark in the domain of
**building environment, building materials, thermal insulation, phase change materials,
and concrete technology**.

You will be given an **AI-generated summary** of a {element_type} element extracted
from a research paper. The original element may be a table of experimental data or an
image (figure/chart). Your task is to construct **1 to 2 high-quality question/answer
pairs** that satisfy ALL of the following constraints:

CONSTRAINTS
-----------
1. The QUESTION must be **stand-alone**: a reader who has never seen this chunk must
   be able to understand it. Do NOT use phrases such as "in this paper", "as shown in
   the table above", "the authors", "the figure", or any other reference that requires
   external context. Always name the material, method, or concept explicitly.

2. The ANSWER must be **fully derivable from the summary alone** — no outside knowledge
   needed. The answer should be concise (numeric value + unit, a short phrase, or 1–3
   sentences for mechanism questions) and **must not** contain hedges or speculation.

3. The QUESTION must be **non-trivial**: it should require RAG to answer correctly.
   Avoid questions that a general-knowledge LLM could answer without consulting the
   paper (e.g., "What is the chemical formula of CaCO3?", "What does PCM stand for?").
   Prefer questions tied to **specific experimental results, material compositions,
   measured values, or mechanisms** described in the summary.

4. The QUESTION must NOT contain the answer verbatim. Do not leak the numeric value
   or key phrase into the question.

5. Also output:
   - `evidence_span`: the exact substring of the **summary** (≤ 300 chars) that
     supports the answer. Copy verbatim — no paraphrase.
   - `question_type`: one of {{factoid, numeric, comparison, mechanism, definition}}.
   - `category`: a short topic tag chosen from this open list (you may also propose
     one new tag if none fits): {{thermal_insulation, envelope_performance,
     phase_change_material, thermal_energy_storage, concrete_technology,
     mechanical_property, durability, mixture_design, microstructure, mechanism,
     measurement_method, simulation, other}}.

REJECTION
---------
If the summary does not contain enough specific, factual content to build a non-trivial
stand-alone QA pair (e.g., it is too vague, generic, or lacks concrete data),
output an EMPTY JSON array `[]` and stop.

OUTPUT FORMAT (STRICT)
----------------------
Output ONLY a JSON array. No markdown fences, no commentary. Each element:

  {{
    "question": "...",
    "gold_answer": "...",
    "evidence_span": "...",
    "question_type": "numeric",
    "category": "thermal_insulation"
  }}

Generate **at most 2** items. Quality over quantity — one excellent pair beats two
mediocre ones.

SUMMARY
-------
Paper filename: {paper_filename}
Page number:    {page_number}
Element type:   {element_type}

----- SUMMARY CONTENT START -----
{chunk_text}
----- SUMMARY CONTENT END -----
"""


REVERSE_ANSWER_PROMPT = """\
You will be asked a single technical question about building materials or building
environment research. Answer **from your own knowledge only** — no external context
is provided.

If you do not know the answer with confidence, reply exactly: `I don't know.`
Otherwise reply with the most concise correct answer (numeric value + unit, or a short
phrase, or one sentence). Do NOT add explanations.

Question: {question}
"""


ANSWER_MATCH_PROMPT = """\
You are an impartial grader for a scientific Q&A benchmark.

Given a QUESTION, a GOLD answer (the ground truth from a research paper), and a
CANDIDATE answer (produced by a language model), decide whether the candidate answer
is **substantively correct** — i.e., it conveys the same factual information as the
gold answer.

Scoring rules:
- For numeric answers, the candidate must be within ±5% of the gold value AND use the
  same physical quantity. Different units that convert exactly are OK (e.g., 0.81 GPa
  vs 810 MPa). Different quantities (e.g., wt% vs vol%) are NOT OK.
- For factoid/definition answers, the candidate must name the same entity or concept.
  Synonyms and reorderings are OK.
- For mechanism/comparison answers, the candidate must capture the same direction and
  the same causal/comparative relation.
- "I don't know" is NEVER a match.

Output STRICT JSON, no commentary:

  {{"match": true | false, "reason": "<one short sentence>"}}

QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

CANDIDATE ANSWER:
{candidate_answer}
"""
