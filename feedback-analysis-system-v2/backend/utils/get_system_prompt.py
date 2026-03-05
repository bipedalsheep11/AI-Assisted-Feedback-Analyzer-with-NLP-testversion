# backend/utils/get_system_prompt.py
# ─────────────────────────────────────────────────────────────────────────────
# System Prompt Builder
#
# The system prompt is the first message the LLM receives in every API call.
# It defines the model's persona, expertise, output format requirements, and
# any domain-specific context that should inform all downstream responses.
#
# Why a dedicated module?
#   All four analysis modules (labelling, sentiment, themes, insights) use the
#   same system prompt.  Centralising it here means you only need to update one
#   file if you want to change the model's persona or inject new context.
# ─────────────────────────────────────────────────────────────────────────────


# What this function does:
#   Builds the system prompt string that is passed to every LLM call.
#   Optionally embeds an extracted program document and program name so the
#   model's analysis is grounded in the actual training context.
#
# Parameters:
#   document_text  (str) — the full text extracted from the uploaded program
#                          context document (PDF, DOCX, or TXT).
#                          Pass an empty string "" if no document was uploaded.
#   program_name   (str) — the name of the training program being evaluated,
#                          e.g. "Leadership Development Series — Q4 2024".
#                          Pass an empty string "" if not provided.
#
# Returns:
#   str — the complete system prompt to pass as the "system" role message

def get_system_prompt(document_text: str = "", program_name: str = "") -> str:

    # Build optional context blocks only if the relevant data was provided.
    # We insert them conditionally so the prompt doesn't have empty sections.
    program_line = (
        f"\nYou are evaluating the programme: {program_name}.\n"
        if program_name.strip()
        else ""
    )

    # If a document was provided, include it as grounding context.
    # We cap the document at 6000 characters to avoid exceeding token limits.
    # 6000 chars ≈ 1500 tokens, which is a safe size for context injection.
    document_section = ""
    if document_text and document_text.strip():
        truncated_doc = document_text.strip()[:6000]
        document_section = f"""

PROGRAMME CONTEXT DOCUMENT (use this to ground your analysis):
{'─' * 60}
{truncated_doc}
{'─' * 60}
"""

    system_prompt = f"""You are a specialist in training program evaluation and organizational learning analytics.
Your role is to analyze post-program survey data — both numerical ratings and free-text responses — and produce precise, data-grounded characterizations of respondent clusters.
{program_line}
Your primary objective is to identify what meaningfully distinguishes each group of respondents from one another, based on how they rated different aspects of the training and what they expressed in their written responses.

When analyzing a cluster, apply the following standards:

1. GROUND EVERY CLAIM IN THE DATA. Do not infer attitudes, motivations, or profiles that are not directly supported by the ratings or text provided. If the data is ambiguous, reflect that ambiguity rather than inventing a clean narrative.

2. PRIORITIZE CONTRAST. You will always be given data for all clusters alongside the cluster you are labeling. Use that comparison to ensure each label and profile is meaningfully distinct — avoid generic descriptions that could apply to multiple clusters.

3. BE SPECIFIC ABOUT TRAINING DIMENSIONS. When identifying what drove a cluster's responses, name the specific aspects of the training involved (e.g. facilitator delivery, content pacing, practical relevance, material clarity) rather than describing sentiment in isolation.

4. USE PRECISE, PROFESSIONAL LANGUAGE. Labels should be concise and descriptive. Summaries should read like analyst notes, not marketing copy. Avoid filler phrases like "overall positive experience" or "room for improvement" unless they are substantially qualified.

5. RETURN ONLY VALID JSON. Your entire response must be a single JSON object matching the structure provided in the user message. Do not include any text before or after the JSON. Do not use markdown formatting, code fences, or commentary of any kind.
{document_section}"""

    return system_prompt
