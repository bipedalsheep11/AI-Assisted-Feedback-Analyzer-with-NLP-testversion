# backend/nlp/analysis_modules.py
# ─────────────────────────────────────────────────────────────────────────────
# LLM Analysis Modules
#
# This file contains all four LLM-powered analysis functions:
#
#   1. label_all_clusters()        — give each cluster a human-readable name
#      and profile based on its ratings and responses.
#
#   2. analyze_sentiment()         — classify each respondent's text as
#      positive / negative / neutral / mixed, and flag urgent responses.
#
#   3. cluster_themes()            — identify recurring themes across all
#      responses, either from predefined labels or discovered automatically.
#
#   4. extract_actionable_insights() — surface specific improvement suggestions
#      that participants embedded in their comments.
#
# All four functions call call_llm_with_retry() from llm_client.py and expect
# the LLM to return structured JSON.  The parse_llm_json() helper handles
# cleaning up the response before parsing.
#
# Dependencies (installed with the project requirements):
#   groq, ollama, python-dotenv  (via llm_client.py)
# ─────────────────────────────────────────────────────────────────────────────

import json
import pandas as pd
from .llm_client import call_llm_with_retry, parse_llm_json
from .format_responses import generate_formatted_responses


# ════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Cluster Labelling
# ════════════════════════════════════════════════════════════════════════════

# What this function does:
#   Asks the LLM to generate a short, meaningful label and a respondent profile
#   for a single cluster.  The LLM receives both the cluster's own data AND
#   the data for all other clusters so it can produce labels that are
#   meaningfully distinct from one another.
#
# Parameters:
#   cluster_id         (int)          — which cluster to label (0-based integer)
#   labeled_df         (pd.DataFrame) — full labeled DataFrame
#   likert_cols        (list of str)  — Likert column names
#   text_cols          (list of str)  — text column names
#   all_clusters_table (list of str)  — [all_ratings_str, all_responses_str]
#                                       built by get_all_clusters_table()
#   system_prompt      (str)          — the LLM's role/persona instructions
#
# Returns:
#   dict with keys: label, respondent_profile, key_drivers, distinguishing_features

def label_cluster_with_llm(
    cluster_id: int,
    labeled_df: pd.DataFrame,
    likert_cols: list,
    text_cols: list,
    all_clusters_table: list,
    system_prompt: str,
) -> dict:
    # Generate the formatted data for this specific cluster.
    # cluster_formatted[0] = ratings summary; cluster_formatted[1] = responses table.
    cluster_formatted = generate_formatted_responses(
        labeled_df, cluster_id, likert_cols, text_cols
    )

    user_prompt = f"""You are analyzing post-program training evaluation data.
Your task is to characterize the respondents in Cluster {cluster_id} — describe
who they are as a group, based on how they rated the training and what they wrote.

Focus on:
- Their overall satisfaction and engagement with the training
- Which aspects of the training they responded most strongly to (positively or negatively),
  such as facilitator quality, pacing, content relevance, or practical applicability
- Consistent patterns in what they valued or found lacking

---

FOR CONTEXT — Average ratings and sample responses for ALL clusters are shown below.
Use this to ensure Cluster {cluster_id}'s label is meaningfully distinct from the others.

All clusters — average ratings:
{all_clusters_table[0]}

All clusters — sample text responses:
{all_clusters_table[1]}

---

YOU ARE LABELING: Cluster {cluster_id}

Cluster {cluster_id} — average ratings:
{cluster_formatted[0]}

Cluster {cluster_id} — sample text responses:
{cluster_formatted[1]}

---

Return ONLY a JSON object. No markdown, no code fences, no commentary.
Use this exact structure:

{{
  "label": "A short 2-4 word phrase that distinctly characterizes this group",
  "respondent_profile": "One to two sentences describing these respondents' overall disposition toward the training and what shaped their experience",
  "key_drivers": ["2-4 specific training aspects that most explain this cluster's ratings and responses"],
  "distinguishing_features": "One sentence explaining what makes this cluster different from the others shown above"
}}"""

    raw = call_llm_with_retry(system_prompt, user_prompt, max_tokens=600)
    result = parse_llm_json(raw, fallback={
        "label":                  f"Cluster {cluster_id}",
        "respondent_profile":     "Could not generate profile.",
        "key_drivers":            [],
        "distinguishing_features": "—",
    })
    return result


# What this function does:
#   Runs label_cluster_with_llm() for every cluster and returns a dictionary
#   mapping cluster IDs (as strings, e.g. "0", "1") to their label objects.
#
# Parameters:
#   best_k             (int)          — total number of clusters
#   labeled_df         (pd.DataFrame) — full labeled DataFrame
#   likert_cols        (list of str)  — Likert column names
#   text_cols          (list of str)  — text column names
#   all_clusters_table (list of str)  — [all_ratings_str, all_responses_str]
#   system_prompt      (str)          — the LLM's role/persona instructions
#
# Returns:
#   dict keyed by string cluster ID: {"0": {label, profile, …}, "1": {…}, …}

def label_all_clusters(
    best_k: int,
    labeled_df: pd.DataFrame,
    likert_cols: list,
    text_cols: list,
    all_clusters_table: list,
    system_prompt: str,
) -> dict:
    labels = {}
    for cluster_id in range(best_k):
        print(f"  Labelling cluster {cluster_id}…")
        label_data = label_cluster_with_llm(
            cluster_id,
            labeled_df,
            likert_cols,
            text_cols,
            all_clusters_table,
            system_prompt,
        )
        # Store using the string version of the ID so Streamlit's session state
        # can serialize it to JSON if needed.
        labels[str(cluster_id)] = label_data
    return labels


# ════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Sentiment Analysis
# ════════════════════════════════════════════════════════════════════════════

# What this function does:
#   Sends all respondents' text responses (grouped by cluster) to the LLM,
#   which classifies each one as positive / negative / neutral / mixed,
#   assigns a confidence level, and flags responses that need urgent attention.
#
# Parameters:
#   all_clusters_responses (list) — [all_ratings_str, all_responses_str]
#   system_prompt          (str)  — the LLM's role/persona instructions
#
# Returns:
#   dict with keys:
#     results         — list of per-respondent dicts
#     cluster_summary — dict mapping cluster_id to % positive/negative/neutral
#     total_classified — int

def analyze_sentiment(
    all_clusters_responses: list,
    system_prompt: str,
) -> dict:

    
    user_prompt = f"""Analyze the sentiment of each respondent based on the CSV-formatted responses below.

    CSV of all responses (format: Respondent, [text columns]):
    {all_clusters_responses[1]}
    
    For EACH respondent row, classify their sentiment and return a JSON object with this structure:
    
    {{
      "results": [
        {{
          "cluster": <integer cluster number>,
          "respondent_id": "<respondent ID, e.g. R001>",
          "sentiment": "<positive | negative | neutral | mixed>",
          "confidence": "<high | medium | low>",
          "flag_urgent": <true | false>,
          "flag_reason": "<if flag_urgent is true, explain in one sentence — otherwise null>",
          "key_phrases": ["<3–6 word phrase 1>", "<phrase 2>", "<phrase 3>"]
        }}
      ],
      "cluster_summary": {{
        "<cluster_id>": {{
          "positive": <integer percentage, e.g. 60>,
          "negative": <integer percentage>,
          "neutral":  <integer percentage>,
          "mixed":    <integer percentage>
        }}
      }}
    }}
    
    SENTIMENT DEFINITIONS:
    - positive: respondent expresses satisfaction, appreciation, or clear benefit from the training
    - negative: respondent expresses dissatisfaction, frustration, or significant problems
    - neutral:  respondent states facts or observations without clear positive/negative tone
    - mixed:    respondent expresses both positive and negative sentiment in the same response
    
    FLAG URGENT — set flag_urgent to true if ANY apply:
    - Strong dissatisfaction that would damage the program's reputation
    - A logistical failure that prevented meaningful participation
    - A safety, health, or welfare concern
    - Explicit statement that the respondent would not recommend or return
    
    Return ONLY valid JSON. No markdown. No code fences. No commentary."""

    raw = call_llm_with_retry(
        system_prompt,
        user_prompt,
        max_tokens=4000,   # Sentiment analysis across all respondents needs more tokens.
        temperature=0.05,
    )

    result = parse_llm_json(raw, fallback={"results": [], "cluster_summary": {}})

    # Ensure required keys exist even if the LLM omitted them.
    if "results" not in result:
        result["results"] = []
    if "cluster_summary" not in result:
        result["cluster_summary"] = {}

    result["total_classified"] = len(result["results"])
    return result


# ════════════════════════════════════════════════════════════════════════════
# MODULE 3 — Thematic Clustering
# ════════════════════════════════════════════════════════════════════════════

# What this function does:
#   Groups all respondents' text responses into themes.
#   Two modes are supported:
#     - Predefined: the user provides theme names and the LLM assigns responses to them.
#     - Auto-discovery: the LLM identifies themes from the data itself.
#
# Parameters:
#   all_clusters_responses (list)          — [all_ratings_str, all_responses_str]
#   system_prompt          (str)           — the LLM's role/persona instructions
#   predefined_themes      (list | None)   — list of theme name strings, or None for
#                                            auto-discovery
#
# Returns:
#   dict with keys:
#     themes          — list of {name, count, description} objects
#     coded_responses — list of per-respondent theme assignments

def cluster_themes(
    all_clusters_responses: list,
    system_prompt: str,
    predefined_themes: list = None,
) -> dict:
    # Build the theme instruction block depending on which mode is active.
    if predefined_themes:
        # Predefined mode: give the model the category list.
        theme_list = "\n".join(f"- {theme}" for theme in predefined_themes)
        theme_instruction = f"""Assign each response to ONE of the following predefined themes.
If a response clearly fits none of them, assign it to "Other" and note the actual theme it represents.

PREDEFINED THEMES:
{theme_list}"""
    else:
        # Auto-discovery mode: let the model identify patterns.
        theme_instruction = """Identify 3 to 6 recurring themes from the responses below.
Themes should be specific enough to be actionable but broad enough to cover multiple responses.
Do not create a unique theme for every response — look for patterns.
Name each theme as a short noun phrase (e.g. "Facilitator Clarity", "Schedule Management", "Material Relevance")."""

    user_prompt = f"""You are performing thematic analysis on post-training evaluation responses.

{theme_instruction}

Average ratings per cluster:
{all_clusters_responses[0]}

CSV of all responses:
{all_clusters_responses[1]}

Return a JSON object with exactly these keys:

{{
  "themes": [
    {{
      "name":        "<theme name>",
      "count":       <number of responses assigned to this theme>,
      "description": "<one sentence characterizing what respondents in this theme are saying>"
    }}
  ],
  "coded_responses": [
    {{
      "respondent_id":  "<id>",
      "cluster":        <integer>,
      "assigned_theme": "<theme name>",
      "theme_fit":      "<strong | moderate | weak>",
      "supporting_quote": "<5-10 word phrase from the response that justifies the assignment>"
    }}
  ]
}}

THEME FIT DEFINITIONS:
- strong:   the response is almost entirely about this theme
- moderate: this theme is clearly present but the response covers other topics too
- weak:     the response only tangentially relates to this theme

Return ONLY valid JSON. No markdown. No code fences. No commentary."""

    raw = call_llm_with_retry(
        system_prompt,
        user_prompt,
        max_tokens=4000,
        temperature=0.1,
    )

    result = parse_llm_json(raw, fallback={"themes": [], "coded_responses": []})

    if "themes"          not in result: result["themes"]          = []
    if "coded_responses" not in result: result["coded_responses"] = []

    return result


# ════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Actionable Insight Extraction
# ════════════════════════════════════════════════════════════════════════════

# What this function does:
#   Identifies concrete, actionable improvement suggestions that participants
#   embedded in their responses.  Insights that appear across multiple clusters
#   are flagged as systemic (recurring or widespread).
#
# Parameters:
#   all_clusters_responses (list) — [all_ratings_str, all_responses_str]
#   system_prompt          (str)  — the LLM's role/persona instructions
#
# Returns:
#   dict with keys:
#     total_insights   — integer count
#     insights         — list of insight objects
#     priority_summary — dict with high/medium/low summary sentences

def extract_actionable_insights(
    all_clusters_responses: list,
    system_prompt: str,
) -> dict:
    user_prompt = f"""You are extracting actionable improvement suggestions from post-training evaluation responses.

An actionable insight is a concrete, specific suggestion that a training manager can act on.
It must be:
- Grounded in something a participant explicitly said or clearly implied
- Specific enough to guide a real decision (not vague praise or general complaint)
- Distinct from other insights — do not repeat the same suggestion twice

Do NOT include:
- General positive feedback with no improvement implication
- Vague statements like "make it better" or "more training needed"
- Observations that describe a problem without any implied solution

Average ratings per cluster:
{all_clusters_responses[0]}

CSV of all responses by cluster:
{all_clusters_responses[1]}

Return a JSON object with exactly these keys:

{{
  "total_insights": <integer>,
  "insights": [
    {{
      "insight_id":      "INS-001",
      "priority":        "<high | medium | low>",
      "category":        "<Facilitator | Content | Pacing | Logistics | Materials | Assessment | Follow-up | Other>",
      "insight":         "<one clear, actionable recommendation written as a directive sentence>",
      "source_clusters": [<cluster_id>, ...],
      "evidence":        "<paraphrased summary of what participants said — do not quote verbatim>",
      "breadth":         "<isolated | recurring | widespread>"
    }}
  ],
  "priority_summary": {{
    "high":   "<one sentence summarizing the high-priority theme overall>",
    "medium": "<one sentence summarizing the medium-priority theme>",
    "low":    "<one sentence summarizing the low-priority theme>"
  }}
}}

PRIORITY DEFINITIONS:
- high:   affects core learning outcomes or would prevent participants from benefiting from future sessions
- medium: affects comfort, convenience, or perceived quality but does not block learning
- low:    an enhancement that would improve experience but is not urgent

BREADTH DEFINITIONS:
- isolated:   mentioned by respondents in only one cluster
- recurring:  appears in two clusters
- widespread: appears in three or more clusters

Sort insights by priority (high first), then by breadth (widespread before isolated).

Return ONLY valid JSON. No markdown. No code fences. No commentary."""

    raw = call_llm_with_retry(
        system_prompt,
        user_prompt,
        max_tokens=3000,
        temperature=0.1,
    )

    result = parse_llm_json(raw, fallback={
        "total_insights":   0,
        "insights":         [],
        "priority_summary": {"high": "—", "medium": "—", "low": "—"},
    })

    if "total_insights"   not in result: result["total_insights"]   = len(result.get("insights", []))
    if "insights"         not in result: result["insights"]         = []
    if "priority_summary" not in result: result["priority_summary"] = {}

    return result
