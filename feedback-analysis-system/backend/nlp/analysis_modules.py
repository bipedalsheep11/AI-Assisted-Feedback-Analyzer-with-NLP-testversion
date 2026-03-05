# backend/nlp/analysis_modules.py
# ─────────────────────────────────────────────────────────────────
# Three AI-powered analysis stages:
#   1. label_clusters     — characterise each cluster with the LLM
#   2. analyze_sentiment  — per-respondent sentiment + urgent flags
#   3. cluster_themes     — thematic grouping of responses
#   4. extract_insights   — actionable improvement suggestions
# ─────────────────────────────────────────────────────────────────

import json
import pandas as pd
from .llm_client import call_llm_with_retry, parse_llm_json


# ════════════════════════════════════════════════════════════════
# 1. CLUSTER LABELLING
# ════════════════════════════════════════════════════════════════

def label_cluster_with_llm(
    cluster_id:         int,
    formatted_ratings:  str,
    formatted_responses:str,
    all_clusters_table: list[str],
    system_prompt:      str,
) -> dict:
    """
    Ask the LLM to characterise a single cluster.

    We provide two sources of context:
      - all_clusters_table: ratings and responses for ALL clusters,
        so the model can make comparisons and ensure each label
        is meaningfully distinct from the others.
      - formatted_ratings + formatted_responses: the specific data
        for the cluster being labelled.

    Parameters
    ----------
    cluster_id          : int — index of the cluster to label
    formatted_ratings   : str — average ratings for this cluster
    formatted_responses : str — sample text responses for this cluster
    all_clusters_table  : list[str] — [all_ratings, all_responses]
    system_prompt       : str — analyst role + output constraints

    Returns
    -------
    dict with keys:
      label, respondent_profile, key_drivers, distinguishing_features
    """
    user_prompt = f"""You are analyzing post-program training evaluation data.
Your task is to characterize the respondents in Cluster {cluster_id} —
describe who they are as a group, based on their ratings and written responses.

Characterization must focus on:
- Their overall satisfaction and engagement with the training
- Which aspects they responded most strongly to (facilitator quality,
  pacing, content relevance, practical applicability)
- Consistent patterns in what they valued or found lacking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT — All clusters (for comparison):

All clusters — average ratings:
{all_clusters_table[0]}

All clusters — sample text responses:
{all_clusters_table[1]}

Treat the responses as answers to the column header questions.
Always reference what question the respondent was answering.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOU ARE LABELLING: Cluster {cluster_id}

Cluster {cluster_id} — average ratings:
{formatted_ratings}

Cluster {cluster_id} — sample text responses:
{formatted_responses}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a JSON object. No markdown. No code fences. No commentary.

{{
  "label": "A short 2-4 word phrase that distinctly characterizes this group",
  "respondent_profile": "One to two sentences describing who these respondents appear to be — their overall disposition toward the training, and what most shaped their experience",
  "key_drivers": ["The 2-4 specific training aspects that most explain this cluster's ratings and responses, drawn directly from the data"],
  "distinguishing_features": "One sentence explaining what makes this cluster different from the other clusters shown above"
}}"""

    raw    = call_llm_with_retry(system_prompt, user_prompt, max_tokens=600)
    result = parse_llm_json(raw)

    if not result:
        # Return a minimal valid dict so the pipeline does not crash
        return {
            "label": f"Cluster {cluster_id}",
            "respondent_profile": "Could not generate profile — LLM response could not be parsed.",
            "key_drivers": [],
            "distinguishing_features": "Parse error — review raw LLM output.",
            "_raw": raw,
        }
    return result


def label_all_clusters(
    best_k:             int,
    labeled_df:         pd.DataFrame,
    likert_cols:        list[str],
    text_cols:          list[str],
    all_clusters_table: list[str],
    system_prompt:      str,
) -> dict:
    """
    Label all clusters by calling label_cluster_with_llm() for each one.

    Parameters
    ----------
    best_k              : int — number of clusters
    labeled_df          : pd.DataFrame — data with 'cluster' column
    likert_cols         : list[str] — Likert column names
    text_cols           : list[str] — text column names
    all_clusters_table  : list[str] — context for comparison
    system_prompt       : str — LLM system prompt

    Returns
    -------
    dict keyed by cluster index (str): "0", "1", …
    """
    from .format_responses import generate_formatted_responses

    labels = {}
    for cluster_id in range(best_k):
        print(f"  Labelling Cluster {cluster_id}…")
        formatted = generate_formatted_responses(
            labeled_df, cluster_id, likert_cols, text_cols
        )
        labels[str(cluster_id)] = label_cluster_with_llm(
            cluster_id          = cluster_id,
            formatted_ratings   = formatted[0],
            formatted_responses = formatted[1],
            all_clusters_table  = all_clusters_table,
            system_prompt       = system_prompt,
        )
    return labels


# ════════════════════════════════════════════════════════════════
# 2. SENTIMENT ANALYSIS
# ════════════════════════════════════════════════════════════════

def analyze_sentiment(
    labeled_df,
    best_k:        int,
    likert_cols:   list,
    text_cols:     list,
    system_prompt: str,
) -> dict:
    """
    Classify the sentiment of every respondent, processing one cluster at a time.

    WHY PER-CLUSTER INSTEAD OF ALL AT ONCE:
    ------------------------------------------
    When all clusters are sent in a single prompt, the LLM output can easily
    exceed the max_tokens ceiling. Roughly 80-120 tokens per respondent means
    40 respondents alone needs ~5,000 output tokens. A truncated response
    produces malformed JSON that cannot be parsed, which is why total_classified
    was coming back as 0. Processing one cluster at a time keeps each response
    well within limits and makes failures isolated and easier to debug.

    Parameters
    ----------
    labeled_df    : pd.DataFrame  -- full survey DataFrame with a 'cluster' column
    best_k        : int           -- total number of clusters
    likert_cols   : list[str]     -- names of Likert-scale columns
    text_cols     : list[str]     -- names of free-text response columns
    system_prompt : str           -- LLM system prompt (includes programme context)

    Returns
    -------
    dict with keys:
        total_classified  (int)           -- total respondents classified across all clusters
        results           (list of dicts) -- one dict per respondent
        cluster_summary   (dict)          -- sentiment % breakdown per cluster key
        _errors           (list)          -- any per-cluster parse failures, for debugging
    """

    # These will be accumulated across all cluster iterations.
    all_results = []
    all_errors  = []

    for cluster_id in range(best_k):

        # generate_formatted_responses() returns a 2-element list:
        #   [0] -- average Likert ratings for this cluster as a readable string
        #   [1] -- respondent free-text responses formatted as a CSV-like string
        # Calling it per-cluster means the LLM receives one small, focused
        # block of text instead of one enormous combined block.
        cluster_data      = generate_formatted_responses(labeled_df, cluster_id, likert_cols, text_cols)
        avg_ratings_block = cluster_data[0]
        responses_block   = cluster_data[1]

        # Count respondents in this cluster so we can tell the model the
        # exact number it must return. This acts as a self-check prompt:
        # if the model returns fewer, we know the output was truncated.
        n_in_cluster = len(labeled_df[labeled_df["cluster"] == cluster_id])

        user_prompt = f"""Analyze the sentiment of EVERY respondent in Cluster {cluster_id}.

There are exactly {n_in_cluster} respondents in this cluster.
You MUST return exactly {n_in_cluster} result objects -- one per respondent row. Do not skip any.

Average Likert ratings for Cluster {cluster_id}:
{avg_ratings_block}

Respondent responses (CSV format -- first column is respondent number, remaining columns are answers):
{responses_block}

Return ONLY a valid JSON object. No markdown. No code fences. No text before or after the JSON.

{{
  "cluster": {cluster_id},
  "results": [
    {{
      "cluster": {cluster_id},
      "respondent_id": "<the respondent number from the first CSV column>",
      "sentiment": "<positive | negative | neutral | mixed>",
      "confidence": "<high | medium | low>",
      "flag_urgent": <true | false>,
      "flag_reason": "<one sentence if flag_urgent is true, otherwise null>",
      "key_phrases": ["<3-6 word phrase>", "<3-6 word phrase>", "<3-6 word phrase>"]
    }}
  ]
}}

SENTIMENT DEFINITIONS:
- positive: respondent expresses satisfaction, appreciation, or perceived benefit
- negative: respondent expresses dissatisfaction, frustration, or significant problems
- neutral:  respondent states facts or observations without clear positive or negative tone
- mixed:    respondent expresses both positive and negative sentiments in the same response

FLAG URGENT -- set flag_urgent to true if ANY of the following apply:
- Strong dissatisfaction that would damage programme credibility if repeated
- A logistical failure that prevented meaningful participation
- A safety, health, or welfare concern raised by the respondent
- An explicit statement that the respondent would not recommend or return to this programme

KEY PHRASES: extract up to 3 short phrases (3-6 words each) capturing the core of each response.

Every row in the CSV above must produce exactly one result object in the results array."""

        # Set max_tokens per cluster.
        # Each result object is roughly 100 tokens. We give a 2x safety buffer
        # to handle verbose flag_reason fields and longer key phrases.
        # The floor of 1000 protects tiny single-cluster datasets.
        tokens_needed = max(1000, n_in_cluster * 200)

        raw    = call_llm_with_retry(system_prompt, user_prompt, max_tokens=tokens_needed)
        parsed = parse_llm_json(raw)

        if not parsed:
            # Record the failure and move on to the next cluster.
            # We do NOT raise here -- one bad cluster should not block
            # the rest of the pipeline from completing.
            all_errors.append(
                f"Cluster {cluster_id}: could not parse JSON. "
                f"Raw output (first 500 chars): {str(raw)[:500]}"
            )
            continue

        cluster_results = parsed.get("results") or []

        # Warn if fewer results came back than expected.
        if len(cluster_results) < n_in_cluster:
            all_errors.append(
                f"Cluster {cluster_id}: expected {n_in_cluster} results, "
                f"got {len(cluster_results)}. Response may have been truncated."
            )

        # Guarantee cluster field is an int on every result, in case the
        # model omitted it or returned a string instead.
        for r in cluster_results:
            r["cluster"] = cluster_id

        all_results.extend(cluster_results)

    # ── Rebuild cluster_summary from actual results ──────────────────
    # We never trust the LLM to compute percentages correctly.
    # We count sentiment occurrences ourselves and convert to percentages.
    cluster_counts = {}
    for r in all_results:
        ckey = str(r.get("cluster", "unknown"))
        if ckey not in cluster_counts:
            cluster_counts[ckey] = {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}
        sentiment = r.get("sentiment", "neutral")
        if sentiment in cluster_counts[ckey]:
            cluster_counts[ckey][sentiment] += 1

    cluster_summary = {}
    for ckey, counts in cluster_counts.items():
        total_in_cluster = sum(counts.values())
        if total_in_cluster == 0:
            cluster_summary[ckey] = {"positive": 0, "neutral": 0, "negative": 0, "mixed": 0}
        else:
            cluster_summary[ckey] = {
                s: round((n / total_in_cluster) * 100)
                for s, n in counts.items()
            }

    return {
        "total_classified": len(all_results),
        "results":          all_results,
        "cluster_summary":  cluster_summary,
        "_errors":          all_errors,
    }


# ════════════════════════════════════════════════════════════════
# 3. THEMATIC CLUSTERING
# ════════════════════════════════════════════════════════════════

def cluster_themes(
    all_clusters_responses: list[str],
    system_prompt:          str,
    predefined_themes:      list[str] | None = None,
) -> dict:
    """
    Group responses into recurring themes.

    Two modes:
      - Predefined: caller supplies theme labels; model assigns each
        response to the closest label (or "Other").
      - Discovery:  model identifies 3-8 themes from the data itself.

    Parameters
    ----------
    all_clusters_responses : list[str] — [ratings_block, responses_block]
    system_prompt          : str
    predefined_themes      : list[str] | None — if provided, uses these

    Returns
    -------
    dict with keys:
      themes (list of theme dicts), coded_responses (list of assignments)
    """
    if predefined_themes:
        theme_list = "\n".join(f"- {t}" for t in predefined_themes)
        theme_instruction = f"""Assign each response to ONE of these predefined themes.
If a response fits none, assign it to "Other" and note the actual theme it represents.

PREDEFINED THEMES:
{theme_list}"""
    else:
        theme_instruction = """Identify 4-8 recurring themes from the responses.
Name each theme as a short noun phrase (e.g. "Facilitator Clarity", "Module Pacing").
Do not create a unique theme per response — look for patterns across respondents."""

    user_prompt = f"""You are performing thematic analysis on post-training evaluation responses.

{theme_instruction}

RESPONDENT RESPONSES:
{all_clusters_responses[1]}

Return ONLY a JSON object. No markdown. No code fences.

{{
  "themes": [
    {{
      "name": "Theme Name",
      "count": <integer — number of responses assigned to this theme>,
      "description": "One sentence characterising what respondents in this theme are saying",
      "clusters": [<list of cluster IDs where this theme appears>]
    }}
  ],
  "coded_responses": [
    {{
      "respondent_id": "<id>",
      "cluster": <integer>,
      "theme": "Theme Name",
      "fit": "<strong | moderate | weak>",
      "supporting_quote": "<5-10 word phrase from the response justifying the assignment>"
    }}
  ]
}}

THEME FIT DEFINITIONS:
- strong:   response is almost entirely about this theme
- moderate: theme is present but response also covers other topics
- weak:     only tangential — flag for manual review"""

    raw    = call_llm_with_retry(system_prompt, user_prompt, max_tokens=2500)
    result = parse_llm_json(raw)

    if not result:
        return {
            "themes": [],
            "coded_responses": [],
            "_parse_error": raw,
        }
    return result


# ════════════════════════════════════════════════════════════════
# 4. ACTIONABLE INSIGHT EXTRACTION
# ════════════════════════════════════════════════════════════════

def extract_actionable_insights(
    all_clusters_responses: list[str],
    system_prompt:          str,
) -> dict:
    """
    Extract concrete, actionable improvement suggestions from all
    clusters simultaneously.

    Analysing all clusters together allows the model to identify
    whether a suggestion is isolated (one cluster only) or systemic
    (multiple clusters), which determines priority.

    Parameters
    ----------
    all_clusters_responses : list[str] — [ratings_block, responses_block]
    system_prompt          : str

    Returns
    -------
    dict with keys:
      total_insights, insights (list), priority_summary (dict)
    """
    user_prompt = f"""Extract concrete, actionable improvement suggestions from these
post-training evaluation responses.

An actionable insight must be:
- Grounded in something a participant explicitly said or clearly implied
- Specific enough to guide a real decision (not vague praise or complaint)
- Distinct from other insights — do not repeat the same suggestion

Do NOT include:
- General positive feedback with no improvement implication
- Vague statements like "make it better"
- Observations that describe a problem without any implied solution

EVALUATION RESPONSES BY CLUSTER:
{all_clusters_responses[1]}

Return ONLY a JSON object. No markdown. No code fences.

{{
  "total_insights": <integer>,
  "insights": [
    {{
      "id": "INS-001",
      "priority": "<high | medium | low>",
      "category": "<Facilitator | Content | Pacing | Logistics | Materials | Assessment | Follow-up | Other>",
      "insight": "<one clear actionable recommendation sentence — written as a recommendation, e.g. 'Provide printed handouts…'>",
      "source_clusters": [<cluster_id>, ...],
      "evidence": "<paraphrased summary of participant comments supporting this insight>",
      "breadth": "<isolated | recurring | widespread>"
    }}
  ],
  "priority_summary": {{
    "high":   "<one sentence summarising the high-priority theme>",
    "medium": "<one sentence summarising the medium-priority theme>",
    "low":    "<one sentence summarising the low-priority theme>"
  }}
}}

PRIORITY:
- high:   issue blocks core learning outcomes
- medium: affects perceived quality but does not block learning
- low:    enhancement — useful but not urgent

BREADTH:
- isolated:   one cluster only
- recurring:  two clusters
- widespread: three or more clusters

Sort: priority descending (high first), then breadth (widespread before isolated)."""

    raw    = call_llm_with_retry(system_prompt, user_prompt, max_tokens=3000)
    result = parse_llm_json(raw)

    if not result:
        return {
            "total_insights": 0,
            "insights": [],
            "priority_summary": {"high": "", "medium": "", "low": ""},
            "_parse_error": raw,
        }
    return result
