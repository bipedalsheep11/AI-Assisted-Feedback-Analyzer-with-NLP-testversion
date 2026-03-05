import json
from groq import Groq

module_dir = os.path.abspath('../nlp')
sys.path.append(module_dir)
import llm_client.call_llm_with_retry as call_llm
# This function classifies the sentiment of a single participant's open-ended
# response and flags it if it requires urgent attention.
#
# We analyze one response at a time (not batched) because:
# - It produces more accurate per-response labels
# - It makes it easy to attach the result back to the correct respondent row
# - It avoids the model conflating sentiments across multiple responses
#
# Input:
#   response_text   (str)   — a single participant's open-ended text response
#   respondent_id   (str)   — an identifier so we can match output back to the row
#                             e.g. a row index or anonymized ID like "R042"
# Output:
#   A dictionary with keys: respondent_id, sentiment, confidence, flag_urgent,
#   flag_reason, key_phrases

def analyze_sentiment(all_clusters_responses, system_prompt):

    # The user prompt describes the specific task for this one response.
    # We embed the response inside a clearly labeled block so the model
    # does not confuse instructions with participant text.
    user_prompt = f"""Analyze the sentiment of each respondent based on a CSV Format text of responses per Cluster.

                Summary of the average ratings per cluster: {all_clusters_responses[0]}
                CSV Format of the Responses: {all_clusters_responses[1]}
                
                
                Classify the sentiment and return a JSON object with exactly these keys:
                
                {{
                  "cluster": "Cluster number",
                  "respondent_id": "respondent number",
                  "sentiment": "<positive | negative | neutral | mixed>",
                  "confidence": "<high | medium | low>",
                  "flag_urgent": <true | false>,
                  "flag_reason": "<if flag_urgent is true, explain in one sentence why this needs immediate attention — otherwise null>",
                  "key_phrases": ["<short phrase 1>", "<short phrase 2>", "<short phrase 3>"]
                }}
                
                SENTIMENT DEFINITIONS:
                - positive: respondent expresses satisfaction, appreciation, or benefit from the training
                - negative: respondent expresses dissatisfaction, frustration, or identifies significant problems
                - neutral: respondent states facts or observations without clear positive or negative tone
                - mixed: respondent expresses both positive and negative sentiments within the same response
                
                FLAG URGENT RULES — set flag_urgent to true if ANY of the following apply:
                - Respondent expresses strong dissatisfaction that would damage the program's credibility if repeated
                - Respondent reports a logistical failure that prevented meaningful participation (e.g. no materials, technical failure, inaccessible venue)
                - Respondent's comment suggests a safety, health, or welfare concern
                - Respondent explicitly states they would not recommend or return to this program
                
                KEY PHRASES: extract up to 3 short phrases (3-6 words each) that best capture the core of this response."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        temperature=0.1,  # Low temperature = more consistent, less creative outputs
                          # For classification tasks, consistency matters more than creativity
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ]
    )

    raw_text = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        # Return a structured fallback so the pipeline does not crash
        result = {
            "respondent_id": respondent_id,
            "sentiment": null,
            "confidence": "low",
            "flag_urgent": False,
            "flag_reason": None,
            "key_phrases": [],
            "parse_error": raw_text  # preserve raw output for manual review
        }
        
    result = call_llm(system_prompt, user_prompt)

    return result


# ── Batch wrapper ────────────────────────────────────────────────
# This function runs analyze_sentiment() across all rows in the DataFrame.
#
# Input:
#   labeled_df      (pd.DataFrame)  — your survey DataFrame with a 'cluster' column
#   text_columns    (list of str)   — open-ended response column names
# Output:
#   A list of sentiment result dicts, one per respondent