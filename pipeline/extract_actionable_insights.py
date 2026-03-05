# This function extracts concrete, actionable improvement suggestions
# embedded in participant responses — across ALL clusters at once.
#
# Why across all clusters:
#   Actionable insights are most useful when the training manager can see
#   whether a suggestion appears in one cluster only (isolated concern)
#   or across multiple clusters (systemic issue requiring priority attention).
#
# Input:
#   labeled_df          (pd.DataFrame)  — full DataFrame with 'cluster' column
#   text_columns        (list of str)   — open-ended response column names
#   cluster_labels      (dict)          — maps cluster_id to human-readable label
#                                         e.g. {0: "Engaged Advocates", 2: "Critical Dissenters"}
#   n_sample_per_cluster (int)          — responses to sample per cluster (default: 10)
# Output:
#   A dictionary with keys: total_insights, insights (list), priority_summary

def extract_actionable_insights(all_clusters_responses, text_cols, cluster_labels, system_prompt, n_sample_per_cluster=10):

    # Build a formatted block of responses grouped by cluster
    # so the model can see which cluster each suggestion came from

    user_prompt = f"""You are extracting actionable improvement suggestions from post-training evaluation responses.

            An actionable insight is a concrete, specific suggestion that a training manager can act on to improve a future program. It must be:
            - Grounded in something a participant explicitly said or clearly implied
            - Specific enough to guide a real decision (not vague praise or complaint)
            - Distinct from other insights — do not repeat the same suggestion twice
            
            Do NOT include:
            - General positive feedback with no improvement implication
            - Vague statements like "make it better" or "more training needed"
            - Observations that describe a problem without any implied solution
            
            EVALUATION RESPONSES BY CLUSTER:
            {all_responses_text}
            
            Return a JSON object with exactly these keys:
            
            {{
              "total_insights": <integer>,
              "insights": [
                {{
                  "insight_id": "INS-001",
                  "priority": "<high | medium | low>",
                  "category": "<Facilitator | Content | Pacing | Logistics | Materials | Assessment | Follow-up | Other>",
                  "insight": "<one clear, actionable sentence written as a recommendation, e.g. 'Provide printed handouts in addition to digital slides to accommodate participants with limited device access'>",
                  "source_clusters": [<cluster_id>, ...],
                  "evidence": "<a paraphrased summary of what participants said that supports this insight — do not quote verbatim>",
                  "breadth": "<isolated | recurring | widespread>"
                }}
              ],
              "priority_summary": {{
                "high":   "<one sentence summarizing the high-priority theme overall>",
                "medium": "<one sentence summarizing the medium-priority theme overall>",
                "low":    "<one sentence summarizing the low-priority theme overall>"
              }}
            }}
            
            PRIORITY DEFINITIONS:
            - high: issue affects core learning outcomes or would prevent participants from benefiting from future sessions
            - medium: issue affects comfort, convenience, or perceived quality but does not block learning
            - low: enhancement that would improve experience but is not urgent
            
            BREADTH DEFINITIONS:
            - isolated: mentioned by respondents in only one cluster
            - recurring: appears in two clusters
            - widespread: appears in three or more clusters
            
            Sort insights by priority (high first), then by breadth (widespread before isolated)."""

        
    result = call_llm(system_prompt, user_prompt)
    return result