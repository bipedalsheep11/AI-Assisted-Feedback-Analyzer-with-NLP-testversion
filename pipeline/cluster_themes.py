def cluster_themes(all_clusters_responses, system_prompt, predefined_themes=None, n_sample=15):


    # Build the theme instruction block depending on which mode is active
    if predefined_themes:
        # Predefined mode: give the model the categories and instruct it to use them
        theme_list = "\n".join(f"- {t}" for t in predefined_themes)
        theme_instruction = f"""Assign each response to ONE of the following predefined themes.
    If a response clearly fits none of the predefined themes, assign it to "Other" and note what theme it actually represents.
    
    PREDEFINED THEMES:
    {theme_list}"""
    
        else:
            # Discovery mode: instruct the model to identify themes from the responses themselves
            theme_instruction = """Identify recurring themes from the responses below.
    Aim for 3 to 6 themes that are specific enough to be actionable but broad enough to cover multiple responses.
    Do not create a unique theme for every response — look for patterns.
    Name each theme as a short noun phrase (e.g. "Facilitator Clarity", "Schedule Management", "Material Relevance")."""
    
        user_prompt = f"""You are performing thematic analysis on each cluster in {all_clusters_responses[1]} responses from a post-training evaluation.

                    {theme_instruction}
                    
                    Return a JSON object with exactly these keys:
                    
                    {{
                      "cluster_id": {cluster_id},
                      "themes_used": ["<theme 1>", "<theme 2>", ...],
                      "coded_responses": [
                        {{
                          "respondent_id": "<id>",
                          "assigned_theme": "<theme name>",
                          "theme_fit": "<strong | moderate | weak>",
                          "supporting_quote": "<5-10 word phrase from the response that justifies the theme assignment>"
                        }}
                      ],
                      "theme_summary": {{
                        "<theme name>": {{
                          "count": <number of responses assigned to this theme>,
                          "description": "<one sentence characterizing what respondents in this theme are saying>"
                        }}
                      }}
                    }}
                    
                    THEME FIT DEFINITIONS:
                    - strong: the response is almost entirely about this theme
                    - moderate: this theme is clearly present but the response also covers other topics
                    - weak: the response only tangentially relates to this theme — flag for manual review"""

    result = call_llm(system_prompt, user_prompt)

    return result