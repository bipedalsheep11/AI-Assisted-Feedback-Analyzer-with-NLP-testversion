module_dir = os.path.abspath('../nlp')
sys.path.append(module_dir)
import llm_client.call_llm_with_retry as call_llm

def label_cluster_with_llm(cluster_id, formatted_ratings, formatted_responses, all_clusters_table, system_prompt, n_sample=8):

    # Initialize the Groq client.
    # It automatically reads GROQ_API_KEY from your environment variables.

    # Build the prompt — now includes both the ratings table for all clusters
    # (for comparison context) and the specific cluster's text responses
    prompt = f"""You are analyzing post-program training evaluation data. 
            Your task is to characterize the respondents in a specific cluster — 
            meaning: describe who they are as a group, based on how they rated 
            the training and what they wrote in their free-text responses.
            
            Characterization should focus on:
            - Their overall satisfaction and engagement with the training
            - Which aspects of the training they responded most strongly to 
              (positively or negatively), such as facilitator quality, 
              pacing, content relevance, or practical applicability
            - Any consistent patterns in what they valued or found lacking
            
            ---
            
            FOR CONTEXT — Average ratings and sample responses for ALL clusters 
            are shown below. Use this to understand how Cluster {cluster_id} 
            differs from the others. Each cluster label should be distinct and 
            meaningful when compared to the rest.
            
            All clusters — average ratings:
            {all_clusters_table[0]}
            
            Treat this as a CSV structure. The first line corresponds to the questions answered by the following lines.
            Base your responses and context on their answers to the questions. ALways go back to the question they are answering.
            All clusters — sample text responses:
            {all_clusters_table[1]}
            
            ---
            
            YOU ARE LABELING: Cluster {cluster_id}
            
            Cluster {cluster_id} — average ratings:
            {formatted_ratings}
            
            Cluster {cluster_id} — sample text responses:
            {formatted_responses}
            
            ---
            
            Return ONLY a JSON object. No markdown, no code fences, no explanation.
            Use this exact structure:
            
            {{
              "label": "A short 2-4 word phrase that distinctly characterizes this group of respondents",
              "respondent_profile": "One to two sentences describing who these respondents appear to be — their overall disposition toward the training, and what most shaped their experience",
              "key_drivers": ["The 2-4 specific training aspects that most explain this cluster's ratings and responses, drawn directly from the data"],
              "distinguishing_features": "One sentence explaining what makes this cluster different from the other clusters shown above"
            }}"""
    
    return call_llm(system_prompt, prompt)

def label_cluster_with_llm_all(num_clusters, formatted_ratings, formatted_responses, all_clusters_table, n_sample=8)
    json_per_cluster = {}
    for cluster_id in range(num_clusters):
        json_per_cluster[f"Cluster {cluster_id}"] = label_cluster_with_llm(cluster_id, formatted_ratings, formatted_responses, all_clusters_table, n_sample=8)
    return json_per_cluster