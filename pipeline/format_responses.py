import pandas as pd

"""
Inputs
    labeled_df   :    dataframe labeled with numeric cluster (pd.DataFrame)
    cluster_id   :    the cluter to be processed (int)
    likert_cols  :    dataframe header names of columns containing likert responses (list)
    text_cols    :    dataframe header names of columns containing text responses (list)

Output
    formatted_ratings    : average ratings of each likert question in a string to be LLM readable
    formatted_responses  : transformed text responses to be more LLM readable
"""

def generate_formatted_responses(labeled_df, cluster_id, likert_cols, text_cols) -> str:
    #Format Likert Responses to the Average.
    avg_ratings = labeled_df.groupby('cluster')[likert_cols].mean().round(2)
    formatted_ratings = "\n ".join(
            f"{key}: {round(val, 1)}/5" for key, val in avg_ratings.iloc[cluster_id,:].items())
    formatted_ratings = f"Cluster: {cluster_id}" + "\n" + "="*30 + "\n" + formatted_ratings
    #Format text responses to resemble CSV
    responses = labeled_df.loc[labeled_df['cluster']== cluster_id]
    sample_responses = responses.copy()[text_cols].to_numpy()
    respondent_index = responses.index.to_numpy()
    
    #Format the responses into a readable
    formatted_responses = "Respondent, " + ", ".join(text_cols).strip() +"\n"+"\n".join(
    f"{i+1}, " + ", ".join(str(t) for t in text) for i, text in zip(responses.index.to_numpy(), sample_responses)
    )
    formatted_responses = f"Cluster: {cluster_id}" + "\n" + "="*30 + "\n" + formatted_responses

    # Format the average ratings into a readable string.
    # .items() returns key-value pairs from the dictionary.
    # We round to 1 decimal place for readability.
    return [formatted_ratings, formatted_responses]

def get_all_clusters_table(labeled_df, best_k, likert_cols, text_cols) -> list:
    all_clusters_avgrating = ""
    all_clusters_responses = ""
    for cluster_id in range(3):
        all_clusters_avgrating = all_clusters_avgrating + generate_formatted_responses(labeled_df, cluster_id, likert_cols, text_cols)[0] + "\n\n"
        all_clusters_responses = all_clusters_responses + generate_formatted_responses(labeled_df, cluster_id, likert_cols, text_cols)[1] + "\n\n"
    return [all_clusters_avgrating, all_clusters_responses]