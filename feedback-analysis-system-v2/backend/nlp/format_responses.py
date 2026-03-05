# backend/nlp/format_responses.py
# ─────────────────────────────────────────────────────────────────────────────
# Response Formatting Utilities
#
# These functions transform the clustered DataFrame into readable text blocks
# that can be embedded directly into LLM prompts.
#
# Why format data for an LLM?
#   Language models read text, not DataFrames or NumPy arrays.
#   We need to convert our structured data into clearly labeled text so the
#   model can understand which respondent said what, and what ratings they gave.
#
# Two formats are produced:
#   1. formatted_ratings    — average Likert scores per cluster, as a string
#   2. formatted_responses  — individual text responses in a CSV-like layout
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd


# ── Single-cluster formatter ──────────────────────────────────────────────────

# What this function does:
#   For one specific cluster, produces two formatted strings:
#     (a) a summary of average ratings per Likert question
#     (b) a table of individual text responses
#
# Parameters:
#   labeled_df   (pd.DataFrame) — the full DataFrame with a 'cluster' column
#   cluster_id   (int)          — the cluster number to format (e.g. 0, 1, 2)
#   likert_cols  (list of str)  — column names that hold numeric rating data
#   text_cols    (list of str)  — column names that hold open-ended text answers
#
# Returns:
#   list of two strings: [formatted_ratings, formatted_responses]

def generate_formatted_responses(
    labeled_df: pd.DataFrame,
    cluster_id: int,
    likert_cols: list,
    text_cols: list,
) -> list:

    separator = "=" * 30

    # ── Part A: Average Likert ratings ────────────────────────────────────────
    if likert_cols:
        # .groupby('cluster')[likert_cols].mean() computes the average of each
        # Likert column for every cluster.  .round(2) limits decimal places.
        # The result is a DataFrame indexed by cluster number.
        avg_ratings = labeled_df.groupby('cluster')[likert_cols].mean().round(2)

        # .loc[cluster_id] selects the row for this specific cluster,
        # returning a pandas Series (a single column of values, one per question).
        # .items() iterates over (column_name, value) pairs.
        cluster_avg = avg_ratings.loc[cluster_id]
        ratings_lines = "\n ".join(
            f"{question}: {round(score, 1)}/5"
            for question, score in cluster_avg.items()
        )
        formatted_ratings = (
            f"Cluster: {cluster_id}\n{separator}\n{ratings_lines}"
        )
    else:
        formatted_ratings = f"Cluster: {cluster_id}\n{separator}\n(No Likert data)"

    # ── Part B: Individual text responses ─────────────────────────────────────
    if text_cols:
        # Filter the DataFrame to only rows belonging to this cluster.
        cluster_rows = labeled_df[labeled_df['cluster'] == cluster_id]

        # Extract just the text columns as a numpy array for easy iteration.
        # Each row of this array is one respondent's text answers.
        response_matrix = cluster_rows[text_cols].to_numpy()

        # .index.to_numpy() gives us the original row indices (0-based integers)
        # from the DataFrame, so we can label respondents consistently.
        original_indices = cluster_rows.index.to_numpy()

        # Build a CSV-like header row: "Respondent, Question1, Question2, ..."
        header = "Respondent, " + ", ".join(text_cols).strip()

        # Build one data row per respondent.
        # We format each row as: "R001, answer1, answer2, ..."
        # str(idx + 1).zfill(3) converts index 0 → "001", 9 → "010", etc.
        # We use zfill to make IDs sort lexicographically (R001 before R010).
        data_rows = "\n".join(
            f"R{str(original_idx + 1).zfill(3)}, "
            + ", ".join(
                str(cell).replace(",", ";").replace("\n", " ")  # Escape commas and newlines
                for cell in text_row
            )
            for original_idx, text_row in zip(original_indices, response_matrix)
        )

        formatted_responses = (
            f"Cluster: {cluster_id}\n{separator}\n{header}\n{data_rows}"
        )
    else:
        formatted_responses = f"Cluster: {cluster_id}\n{separator}\n(No text responses)"

    return [formatted_ratings, formatted_responses]


# ── All-clusters formatter ────────────────────────────────────────────────────

# What this function does:
#   Runs generate_formatted_responses() for every cluster and concatenates
#   the results.  The LLM receives one long string covering all clusters,
#   which lets it compare them side by side.
#
# Parameters:
#   labeled_df   (pd.DataFrame) — full labeled DataFrame
#   best_k       (int)          — total number of clusters
#   likert_cols  (list of str)  — Likert column names
#   text_cols    (list of str)  — text column names
#
# Returns:
#   list of two strings: [all_ratings_combined, all_responses_combined]

def get_all_clusters_table(
    labeled_df: pd.DataFrame,
    best_k: int,
    likert_cols: list,
    text_cols: list,
) -> list:
    all_ratings    = ""
    all_responses  = ""

    # Iterate over all cluster IDs (0 through best_k - 1).
    # We use range(best_k) which gives [0, 1, 2, …, best_k-1].
    for cluster_id in range(best_k):
        # Generate both formatted strings for this cluster.
        cluster_formatted = generate_formatted_responses(
            labeled_df, cluster_id, likert_cols, text_cols
        )

        # Append each block with a blank line between clusters for readability.
        all_ratings   += cluster_formatted[0] + "\n\n"
        all_responses += cluster_formatted[1] + "\n\n"

    return [all_ratings.strip(), all_responses.strip()]
