"""
This module returns a DataFrame or CSV with label/clusters.
"""

#Define Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCAfrom sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def separate_likert_from_text(dataframe: pd.DataFrame, likert_max_unique: int = 10) -> dict[list]:
    likert_columns = []
    text_columns   = []
    other_columns  = []

    for col_name in dataframe.columns:
        column_data = dataframe[col_name].dropna() #extracts column as a seriees

        #check if column is empty
        if len(column_data) == 0:
            other_columns.append(col_name)
            continue

        #Check 1: if values are numeric
        is_numeric = pd.api.types.is_numeric_dtype(column_data)

        #Check2: count number of unique values
        unique_count = column_data.unique()

        #Check 4: Contains alphabetic chars?
        has_letters = column_data.astype(str).str.contains('[a-zA-Z]').any()

        #----Decision Logic------
        if is_numeric and not has_letters:
            likert_columns.append(col_name)
        elif not is_numeric and has_letters:
            text_columns.append(col_name)
        else:
            other_columns.append(col_name)

    return {'likert': likert_columns, 'text':   text_columns, 'other':  other_columns}

def normalize_likert(likert_df) -> np.array: 

    # MinMaxScaler is a transformer from scikit-learn.
    # It rescales each column independently so that the minimum value in that column
    # becomes 0.0 and the maximum value becomes 1.0. All values in between are
    # proportionally scaled. 
    scaler = MinMaxScaler()

    # .fit_transform() does two things in one call:
    # 1. .fit() — the scaler learns the min and max of each column from the data
    # 2. .transform() — it applies the scaling formula to convert values to 0–1
    # The result is a 2D numpy array with shape (num_respondents, num_likert_columns)
    normalized_likert = scaler.fit_transform(likert_df)
    print(f"Normalized Likert array shape: {normalized_likert.shape}")
    return normalized_likert  

def embed_text_responses(text_df) -> np.array:
    # SentenceTransformer loads a pretrained embedding model.
    # 'all-MiniLM-L6-v2' is a widely used lightweight model. It:
    #   - Produces 384-dimensional embeddings
    #   - Is fast enough to run on a standard laptop CPU
    #   - Performs well on English survey text
    # The model files are downloaded automatically on first use (~80MB).
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # We concatenate all open-ended text columns for each respondent into one string.
    # This way, each respondent is represented by all their text together, not separately.
    #
    # .fillna("") replaces any missing (NaN) values with an empty string,
    # because the embedding model cannot process Python's NaN (Not a Number) values.
    #
    # .agg(" ".join, axis=1) applies a function across each row (axis=1 means row-wise).
    # For each row, it joins the values of all selected text columns with a space between them.
    combined_text = text_df.fillna("").agg(" ".join, axis=1)

    # Convert the pandas Series to a plain Python list.
    # The .encode() method below expects a list of strings.
    text_list = combined_text.tolist()

    # .encode() runs each string through the embedding model and returns
    # a 2D numpy array. Each row is one respondent's embedding vector.
    # show_progress_bar=True prints a progress indicator, useful for larger datasets.
    embeddings = model.encode(text_list, show_progress_bar=True)

    print(f"Text embedding array shape: {embeddings.shape}")
    # Expected output: (num_respondents, 384)

    return embeddings

def combine_features(normalized_likert, text_embeddings):
    combined = np.hstack([normalized_likert, text_embeddings])
    print(f"Combined feature array shape: {combined.shape}")
    return combined

def reduce_dimensions(combined_features, n_components=50):
    # Cap n_components at the number of features available.
    # You cannot reduce to more dimensions than you started with.
    n_components = min(n_components, combined_features.shape[1])

    # PCA() initializes the PCA transformer.
    # n_components tells it how many dimensions to keep.
    # random_state=42 makes the result reproducible — PCA has a random initialization step,
    # and setting this ensures you get the same result every time you run it.
    pca = PCA(n_components=n_components, random_state=42)

    # .fit_transform() learns the principal components from the data
    # and immediately applies the transformation.
    reduced = pca.fit_transform(combined_features)

    # explained_variance_ratio_ is an array that tells you what fraction of the
    # total variation in the data each component captures.
    # Summing them tells you how much information is retained overall.
    variance_retained = sum(pca.explained_variance_ratio_) * 100
    print(f"Reduced to {n_components} dimensions, retaining {variance_retained:.1f}% of variance.")

    return reduced

def auto_cluster(features, min_k=2, max_k=10):

    # This dictionary will store the silhouette score for each k we test.
    # Key = k (number of clusters), Value = silhouette score
    scores = {}

    # This dictionary stores the cluster labels produced by each k,
    # so we do not have to re-run K-Means after finding the best k.
    all_labels = {}

    for k in range(min_k, max_k + 1):

        # Run K-Means with this value of k.
        # n_init=10 runs the algorithm 10 times with different random starts
        # and keeps the best result — this reduces the chance of a poor solution.
        # random_state=42 makes results reproducible.
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)

        # .fit_predict() trains the model and returns cluster labels in one step.
        # labels is a 1D array like [0, 2, 1, 0, 3, ...] — one integer per respondent.
        labels = kmeans.fit_predict(features)

        # silhouette_score() computes the average silhouette score across all points.
        # It needs:
        #   features — the actual data points (so it can measure distances)
        #   labels   — the cluster assignment for each point
        # A higher score means clusters are more distinct and well-separated.
        score = silhouette_score(features, labels)

        # Store both the score and the labels for this k
        scores[k] = score
        all_labels[k] = labels

        print(f"  k={k:2d}  →  silhouette score: {score:.4f}")

    # Find the k with the highest silhouette score.
    # max() with a key function finds the dictionary key whose value is largest.
    # scores.get is passed as the key function — it looks up the score for each k.
    best_k = max(scores, key=scores.get)

    print(f"\n✓ Best k = {best_k}  (silhouette score: {scores[best_k]:.4f})")

    # Retrieve the labels that were already computed for the best k.
    best_labels = all_labels[best_k]

    # Build a summary dict so you can inspect all scores if you want to
    scores_summary = {
        "best_k": best_k,
        "best_score": scores[best_k],
        "all_scores": scores
    }

    return best_k, best_labels, scores_summary

def attach_and_inspect_clusters(dataframe, cluster_labels):
    # .copy() creates a full copy of the DataFrame so we do not accidentally
    # modify the original data in place
    labeled_df = dataframe.copy()

    # Add the cluster assignments as a new column called 'cluster'
    labeled_df['cluster'] = cluster_labels

    return labeled_df

def run_clustering_pipeline(survey_data) -> dict:
    likert_cols, text_cols, other_cols = separate_likert_from_text(dataframe).values()
    likert_df = dataframe[likert_cols]
    text_df = dataframe[text_cols]

    #Stage 1: Normalize the likert scale:
    normalized_likert = normalize_likert(likert_df)

    #Stage 2: Generate text embeddings:
    text_embeddings = embed_text_responses(text_df)

    #Stage 3: Combine likert and text embeddings
    combined_features = combine_features(normalized_likert, text_embeddings)

    #Stage 4: Reduce feature dimensions using PCA
    reduced_features = reduce_dimensions(combined_features, n_components=30)

    #Stage 5: Auto cluster the responses
    best_k, best_labels, score_summary = auto_cluster(reduced_features)

    #Stage 6: Label Data
    labeled_df = attach_and_inspect_clusters(dataframe, best_labels)
    
    return {"Labeled Survey Data": labeled_df, "Clusters": best_k, "Likert+Text": [likert_cols,text_cols]}

