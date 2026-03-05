# backend/nlp/auto_clustering.py
# ─────────────────────────────────────────────────────────────────────────────
# Automatic Respondent Clustering Pipeline
#
# This module groups survey respondents into clusters based on how similar
# their responses are — both their numerical Likert ratings and their
# free-text answers.
#
# Conceptual overview of the pipeline:
#   1. Separate columns into Likert (numbers) vs text (open-ended responses).
#   2. Normalize Likert scores to a 0–1 scale so they're comparable.
#   3. Encode text responses into embedding vectors (numerical representations
#      that capture meaning).
#   4. Combine both into a single feature array per respondent.
#   5. Reduce dimensions with PCA to speed up clustering.
#   6. Run K-Means clustering and find the best number of clusters (k).
#   7. Attach cluster labels back to the original DataFrame.
#
# Install dependencies:
#   pip install pandas numpy scikit-learn sentence-transformers
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer


# ── Step 1: Column separation ─────────────────────────────────────────────────

# What this function does:
#   Inspects every column in the DataFrame and sorts it into one of three
#   categories: Likert (numeric rating scale), text (open-ended prose), or
#   other (identifiers, dates, booleans, etc. that don't fit either).
#
# Parameters:
#   dataframe        (pd.DataFrame) — the full survey DataFrame
#   likert_max_unique (int)         — if a numeric column has this many or fewer
#                                     unique values, it's treated as a Likert scale.
#                                     Default: 10.  A rating-1-to-5 column has 5
#                                     unique values; a continuous score like 87.4
#                                     would have many more.
#
# Returns:
#   dict with keys 'likert', 'text', and 'other', each holding a list of
#   column name strings.

def separate_likert_from_text(
    dataframe: pd.DataFrame,
    likert_max_unique: int = 10,
) -> dict:

    likert_columns = []
    text_columns   = []
    other_columns  = []

    for col_name in dataframe.columns:
        # Drop NaN values before inspecting — empty cells don't tell us anything
        # about the column's type.
        column_data = dataframe[col_name].dropna()

        # If every cell in the column is empty, we can't classify it.
        if len(column_data) == 0:
            other_columns.append(col_name)
            continue

        # Check 1: Is the column stored as a numeric dtype (int or float)?
        # pd.api.types.is_numeric_dtype() returns True for int64, float64, etc.
        is_numeric = pd.api.types.is_numeric_dtype(column_data)

        # Check 2: How many distinct values does this column have?
        # A Likert scale like 1–5 has at most 5 unique values.
        unique_count = column_data.nunique()  # .nunique() returns the count, not the values

        # Check 3: Does any cell contain alphabetic characters?
        # .astype(str) converts every cell to a string for safe inspection.
        # .str.contains('[a-zA-Z]') returns True/False per cell.
        # .any() returns True if at least one cell matched.
        has_letters = column_data.astype(str).str.contains('[a-zA-Z]').any()

        # ── Decision logic ────────────────────────────────────────────────────
        if is_numeric and not has_letters and unique_count <= likert_max_unique:
            # Numeric, no text, and few unique values → Likert scale.
            likert_columns.append(col_name)
        elif not is_numeric and has_letters:
            # Non-numeric and contains letters → open-ended text response.
            text_columns.append(col_name)
        elif is_numeric and unique_count > likert_max_unique:
            # Numeric but too many unique values to be a rating scale
            # (e.g. age, score out of 100) → treat as other.
            other_columns.append(col_name)
        else:
            other_columns.append(col_name)

    return {
        "likert": likert_columns,
        "text":   text_columns,
        "other":  other_columns,
    }


# ── Step 2: Normalize Likert scores ──────────────────────────────────────────

# What this function does:
#   Rescales each Likert column so its minimum value becomes 0.0 and its
#   maximum becomes 1.0.  This prevents columns with higher raw ranges
#   (e.g. 1–7) from dominating columns with smaller ranges (e.g. 1–5)
#   in the clustering algorithm.
#
# Why MinMaxScaler?
#   It preserves the relative distances between values within a column while
#   making all columns live on the same 0–1 scale.  This is the standard
#   approach for Likert data where we don't assume a normal distribution.
#
# Parameters:
#   likert_df  (pd.DataFrame) — a DataFrame containing only Likert columns
#
# Returns:
#   np.ndarray — 2D array of shape (n_respondents, n_likert_columns)

def normalize_likert(likert_df: pd.DataFrame) -> np.ndarray:
    # Fill missing Likert values with the column median before scaling.
    # An empty rating cell would otherwise become NaN in the output,
    # which would break downstream calculations.
    filled_df = likert_df.fillna(likert_df.median())

    # MinMaxScaler rescales each column independently.
    scaler = MinMaxScaler()

    # .fit_transform() learns min/max per column and applies the rescaling
    # in a single step. Returns a 2D numpy array.
    normalized = scaler.fit_transform(filled_df)

    print(f"  Normalized Likert shape: {normalized.shape}")
    return normalized


# ── Step 3: Embed text responses ──────────────────────────────────────────────

# What this function does:
#   Converts free-text survey answers into fixed-length numerical vectors
#   called "embeddings." Each embedding captures the semantic meaning of
#   a response, so similar responses end up with similar vectors.
#
# What is an embedding?
#   Imagine mapping every English sentence onto a point in 384-dimensional
#   space. Sentences with similar meanings land near each other.
#   "The facilitator was excellent" and "The trainer was outstanding" would
#   have similar embeddings, even though the words are different.
#
# Why 'all-MiniLM-L6-v2'?
#   It's a well-tested lightweight model that runs on CPU in a reasonable time,
#   produces high-quality embeddings for English text, and is small (~80 MB).
#
# Parameters:
#   text_df  (pd.DataFrame) — DataFrame containing only the text response columns
#
# Returns:
#   np.ndarray — 2D array of shape (n_respondents, 384)

def embed_text_responses(text_df: pd.DataFrame) -> np.ndarray:
    # Load the SentenceTransformer model.
    # On the first run this downloads the model weights (~80 MB).
    # Subsequent runs use the cached version.
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Combine all text columns for each respondent into one string.
    # .fillna("") replaces NaN cells with an empty string — the model
    # cannot process Python's NaN (Not a Number) value.
    # .agg(" ".join, axis=1) concatenates across columns, row by row.
    combined_text = text_df.fillna("").astype(str).agg(" ".join, axis=1)

    # Convert the pandas Series to a plain Python list of strings,
    # because .encode() expects that format.
    text_list = combined_text.tolist()

    # .encode() runs every string through the neural network and returns
    # one 384-dimensional vector per string.
    embeddings = model.encode(text_list, show_progress_bar=True)

    print(f"  Text embedding shape: {embeddings.shape}")
    return embeddings


# ── Step 4: Combine features ──────────────────────────────────────────────────

# What this function does:
#   Joins the normalized Likert array and the text embeddings side by side
#   so that each respondent is represented by a single combined feature vector.
#
# Parameters:
#   normalized_likert  (np.ndarray) — shape (n_respondents, n_likert_cols)
#   text_embeddings    (np.ndarray) — shape (n_respondents, 384)
#
# Returns:
#   np.ndarray — shape (n_respondents, n_likert_cols + 384)

def combine_features(
    normalized_likert: np.ndarray,
    text_embeddings: np.ndarray,
) -> np.ndarray:
    # np.hstack() stacks arrays horizontally (column-wise).
    # Row 0 of the result contains row 0 from likert + row 0 from embeddings.
    combined = np.hstack([normalized_likert, text_embeddings])
    print(f"  Combined feature shape: {combined.shape}")
    return combined


# ── Step 5: Dimensionality reduction ─────────────────────────────────────────

# What this function does:
#   Reduces the combined feature array from potentially 400+ dimensions to
#   a smaller number using Principal Component Analysis (PCA).
#
# What is PCA?
#   PCA finds the directions (principal components) in the data that capture
#   the most variation.  It's like finding the "most informative viewing
#   angles" for the data.  Keeping the top 50 components typically retains
#   95%+ of the information while dramatically speeding up clustering.
#
# Why reduce dimensions?
#   K-Means performs poorly in very high dimensions (the "curse of
#   dimensionality").  Reducing to 30–50 dimensions improves cluster quality
#   and makes the algorithm much faster.
#
# Parameters:
#   combined_features  (np.ndarray) — the combined Likert + embedding array
#   n_components       (int)        — how many dimensions to keep. Default: 50
#
# Returns:
#   np.ndarray — shape (n_respondents, n_components)

def reduce_dimensions(
    combined_features: np.ndarray,
    n_components: int = 50,
) -> np.ndarray:
    # We cannot keep more components than the original number of features.
    n_components = min(n_components, combined_features.shape[1])

    # random_state=42 makes PCA reproducible — it has an internal random step.
    pca = PCA(n_components=n_components, random_state=42)

    # .fit_transform() learns the principal components and applies the
    # transformation in one call.
    reduced = pca.fit_transform(combined_features)

    # explained_variance_ratio_ is an array showing what fraction of total
    # data variance each component captures. Summing gives the total retained.
    variance_retained = sum(pca.explained_variance_ratio_) * 100
    print(f"  Reduced to {n_components} dims — {variance_retained:.1f}% variance retained")

    return reduced


# ── Step 6: Find best k and cluster ──────────────────────────────────────────

# What this function does:
#   Runs K-Means clustering for every integer k from min_k to max_k.
#   For each k it computes the silhouette score — a measure of how well
#   separated the clusters are.  The k with the highest score wins.
#
# What is the silhouette score?
#   For each data point it measures:
#     (a) how close the point is to others in its own cluster
#     (b) how far the point is from points in the nearest other cluster
#   Score = (b - a) / max(a, b). Ranges from -1 (wrong cluster) to +1 (ideal).
#   A score above 0.5 is generally considered good.
#
# Parameters:
#   features   (np.ndarray) — the reduced feature array
#   min_k      (int)        — smallest number of clusters to try. Default: 2
#   max_k      (int)        — largest number of clusters to try. Default: 8
#   force_k    (int | None) — if set, skip the search and use exactly this k
#
# Returns:
#   tuple — (best_k: int, best_labels: np.ndarray, scores_summary: dict)

def auto_cluster(
    features: np.ndarray,
    min_k: int = 2,
    max_k: int = 8,
    force_k: int = None,
) -> tuple:
    # If the caller supplied a fixed k, skip the search.
    if force_k is not None:
        print(f"  Using forced k={force_k}")
        kmeans = KMeans(n_clusters=force_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)
        return force_k, labels, {"best_k": force_k, "forced": True}

    scores     = {}   # Maps k → silhouette score
    all_labels = {}   # Maps k → cluster label array

    # Cap max_k so we never request more clusters than data points.
    max_k = min(max_k, features.shape[0] - 1)

    for k in range(min_k, max_k + 1):
        # n_init=10 runs K-Means 10 times from different random starts
        # and returns the best result.  This reduces the chance of a
        # poor local optimum.
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)

        # silhouette_score needs the raw features (to compute distances)
        # and the cluster assignments.
        score  = silhouette_score(features, labels)

        scores[k]     = score
        all_labels[k] = labels
        print(f"    k={k:2d}  →  silhouette: {score:.4f}")

    # Find the k with the highest silhouette score.
    best_k      = max(scores, key=scores.get)
    best_labels = all_labels[best_k]
    print(f"\n  ✓ Best k = {best_k}  (score: {scores[best_k]:.4f})")

    scores_summary = {
        "best_k":     best_k,
        "best_score": scores[best_k],
        "all_scores": scores,
    }
    return best_k, best_labels, scores_summary


# ── Step 7: Attach cluster labels back to DataFrame ───────────────────────────

# What this function does:
#   Adds a new 'cluster' column to the original DataFrame so that every
#   respondent row now has its cluster assignment.
#
# Parameters:
#   dataframe       (pd.DataFrame) — the original survey DataFrame (unmodified)
#   cluster_labels  (np.ndarray)   — 1D array of integer cluster IDs, one per row
#
# Returns:
#   pd.DataFrame — a copy of the original DataFrame with a new 'cluster' column

def attach_clusters(dataframe: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    # .copy() prevents us from modifying the DataFrame that was passed in.
    labeled_df = dataframe.copy()
    labeled_df['cluster'] = cluster_labels
    return labeled_df


# ── Main pipeline entry point ─────────────────────────────────────────────────

# What this function does:
#   Orchestrates all six steps above from a raw DataFrame to a clustered one.
#   Returns everything the Streamlit app needs: the labeled DataFrame, the
#   number of clusters, column lists, and 2D PCA coordinates for the scatter plot.
#
# Parameters:
#   dataframe   (pd.DataFrame) — the raw survey DataFrame (already loaded)
#   force_k     (int | None)   — if provided, skip auto-selection and use this k
#   n_pca_dims  (int)          — number of dimensions to reduce to before clustering
#
# Returns:
#   dict with keys:
#     labeled_df  — DataFrame with 'cluster' column
#     best_k      — integer number of clusters
#     likert_cols — list of Likert column names
#     text_cols   — list of text column names
#     pca_coords  — 2D numpy array for scatter plot visualization

def run_clustering_pipeline(
    dataframe: pd.DataFrame,
    force_k: int = None,
    n_pca_dims: int = 30,
) -> dict:
    print("\n=== Clustering Pipeline ===")

    # ── Stage 1: Identify column types ───────────────────────────────────────
    print("\nStage 1: Separating column types…")
    column_types = separate_likert_from_text(dataframe)
    likert_cols  = column_types["likert"]
    text_cols    = column_types["text"]

    print(f"  Likert columns ({len(likert_cols)}): {likert_cols}")
    print(f"  Text columns  ({len(text_cols)}): {text_cols}")

    # ── Stage 2: Build feature matrix ────────────────────────────────────────
    print("\nStage 2: Building feature matrix…")

    # We need at least one column type to proceed.
    if not likert_cols and not text_cols:
        raise ValueError(
            "No Likert or text columns detected. "
            "Check that your CSV has numeric rating columns and/or text response columns."
        )

    feature_parts = []

    if likert_cols:
        normalized_likert = normalize_likert(dataframe[likert_cols])
        feature_parts.append(normalized_likert)

    if text_cols:
        print("  Generating text embeddings (this may take 1–2 minutes on first run)…")
        text_embeddings = embed_text_responses(dataframe[text_cols])
        feature_parts.append(text_embeddings)

    # Combine whatever parts we have into one matrix.
    if len(feature_parts) == 2:
        combined_features = combine_features(feature_parts[0], feature_parts[1])
    else:
        # Only one type of column available — use it directly.
        combined_features = feature_parts[0]

    # ── Stage 3: Reduce dimensions ────────────────────────────────────────────
    print("\nStage 3: Reducing dimensions with PCA…")
    reduced_features = reduce_dimensions(combined_features, n_components=n_pca_dims)

    # Also create a 2-component PCA for the 2D scatter plot visualization.
    # This is separate from the clustering PCA — we want exactly 2 dimensions
    # to plot on x/y axes regardless of how many we use for clustering.
    pca_2d = PCA(n_components=2, random_state=42)
    pca_coords = pca_2d.fit_transform(combined_features)

    # ── Stage 4: Cluster ──────────────────────────────────────────────────────
    print("\nStage 4: Finding optimal clusters…")
    best_k, best_labels, score_summary = auto_cluster(
        reduced_features,
        force_k=force_k,
    )

    # ── Stage 5: Attach labels to DataFrame ───────────────────────────────────
    print("\nStage 5: Attaching cluster labels…")
    labeled_df = attach_clusters(dataframe, best_labels)

    print(f"\n✓ Pipeline complete — {best_k} clusters, {len(labeled_df)} respondents")

    return {
        "labeled_df":  labeled_df,
        "best_k":      best_k,
        "likert_cols": likert_cols,
        "text_cols":   text_cols,
        "pca_coords":  pca_coords,
        "scores":      score_summary,
    }
