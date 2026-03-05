# backend/utils/data_cleaning.py
# ─────────────────────────────────────────────────────────────────────────────
# Data Cleaning Utilities
#
# Survey data is rarely clean straight out of a spreadsheet.
# Likert scale responses in particular can arrive in many formats:
#   - Plain integers:            3
#   - Floats with decimal:       4.0
#   - Text labels:               "Good", "Satisfied", "VERY GOOD"
#   - Combined format:           "4 - Satisfied", "3: Fair"
#   - Responses out of range:    0, 6 (for a 1–5 scale)
#
# These functions normalize all of those into plain integers so the
# clustering and visualization pipeline can process them reliably.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd


# ── Single-value converter ────────────────────────────────────────────────────

# What this function does:
#   Takes one cell value from a Likert column — whatever format it's in —
#   and tries to return a plain integer.
#   Returns None if the value cannot be interpreted, so the caller can
#   identify and log problem cells without crashing.
#
# Parameters:
#   raw_value   (any)   — a single cell value (int, float, or string)
#   label_map   (dict)  — maps text labels to integers.
#                         e.g. {"Poor": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5}
#                         Pass an empty dict {} if your data is purely numeric.
#
# Returns:
#   int | None — the parsed integer, or None on failure

def convert_likert_value(raw_value, label_map: dict):
    # pd.isna() returns True for NaN, None, and pandas NA types.
    # We return None immediately — a missing value is not a parsing failure.
    if pd.isna(raw_value):
        return None

    # Convert to string and strip whitespace so we can apply text operations
    # consistently, regardless of whether the original was a number or text.
    cleaned = str(raw_value).strip()

    # ── Attempt 1: Plain number ("3" or "4.0") ───────────────────────────────
    # We go through float first because int("4.0") raises a ValueError,
    # but int(float("4.0")) works correctly.
    try:
        return int(float(cleaned))
    except ValueError:
        pass  # Move to the next attempt.

    # ── Attempt 2: Text label ("Good", "VERY GOOD", "very good") ────────────
    # Normalize both the value and the map keys to lowercase for case-insensitive matching.
    normalized = cleaned.lower()
    lowercase_map = {key.lower(): value for key, value in label_map.items()}

    if normalized in lowercase_map:
        return lowercase_map[normalized]

    # ── Attempt 3: Combined format ("4 - Satisfied", "3: Fair") ─────────────
    # If the first character is a digit, try extracting the leading number.
    if cleaned and cleaned[0].isdigit():
        # Split on whitespace and take the first piece ("4" from "4 - Satisfied").
        leading_part = cleaned.split()[0].rstrip(".-:")  # Strip trailing punctuation.
        try:
            return int(float(leading_part))
        except ValueError:
            pass

    # ── Fallback: Could not interpret this value ─────────────────────────────
    print(f"  Warning: Could not convert Likert value '{raw_value}' — returning None.")
    return None


# ── Single-column cleaner ─────────────────────────────────────────────────────

# What this function does:
#   Applies convert_likert_value() to every cell in one DataFrame column.
#
# Parameters:
#   dataframe    (pd.DataFrame) — the full survey DataFrame
#   column_name  (str)          — which column to clean
#   label_map    (dict)         — text-to-integer mapping (can be empty {})
#
# Returns:
#   pd.Series — the cleaned column as a Series of integers/None values

def clean_likert_column(
    dataframe: pd.DataFrame,
    column_name: str,
    label_map: dict,
) -> pd.Series:
    # .apply() calls convert_likert_value() on every cell in the column.
    # lambda wraps it because apply() passes one argument (the cell value)
    # but convert_likert_value() needs two (value + label_map).
    return dataframe[column_name].apply(
        lambda cell: convert_likert_value(cell, label_map)
    )


# ── All-columns cleaner ────────────────────────────────────────────────────────

# What this function does:
#   Iterates over a list of Likert column names and cleans each one in place.
#   Skips any column name that doesn't actually exist in the DataFrame,
#   printing a warning rather than crashing.
#
# Parameters:
#   dataframe      (pd.DataFrame) — the full survey DataFrame
#   likert_columns (list of str)  — column names to clean
#   label_map      (dict)         — text-to-integer mapping; pass {} if not needed
#
# Returns:
#   pd.DataFrame — the same DataFrame with the specified columns cleaned

def clean_all_likert_columns(
    dataframe: pd.DataFrame,
    likert_columns: list,
    label_map: dict,
) -> pd.DataFrame:
    for column_name in likert_columns:

        # Guard: make sure the column exists before trying to access it.
        if column_name not in dataframe.columns:
            print(f"  Warning: Column '{column_name}' not found in DataFrame — skipping.")
            continue

        # Clean the column and overwrite it in the DataFrame.
        dataframe[column_name] = clean_likert_column(dataframe, column_name, label_map)
        print(f"  Cleaned: '{column_name}'")

    return dataframe


# ── Index column dropper ───────────────────────────────────────────────────────

# What this function does:
#   Many CSV exports include an auto-generated index column as the first column
#   (e.g. a column named "" or "Unnamed: 0" or "index").
#   This function removes it if detected, so it doesn't pollute the analysis.
#
# Parameters:
#   dataframe  (pd.DataFrame) — the raw DataFrame immediately after loading
#
# Returns:
#   pd.DataFrame — the DataFrame with the spurious index column removed (if found)

def drop_index_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    # These are the names that pandas or Excel typically give to auto-index columns.
    index_column_names = {"", "unnamed: 0", "index", "id", "#"}

    first_col = dataframe.columns[0].lower().strip()
    if first_col in index_column_names:
        # .iloc[:, 1:] selects all rows (:) and all columns from index 1 onwards,
        # effectively dropping the first column.
        print(f"  Dropping likely index column: '{dataframe.columns[0]}'")
        return dataframe.iloc[:, 1:]

    return dataframe
