import pandas as pd
import fitz  
import re
import io
import ipywidgets as widgets

# This function converts a single Likert scale value — regardless of whether
# it is stored as a number, a text label, or a combined format like "4 - Satisfied" —
# into a plain integer.
#
# Input:
#   raw_value      : a single cell value from your DataFrame (could be int, float, or string)
#   label_map      : a dictionary where keys are text labels and values are integers
#                    e.g. {"Poor": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5}
#
# Returns:
#   An integer representing the numeric Likert score, or None if the value
#   cannot be interpreted (so you can spot and handle problem rows separately).

def convert_likert_value(raw_value, label_map):

    # If the cell is empty (pandas represents missing values as float NaN),
    # we return None immediately. We don't try to convert a missing value.
    # pd.isna() returns True for NaN, None, and similar "missing" sentinels.
    if pd.isna(raw_value):
        return None

    # Convert the value to a string so we can apply text operations consistently,
    # regardless of whether it came in as a number or text.
    # str() turns any Python value into its string representation.
    # .strip() removes leading and trailing whitespace (spaces, tabs, newlines).
    # This handles cases like "  Good  " which should match "Good".
    cleaned = str(raw_value).strip()

    # --- Attempt 1: The value is already a plain number ---
    # Some rows may already be stored as "3" or "4.0".
    # We try to convert directly to float first (not int) because "4.0" cannot
    # be passed directly to int() without going through float first.
    # int(float("4.0")) works; int("4.0") raises an error.
    try:
        return int(float(cleaned))
    except ValueError:
        # If conversion fails (e.g. the value is "Good"), Python raises a ValueError.
        # We catch it here and move on to the next attempt rather than crashing.
        pass

    # --- Attempt 2: The value is a text label like "Good" or "very good" ---
    # We normalize to lowercase so "Good", "GOOD", and "good" all match the same key.
    # The label_map keys should also be lowercase for this to work — we handle that below.
    normalized = cleaned.lower()

    # We also build a lowercase version of the label_map for comparison.
    # Dictionary comprehension: for each key-value pair in label_map,
    # create a new pair where the key is lowercased.
    # This means the caller does not have to worry about capitalization in their map.
    lowercase_map = {key.lower(): value for key, value in label_map.items()}

    if normalized in lowercase_map:
        # The text label matched a known key — return its mapped integer.
        return lowercase_map[normalized]

    # --- Attempt 3: The value is a combined format like "4 - Satisfied" or "4: Good" ---
    # We check if the first character is a digit. If it is, we try to extract
    # just the leading number before any separator character.
    # Common separators are " - ", ": ", or just a space followed by text.
    if cleaned[0].isdigit():
        # .split() with no argument splits on any whitespace.
        # We take the first piece [0], which would be "4" in "4 - Satisfied".
        # We strip any trailing punctuation like "." or ":" from that piece.
        leading_part = cleaned.split()[0].rstrip(".-:")
        try:
            return int(float(leading_part))
        except ValueError:
            pass

    # --- Fallback: We could not interpret this value ---
    # Returning None lets the calling code identify and log problem rows.
    print(f"Warning: Could not convert value '{raw_value}' — returning None.")
    return None



def clean_all_likert_columns(dataframe, likert_columns, label_map):
    
    # This function applies convert_likert_value() to an entire column in your DataFrame.
    #
    # Input:
    #   dataframe      : your full pandas DataFrame
    #   column_name    : the name of the column to clean, as a string
    #   label_map      : the same label-to-integer dictionary passed to convert_likert_value()
    #
    # Returns:
    #   A new pandas Series (a single column) with all values converted to integers or None.

    # We loop over each column name in the list.
    # On each iteration, "column_name" holds the current column's name as a string.
    for column_name in likert_columns:

        # Check that the column actually exists in the DataFrame before trying to clean it.
        # This prevents a confusing KeyError if there is a typo in your column list.
        if column_name not in dataframe.columns:
            print(f"Warning: Column '{column_name}' not found in DataFrame — skipping.")
            continue  # "continue" skips the rest of this iteration and moves to the next column

        # Apply the existing clean_likert_column() function to this column.
        # The result overwrites the original column in the DataFrame.
        dataframe[column_name] = clean_likert_column(dataframe, column_name, label_map)

        # Confirm each column as it is cleaned so you can spot problems early.
        print(f"Cleaned: '{column_name}'")

    return dataframe