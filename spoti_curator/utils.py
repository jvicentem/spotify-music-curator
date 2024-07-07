from itertools import combinations

import numpy as np
import pandas as pd

from spoti_curator.constants import CONFIG_PATH, Column

REF_COL_PREFIX = lambda x: f'{x}_ref'
REF_SIMIL_COL_PREFIX = lambda x: REF_COL_PREFIX(f'{x}_simil')
        
def transform_simil_df(simil_df, max_comparisons_val):
    # Assuming your original dataframe is named 'df'
    # If not, replace 'df' with your actual dataframe name

    max_comparisons_orig_val = max_comparisons_val

    if max_comparisons_val == -1:
        max_comparisons_val = len(simil_df.columns) - 2

    ref_ids_cols = [REF_COL_PREFIX(x) for x in list(range(1, max_comparisons_val+1))]
    val_cols = [REF_SIMIL_COL_PREFIX(x) for x in list(range(1, max_comparisons_val+1))]

    # Create a copy of the dataframe to work with
    df_new = simil_df.copy()

    # Identify the columns to sort (excluding 'other_ids')
    cols_to_sort = simil_df.columns.drop([Column.TRACK_ID, Column.TRACK_ARTISTS])

    def _process_row(row):
        # Sort the row by values in descending order
        sorted_series = row.sort_values(ascending=False)
        
        # Get the top n values and their corresponding column names
        top_n = [(col, val) for col, val in sorted_series.items()]

        if max_comparisons_orig_val > 0:
            top_n = top_n[:max_comparisons_val]
        
        return top_n

    # Apply the function to each row
    df_new['aux'] = df_new[cols_to_sort].apply(_process_row, axis=1)    

    df_new = df_new.drop(columns=cols_to_sort)

    def _get_correct_ref(x, i, k=0):
        if i < len(x):
            return x[i][k]
        else:
            return None

    for i, col in enumerate(ref_ids_cols):
        df_new[col] = df_new['aux'].apply(lambda x: _get_correct_ref(x, i, 0))

    for i, col in enumerate(val_cols):
        df_new[col] = df_new['aux'].apply(lambda x: _get_correct_ref(x, i, 1))

    # Reorder the columns
    df_new = df_new[[Column.TRACK_ID, Column.TRACK_ARTISTS] + ref_ids_cols + val_cols]
    df_new = df_new.reset_index(drop=True)

    return df_new    

def get_song_artists_df(songs_df):
    # Function to generate the required combinations and new rows
    def generate_combinations(row):
        track_id = row[Column.TRACK_ID]
        artists = row[Column.TRACK_ARTISTS]
        new_rows = []
        
        # Generate all combinations of the artists list
        for i in range(1, len(artists) + 1):
            for combo in combinations(artists, i):
                new_rows.append({Column.TRACK_ID: track_id, Column.TRACK_ARTISTS: ':'.join(combo)})
        
        return new_rows

    # Apply the function to each row and create a new DataFrame
    new_df = pd.DataFrame([new_row for idx, row in songs_df.iterrows() for new_row in generate_combinations(row)])    

    return new_df