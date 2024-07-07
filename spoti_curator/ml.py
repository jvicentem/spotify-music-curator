import os

import pandas as pd

from spoti_curator.constants import DEBUG_DF_PATH, ML_DF_PATH, Column, Config
from spoti_curator.spoti_utils import get_songs_from_pl

def create_ml_df(sp, config, prev_pls_songs):
    # check what songs are in possitive class pl

    # calculate positive - negative class
    songs_in_pl_class_pl_df = get_songs_from_pl(sp, config[Config.POSSITIVE_CLASS_PL])

    positive_songs = songs_in_pl_class_pl_df[songs_in_pl_class_pl_df[Column.TRACK_ID].isin(prev_pls_songs)][Column.TRACK_ID].values

    if os.path.isfile(DEBUG_DF_PATH):
        debug_df = pd.read_pickle(DEBUG_DF_PATH)    

    # get distances and features (use past debug_df if possible)
    feats_and_dists = debug_df[debug_df[Column.IS_HARD_RULES] == 0].copy()
    feats_and_dists[Column.LIKED_SONG] = feats_and_dists[feats_and_dists[Column.TRACK_ID].isin(positive_songs).astype(int)]

    for track_id, track_artists in feats_and_dists[[Column.TRACK_ID, Column.TRACK_ARTISTS]].items():
        # get song or artist clip
        #clip_file_name = get_song_clip(track_id, track_artists)

        # calculate embeddings (read clip_file_name, get embeddings)        

        # calculate custom subgenre

    # save ml_df concatening it with prev ml_df    

    pass