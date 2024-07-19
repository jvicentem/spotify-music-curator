import os

import dotenv
import pandas as pd

from spoti_curator.constants import DEBUG_DF_PATH, ML_DF_PATH, Column, Config, get_config
from spoti_curator.spoti_utils import create_playlist, get_prev_pls_songs, get_songs_feats, get_songs_from_pl, get_user_pls, login

def create_ml_df(sp, config):
    if os.path.isfile(DEBUG_DF_PATH):
        debug_df = pd.read_pickle(DEBUG_DF_PATH)  

    # check what songs are in possitive class pl

    # calculate positive - negative class
    songs_in_pl_class_pl_df = get_songs_from_pl(sp, config[Config.POSSITIVE_CLASS_PL])

    positive_songs = songs_in_pl_class_pl_df[songs_in_pl_class_pl_df[Column.TRACK_ID].isin(debug_df[Column.TRACK_ID])][Column.TRACK_ID].values  

    # get distances and features (use past debug_df if possible)
    feats_and_dists = debug_df.copy()
    feats_and_dists[Column.LIKED_SONG] = feats_and_dists[Column.TRACK_ID].isin(positive_songs).astype(int) | (feats_and_dists[Column.IS_REF_PL] == 1).astype(int)

    # as a first version, let's use ML on features

    # save ml_df concatening it with prev ml_df   

    # for track_id, track_artists in feats_and_dists[[Column.TRACK_ID, Column.TRACK_ARTISTS]].items():
    #     # get song or artist clip
    #     #clip_file_name = get_song_clip(track_id, track_artists)

    #     # calculate embeddings (read clip_file_name, get embeddings)        

    #     # calculate custom subgenre     

    pass

if __name__ == '__main__':
    dotenv.load_dotenv(dotenv_path='./spoti_curator/.env')

    sp = login()

    # get yaml config
    config = get_config()    

    ml_df = create_ml_df(sp, config)    
 
    create_ml_df()    