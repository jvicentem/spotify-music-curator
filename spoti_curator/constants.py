import yaml


CONFIG_PATH = './config/config.yaml' 
ASSETS_PATH = './assets'
DEBUG_DF_PATH = f'{ASSETS_PATH}/debug_df.pkl'
ML_DF_PATH = f'{ASSETS_PATH}/ml_df.pkl'

def get_config():
    with open(CONFIG_PATH) as stream:
        return yaml.safe_load(stream)

config = get_config()

class Column:
    TRACK_ID = 'track_id'
    TRACK_NAME = 'name'
    TRACK_ARTISTS = 'artists'

    PL_URL = 'pl_url'
    IS_REF_PL = 'is_ref_pl'    

    IS_HARD_RULES = 'is_hard_rules'

    LIKED_SONG = 'liked_song'

class Config:
    ORIGIN_PLS = 'origin_playlists'
    PLS_TO_CURATE = 'playlists_to_curate'
    REF_PL = 'reference_playlist_url'
    POSSITIVE_CLASS_PL = 'positive_class_pl'

    RESULT_PLS = 'result_playlists'

    PL_NAME = 'name'
    SIMILITUDE_RANGE = 'similitude_range'
    N_SONGS = 'n_songs'
    INCLUDE_FAV_ARTISTS = 'artist_in_fav_pl_are_included'
    
    MAX_COMPARISONS = 'max_simil_comparisons'

    USER = 'user'

    FAVED_ARTISTS_SECTION = 'faved_artists'
    MIN_SONGS_IN_FAV_PL = 'min_songs_in_fav_pl'
    FAV_PLAYLISTS_URL = 'fav_playlists_url'