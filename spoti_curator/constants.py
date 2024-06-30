import yaml

def get_config():
    with open(CONFIG_PATH) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return {}


CONFIG_PATH = './config/config.yaml' 

config = get_config()

class Column:
    TRACK_ID = 'track_id'
    TRACK_NAME = 'name'
    TRACK_ARTISTS = 'artists'

    PL_URL = 'pl_url'
    IS_REF_PL = 'is_ref_pl'    

class Config:
    ORIGIN_PLS = 'origin_playlists'
    PLS_TO_CURATE = 'playlists_to_curate'
    REF_PL = 'reference_playlist_url'

    RESULT_PLS = 'result_playlists'

    PL_NAME = 'name'
    SIMILITUDE_RANGE = 'similitude_range'
    N_SONGS = 'n_songs'
    
    MAX_COMPARISONS = 'max_simil_comparisons'

    USER = 'user'

    FAVED_ARTISTS_SECTION = 'faved_artists'
    MIN_SONGS_IN_FAV_PL = 'min_songs_in_fav_pl'
    FAV_PLAYLISTS_URL = 'fav_playlists_url'