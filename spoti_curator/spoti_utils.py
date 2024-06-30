import pandas as pd
import spotipy
from spoti_curator.constants import Column
from spotipy.oauth2 import SpotifyOAuth
import traceback


def login():
    scope='user-library-read playlist-modify-private playlist-read-private'

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

    return sp

def create_playlist(sp, tracks_df, user, pl_name):
    try:
        response = sp.user_playlist_create(user, pl_name, public=False, collaborative=False, description='')

        pl_created_id = response['id']

        track_ids_parameter = [f'spotify:track:{x}' for x in tracks_df[Column.TRACK_ID]]

        sp.user_playlist_add_tracks(user, pl_created_id, track_ids_parameter)
    except Exception:
        return traceback.format_exc()

def get_user_pls(sp):
    first_call = sp.current_user_playlists(limit=50)

    result = first_call['items']

    offset = 50
    if first_call['next'] is not None:
        next_call = sp.current_user_playlists(limit=50, offset=offset)

        result += next_call['items']

        if next_call['next'] is not None:
            offset += 50

    return result

def get_pl_artists(pl_df):
    return list(set([aa for a in pl_df[Column.TRACK_ARTISTS].values for aa in a]))

def get_songs_from_pl(sp, pl_url):
    pl_tracks = sp.playlist_tracks(pl_url)

    tracks = []

    for item in pl_tracks['items']:
        if item['track'] is not None:
            track = item['track']

            track_row = {Column.TRACK_ID: track['id'],
                         Column.TRACK_NAME: track['name'],
                         Column.TRACK_ARTISTS: [i['id'] for i in track['artists']],
                         Column.PL_URL: pl_url
                        }
            
            tracks.append(track_row)
        
    return pd.DataFrame(tracks)

def get_songs_feats(sp, songs_df):
    """
    Retrieves audio features for given track IDs using Spotipy in batches of 100.

    Parameters:
    sp (spotipy.Spotify): An authenticated Spotipy client instance.
    songs_df (pd.DataFrame): A DataFrame containing track IDs and a reference playlist indicator.

    Returns:
    pd.DataFrame: A DataFrame of audio features for the tracks.
    """

    feats_list = []
    track_ids = songs_df[Column.TRACK_ID].drop_duplicates().values
    is_ref_pl_dict = dict(songs_df[[Column.TRACK_ID, Column.IS_REF_PL]].drop_duplicates().values)

    artists_pl_dict = dict(songs_df[[Column.TRACK_ID, Column.TRACK_ARTISTS]].drop_duplicates(subset=[Column.TRACK_ID]).values)

    # Process in batches of 100
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        audio_features_list = sp.audio_features(batch)

        for audio_features in audio_features_list:
            if audio_features is not None:
                # Prepare the dictionary of audio features
                features = {
                    Column.TRACK_ID: audio_features['id'],
                    "danceability": audio_features['danceability'],
                    "energy": audio_features['energy'],
                    "key": audio_features['key'],
                    "loudness": audio_features['loudness'],
                    "mode": audio_features['mode'],
                    "speechiness": audio_features['speechiness'],
                    "acousticness": audio_features['acousticness'],
                    "instrumentalness": audio_features['instrumentalness'],
                    "liveness": audio_features['liveness'],
                    "valence": audio_features['valence'],
                    "tempo": audio_features['tempo'],
                    Column.IS_REF_PL: is_ref_pl_dict[audio_features['id']],
                    Column.TRACK_ARTISTS: artists_pl_dict[audio_features['id']],
                }

                feats_list.append(features)

    features_df = pd.DataFrame(feats_list)

    return features_df


def get_song_clip():
    pass