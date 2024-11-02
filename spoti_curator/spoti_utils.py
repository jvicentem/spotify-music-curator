from collections import Counter

import pandas as pd
import spotipy
from spoti_curator.constants import Column, Config
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
    call_result = sp.current_user_playlists(limit=50)

    result = call_result['items']

    while call_result['next']:
        call_result = sp.next(call_result)
        result += call_result['items']    

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

    while pl_tracks['next']:
        pl_tracks = sp.next(pl_tracks)

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

def get_prev_pls_songs(sp, config):
    user_pls = get_user_pls(sp)
    pl_to_create_names = [v[Config.PL_NAME] for _, v in config[Config.RESULT_PLS].items()]

    pls_created = []

    for pl in user_pls:
        if pl['name'].strip() != '' and any([x in pl['name'] for x in pl_to_create_names]):
            pls_created.append((pl['id'], pl['name']))

    prev_pls_songs = []
    for pl_id, pl_name in pls_created:
        pl_songs_df = get_songs_from_pl(sp, pl_id)
        pl_songs_df['pl_name'] = pl_name

        prev_pls_songs.append(pl_songs_df)

    return pd.concat(prev_pls_songs, ignore_index=True)    

def get_artists_genres(sp, artists_list, manipulate_genres=False):
    unique_artists = list(set([artists[0] for artists in artists_list]))

    # Process in batches of 100

    ref_artists_info_aux = None
    for i in range(0, len(unique_artists), 50):
        batch = unique_artists[i:i+50]

        if ref_artists_info_aux is None:
            ref_artists_info_aux = sp.artists(batch)
        else:
            ref_artists_info_aux['artists'] += sp.artists(batch)['artists']

    artists_missing_genres = {}
    for t in ref_artists_info_aux['artists']:
        if len(t['genres']) == 0:
            artists_missing_genres[t['id']] = []

    for am_id, _ in artists_missing_genres.items():
        genres = []

        rel_artists = sp.artist_related_artists(am_id)

        for ra in rel_artists['artists']:
            for g in ra['genres']:
                if manipulate_genres:
                    aux_g = g.replace('pop', '')

                    if aux_g not in genres and aux_g != '':
                        genres.append(aux_g)
                else:
                    genres.append(g)
        
        count_genres = Counter(genres)

        final_genres = []
        for c in count_genres.keys():
            if count_genres[c] > 1:
                final_genres.append(c)

            if len(final_genres) == 3:
                break

        artists_missing_genres[am_id] = final_genres

    artist_ids = []
    genres = []

    for t in ref_artists_info_aux['artists']:
        artist_ids.append(t['id'])

        if t['id'] in list(artists_missing_genres.keys()):
            genres.append(', '.join(artists_missing_genres[t['id']]))  
        else:  
            if manipulate_genres:
                genres_manipulated = [gg.replace('pop', '') for gg in t['genres'] if gg.replace('pop', '') != '']

                genres.append(', '.join(list(set(genres_manipulated))))
            else:
                genres.append(', '.join(t['genres']))

    artists_genres_df = pd.DataFrame({'artist': artist_ids, Column.GENRES: genres})

    if manipulate_genres:        
        artists_genres_df.loc[artists_genres_df['artist'] == '54R6Y0I7jGUCveDTtI21nb', Column.GENRES] = 'funk, disco, reggae, r&b'
        artists_genres_df.loc[artists_genres_df['artist'] == '6M2wZ9GZgrQXHCFfjv46we', Column.GENRES] = 'disco, funk, synth'

    return artists_genres_df

def get_song_clip():
    pass