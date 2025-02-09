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
    try:
        pl_tracks = sp.playlist_tracks(pl_url)

        tracks = []

        for item in pl_tracks['items']:
            if item['track'] is not None:
                track = item['track']

                artists_list = [i['id'] for i in track['artists'] if i['id'] is not None ]

                if len(artists_list) > 0:
                    track_row = {Column.TRACK_ID: track['id'],
                                Column.TRACK_NAME: track['name'],
                                Column.TRACK_ARTISTS: artists_list,
                                Column.PL_URL: pl_url
                                }
                    
                    tracks.append(track_row)      

        while pl_tracks['next']:
            pl_tracks = sp.next(pl_tracks)

            for item in pl_tracks['items']:
                if item['track'] is not None:
                    track = item['track']

                    artists_list = [i['id'] for i in track['artists'] if i['id'] is not None ]

                    if len(artists_list) > 0:
                        track_row = {Column.TRACK_ID: track['id'],
                                    Column.TRACK_NAME: track['name'],
                                    Column.TRACK_ARTISTS: artists_list,
                                    Column.PL_URL: pl_url
                                    }
                        
                        tracks.append(track_row)                   
            
        return pd.DataFrame(tracks)
    except:
        return None

def get_prev_pls_songs(sp, config):
    user_pls = get_user_pls(sp)
    pl_to_create_names = [v[Config.PL_NAME] for _, v in config[Config.RESULT_PLS].items()]

    pls_created = []

    for pl in user_pls:
        if pl is not None:
            if pl['name'].strip() != '' and any([x in pl['name'] for x in pl_to_create_names]):
                pls_created.append((pl['id'], pl['name']))

    prev_pls_songs = []
    for pl_id, pl_name in pls_created:
        pl_songs_df = get_songs_from_pl(sp, pl_id)
        pl_songs_df['pl_name'] = pl_name

        prev_pls_songs.append(pl_songs_df)

    return pd.concat(prev_pls_songs, ignore_index=True)    

def get_artists_genres(sp, artists_list, manipulate_genres=False):
    unique_artists = [x for x in list(set([artists[0] for artists in artists_list])) if x is not None]

    # Process in batches of 100

    ref_artists_info_aux = None
    for i in range(0, len(unique_artists), 50):
        batch = unique_artists[i:i+50]

        if ref_artists_info_aux is None:
            ref_artists_info_aux = sp.artists(batch)
        else:
            ref_artists_info_aux['artists'] += sp.artists(batch)['artists']

    artist_ids = []
    genres = []
    popularity = []

    for t in ref_artists_info_aux['artists']:
        if t is not None:
            artist_ids.append(t['id'])
            popularity.append(t['popularity'])

            # if manipulate_genres:
            #     genres_manipulated = [gg.replace('pop', '') for gg in t['genres'] if gg.replace('pop', '') != '' and gg != 'k-pop']

            #     genres.append(', '.join(list(set(genres_manipulated))))
            # else:
            genres.append(', '.join(list(set(t['genres']))))

    artists_genres_df = pd.DataFrame({'artist': artist_ids, Column.GENRES: genres, Column.POPULARITY_ARTIST: popularity})

    if manipulate_genres:        
        artists_genres_df.loc[artists_genres_df['artist'] == '54R6Y0I7jGUCveDTtI21nb', Column.GENRES] = 'funk, disco, reggae, r&b'
        artists_genres_df.loc[artists_genres_df['artist'] == '6M2wZ9GZgrQXHCFfjv46we', Column.GENRES] = 'disco, funk, synth'

    return artists_genres_df

def get_song_popularity(sp, tracks_list):
    unique_tracks = [x for x in list(set(tracks_list)) if x is not None]

    # Process in batches

    track_ids = []
    popularities = []
    for i in range(0, len(unique_tracks), 50):
        batch = unique_tracks[i:i+50]

        ref_tracks_info_aux = sp.tracks(batch)

        track_ids += batch
        popularities += [x['popularity'] if x is not None and 'popularity' in x else 0.0 for x in ref_tracks_info_aux['tracks']]
    
    return pd.DataFrame({Column.TRACK_ID: track_ids, Column.POPULARITY_SONG: popularities})