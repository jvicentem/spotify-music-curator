from datetime import datetime
import itertools
import os
import logging

import numpy as np
import pandas as pd
from openai import OpenAI

from spoti_curator.embeddings import GenreAnalyzer
from spoti_curator.ml import create_ml_df, train_and_predict, FEATURES_TO_USE

logging.basicConfig(filename='spoti_recommender.log',
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.INFO)

logger = logging.getLogger()

today = datetime.today().strftime('%Y/%m/%d')

from spoti_curator.constants import FIX_GENRE_SIMIL_SUFFIX, Column, Config, get_config
from spoti_curator.spoti_utils import create_playlist, get_prev_pls_songs, get_song_popularity, get_songs_from_pl, get_artists_genres, login


def get_chatgpt_response(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Sends a prompt to ChatGPT and retrieves the response.
    
    Args:
        prompt (str): The input prompt for ChatGPT.
        model (str): The language model to use (default is "gpt-4").

    Returns:
        str: The response from ChatGPT.
    """
    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def do_recommendation():
    """
    Call each function to generate the curated playlists.
    """

    sp = login()

    # get yaml config
    config = get_config()

    # get songs from all playlists
    songs_in_pls_df = None
    for pl in config[Config.ORIGIN_PLS][Config.PLS_TO_CURATE]:
        if songs_in_pls_df is None:
            songs_in_pls_df = get_songs_from_pl(sp, pl)
        else:
            songs_in_pls_df = pd.concat([songs_in_pls_df, get_songs_from_pl(sp, pl)], ignore_index=True)     

    songs_in_pls_df[Column.IS_REF_PL] = 0

    songs_in_pls_df = songs_in_pls_df.drop_duplicates(subset=Column.TRACK_ID)

    # get songs from ref playlist
    songs_in_ref_pls_df = None
    for pl in config[Config.ORIGIN_PLS][Config.REF_PL]:
        if songs_in_ref_pls_df is None:
            songs_in_ref_pls_df = get_songs_from_pl(sp, pl)
        else:
            songs_in_ref_pls_df = pd.concat([songs_in_ref_pls_df, get_songs_from_pl(sp, pl)], ignore_index=True)     

    songs_in_ref_pls_df[Column.IS_REF_PL] = 1

    songs_in_ref_pls_df = songs_in_ref_pls_df.drop_duplicates(subset=Column.TRACK_ID)

    songs_in_pls_df = pd.concat([songs_in_pls_df, songs_in_ref_pls_df], ignore_index=True)   

    # get prev songs
    prev_pls_songs = get_prev_pls_songs(sp, config)

    # create prompt

    ## get genres for candidate songs and ref songs
    artists_genres_df = get_artists_genres(sp, songs_in_pls_df[Column.TRACK_ARTISTS].to_list(), manipulate_genres=True) 

    ## remove all songs that appear in previous pls or like or outliers pls
    fav_songs_df = pd.concat([get_songs_from_pl(sp, pl_url) 
                              for pl_url in config[Config.FAVED_ARTISTS_SECTION][Config.FAV_PLAYLISTS_URL]], ignore_index=True)

    fav_songs_ids = fav_songs_df[Column.TRACK_ID].tolist()
    songs_in_ref_pls_ids = songs_in_ref_pls_df[Column.TRACK_ID].tolist()
    prev_added_songs_ids = prev_pls_songs[Column.TRACK_ID].tolist()
    cand_songs_in_pls_df = songs_in_pls_df[~ songs_in_pls_df[Column.TRACK_ID].isin(fav_songs_ids + songs_in_ref_pls_ids + prev_added_songs_ids)]
    cand_songs_in_pls_df = cand_songs_in_pls_df.drop_duplicates(subset=[Column.TRACK_ID])

    ## if a song has two artists, concatenate their genres
    ## songs without genre won't be suggested (removed here)
    song_genres_df = songs_in_pls_df.copy()

    def _concat_genres(artists_list):
        concated_genres = []

        for artist in artists_list:
            aux_df = artists_genres_df[artists_genres_df['artist'] == artist]

            if aux_df.shape[0] > 0:
                if aux_df[Column.GENRES].values[0] != '':
                    concated_genres += aux_df[Column.GENRES].values[0].split(',')

        return ','.join(concated_genres)

    song_genres_df[Column.GENRES_CONCAT] = songs_in_pls_df[Column.TRACK_ARTISTS].apply(lambda x: _concat_genres(x))
    song_genres_df = song_genres_df[song_genres_df[Column.GENRES_CONCAT] != '']
    
    ## from all the songs create a set of genres. 
    ref_songs_genres_df = song_genres_df[song_genres_df[Column.IS_REF_PL] == 1].copy()
    cand_songs_genres_df = song_genres_df[(song_genres_df[Column.IS_REF_PL] == 0) & 
                                          ( ~ song_genres_df[Column.TRACK_ID].isin(fav_songs_ids + songs_in_ref_pls_ids + prev_added_songs_ids) )
                                          ].copy()
    cand_songs_genres_df['artists_str'] = cand_songs_genres_df[Column.TRACK_ARTISTS].apply(lambda x: ','.join(sorted(x)))

    cand_songs_genres_unique_df = cand_songs_genres_df[['artists_str', Column.GENRES_CONCAT]].drop_duplicates().reset_index(drop=True)

    ref_songs_genres_list = ref_songs_genres_df[Column.GENRES_CONCAT].unique().tolist()

    ## The prompt will be something like "These are the genres of artists he likes (each list of genres defines one artist): 
    #   [rock, pop, indie], [dance, house, electro]"

    ## create the prompt, make it return a list with artists indices sorted by similarity to the ref songs 

    cand_genres_artists_map = {}

    for gs in cand_songs_genres_unique_df[Column.GENRES_CONCAT].unique():
        aux_art_list_df = cand_songs_genres_unique_df[cand_songs_genres_unique_df[Column.GENRES_CONCAT] == gs]['artists_str'].unique().tolist()
        cand_genres_artists_map[gs] = aux_art_list_df

    cand_genres_dict = { i: k for i, k in enumerate(cand_genres_artists_map.keys()) }

    cand_genres_str = "".join([f"{i}. {k} \n" for i, k in cand_genres_dict.items() ])

    prompt = f'''
It's your task to recommend artists based on their genre. A list of reference artists and their genres is provided and later a list of candidate artists is provided.

Here are the reference artists. Each list represents the genres of each reference artist:

{", ".join( ["[" + x + "]" for x in ref_songs_genres_list] )}

These are the candidate artists. Only their genres are presented with an index on the left.

{ cand_genres_str }

Please, do the following now: First, sort the candidate artists from most probable to be liked by the user to least. Then, place their indices in a Python list. 
Make sure you include all the {len(cand_genres_dict.keys())} candidate artists indices in the resulting list. 
Do not return any other text, just and only the Python list with the requested content.
'''
    
    ## send prompt, receive it and process it.
    response = get_chatgpt_response(prompt, model="o1-preview")

    reco_list = eval(response.strip('```').strip('python').strip('\n'))

    ## convert reco_list to song recos 
    ### if several artists are mapped to a genre string (tie), recommend the song with highest popularity. If still ties, 
    ### use artist popularity. If still ties, pick one random.

    genres_reco_list_aux = [cand_genres_dict[i] for i in reco_list]

    artists_reco_list_aux = [cand_genres_artists_map[gr] for gr in genres_reco_list_aux]
    artists_reco_list_aux = list(itertools.chain(*artists_reco_list_aux))

    ## recover artists popularity
    def _recover_artists_pop(artists_str):
        aux_list = artists_str.split(',')
        
        return artists_genres_df[artists_genres_df['artist'].isin(aux_list)][Column.POPULARITY_ARTIST].max()

    cand_songs_genres_df[Column.POPULARITY_ARTIST] = cand_songs_genres_df['artists_str'].apply(lambda x: _recover_artists_pop(x))

    ## for each artist, keep the song with the highest popularity that appear in the candidates pls
    cand_songs_popularity_df = get_song_popularity(sp, cand_songs_genres_df[Column.TRACK_ID].values)    

    cand_songs_full_df = pd.merge(cand_songs_genres_df, cand_songs_popularity_df, on=Column.TRACK_ID)

    order_map = {val: i for i, val in enumerate(artists_reco_list_aux)}
    cand_songs_full_df['rank'] = cand_songs_full_df['artists_str'].map(order_map)

    # Sort by rank while maintaining the original order within ranks
    cand_songs_full_sorted_df = cand_songs_full_df.sort_values(['rank', Column.POPULARITY_ARTIST, Column.POPULARITY_SONG], ascending=[True, False, False]).drop('rank', axis=1)
    cand_songs_full_sorted_df = cand_songs_full_sorted_df.drop_duplicates(subset='artists_str', keep='first')

    ## create reco pls   
    ### add fav artists songs if configured in yaml (and also other possible conditions in the config)
    fav_songs_df['artists_str'] = fav_songs_df[Column.TRACK_ARTISTS].apply(lambda x: ':'.join(sorted(x)))

    count_artists_songs = (fav_songs_df[[Column.TRACK_ID, 'artists_str']]     
     .groupby('artists_str')
     [Column.TRACK_ID]
     .count()
     .reset_index()
    )

    count_artists_songs = count_artists_songs[count_artists_songs[Column.TRACK_ID] > config[Config.FAVED_ARTISTS_SECTION][Config.MIN_SONGS_IN_FAV_PL]]

    must_include_cand_songs_df = cand_songs_full_sorted_df[ cand_songs_full_sorted_df['artists_str'].isin( count_artists_songs['artists_str'].values ) ]

    create_reco_pls(sp, cand_songs_full_sorted_df, must_include_cand_songs_df, config)

def create_reco_pls(sp, cand_songs_sorted_df, must_include_df, config):
    # let's know first how many pls we are going to create, and what is their configuration
    result_playlists = [v for _, v in config[Config.RESULT_PLS].items()]

    # Sort list by 'order' field
    pls_to_create = sorted(result_playlists, key=lambda x: x['order'])    

    pls_dfs = []

    prev_pl_last_index = 0

    # for each pl to create
    for pl in pls_to_create:
        pl_name = f'{pl[Config.PL_NAME]} ({today})'

        new_reco_pl_songs_df = cand_songs_sorted_df.iloc[prev_pl_last_index : pl[Config.N_SONGS]].copy()

        prev_pl_last_index = prev_pl_last_index + pl[Config.N_SONGS]

        if pl[Config.INCLUDE_FAV_ARTISTS]:
            new_reco_pl_songs_df = pd.concat([new_reco_pl_songs_df, must_include_df])

        pls_dfs.append(new_reco_pl_songs_df)

        ## if nothing gets selected, pass
        if len(new_reco_pl_songs_df) == 0:
            logger.info('No songs to recommend')
        else:
            logger.info(f'Creating "{pl_name}" playlist! {len(new_reco_pl_songs_df)} songs in this playlist.')

            error = create_playlist(sp, new_reco_pl_songs_df, config[Config.USER], pl_name)
            if error is not None:
                logger.error(error)
