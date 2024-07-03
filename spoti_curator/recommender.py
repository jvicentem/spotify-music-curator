import logging

logging.basicConfig(filename='spoti_recommender.log',
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.INFO)

logger = logging.getLogger()

from datetime import datetime

today = datetime.today().strftime('%Y/%m/%d')

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from spoti_curator.constants import Column, Config, get_config
from spoti_curator.spoti_utils import create_playlist, get_songs_feats, get_songs_from_pl, get_user_pls, login
from spoti_curator.utils import REF_SIMIL_COL_PREFIX, transform_simil_df




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
            songs_in_pls_df = pd.concat([songs_in_pls_df, get_songs_from_pl(sp, pl)])     

    songs_in_pls_df[Column.IS_REF_PL] = 0

    songs_in_pls_df = songs_in_pls_df.drop_duplicates(subset=Column.TRACK_ID)

    # get songs from ref playlist
    songs_in_ref_pls_df = None
    for pl in config[Config.ORIGIN_PLS][Config.REF_PL]:
        if songs_in_ref_pls_df is None:
            songs_in_ref_pls_df = get_songs_from_pl(sp, pl)
        else:
            songs_in_ref_pls_df = pd.concat([songs_in_ref_pls_df, get_songs_from_pl(sp, pl)])     

    songs_in_ref_pls_df[Column.IS_REF_PL] = 1

    songs_in_ref_pls_df = songs_in_ref_pls_df.drop_duplicates(subset=Column.TRACK_ID)

    songs_in_pls_df = pd.concat([songs_in_pls_df, songs_in_ref_pls_df])   

    # songs_artists_df = get_song_artists_df(songs_in_pls_df)

    # get song features
    songs_feats_df = get_songs_feats(sp, songs_in_pls_df)

    # calculating feature similarity for the found songs
    simil_df = _feature_similarity(songs_feats_df)

    # depending on the similitudes, songs will go to some or other pls (distance range configured)
    # get the top n songs config with the lowest similitude
    # create pls according to their config (call hard_rules to ensure this)

    # create columns for each ref_song id like this: 1_ref, 2_ref, 3_ref, the order is indicated by the similarity (higher better)
    # create columns for the similitudes of the previous ref songs like this 1_ref_simil, 2_ref_simil, 3_ref_simil
    simil_new_df = transform_simil_df(simil_df, config[Config.MAX_COMPARISONS])   

    fav_songs_df = pd.concat([get_songs_from_pl(sp, pl_url) 
                              for pl_url in config[Config.FAVED_ARTISTS_SECTION][Config.FAV_PLAYLISTS_URL]])
    
    ## apply hard rules (keep songs from faved artists)
    simil_new_df, only_hard_rules_df = _hard_rules(sp, simil_new_df, fav_songs_df, config)     

    debug_df = _create_reco_pls(sp, simil_new_df, only_hard_rules_df, config)

    # ml system: 
    # create a df from songs in the "positive class playlist" and songs from prev
    # playlists. The latter will be 0s.
    # For each song, its distance is calculated to any of the songs from "reference playlist"
    # Because each song in the reference playlist is associated with a "subgenre" that you like,
    # this will help to build a training, val and test dataset that is correctly ballanced between "subgenres"
    # If there are too much songs of a subgenre, some of them must be removed. If more are needed, we can use the
    # ones from "the reference playlist"

    # Once this is done, an ml model can be trained using AutoML, forcing to use random samples balanced by "subgenre" always.
    # If using embeddings, force that feature and distance variables are always included (or build two models and then a metamodel...)
    # The model is stored in a folder with the date, and a small txt that serves as a report of the model.
    
    # The dataframe that is built every day will be bigger and bigger on purpose, to have a big historical data.
    # It will also serve as a cache for avoiding calls into the api and calculating again embeddings.
    
    #train_df = _create_ml_df(debug_df, songs_feats_df)

    #_train_ml_model(simil_new_df)

def _feature_similarity(songs_feats_df):
    """
    Output shape: reference track ids on column names. non-reference track ids in TRACK_ID column.
    Each value is the cosine similarity between each non-ref song and each ref-song
    """
    ref_track_ids = songs_feats_df[songs_feats_df[Column.IS_REF_PL] == 1][Column.TRACK_ID].values
    non_ref_track_ids = songs_feats_df[songs_feats_df[Column.IS_REF_PL] == 0][Column.TRACK_ID].values

    non_ref_artists = songs_feats_df[songs_feats_df[Column.IS_REF_PL] == 0][Column.TRACK_ARTISTS].values

    songs_feats_df_aux = songs_feats_df.copy().drop(columns=['mode', 'key', Column.TRACK_ID])
    for c in songs_feats_df_aux.columns:
        if c not in [Column.IS_REF_PL, Column.TRACK_ARTISTS]:
            songs_feats_df_aux[c] = ((songs_feats_df_aux[c] - songs_feats_df_aux[c].min()) 
                                    /  
                                    max(songs_feats_df_aux[c].max() - songs_feats_df_aux[c].min(), 0.00001)
                                    )
            
    ref_songs = songs_feats_df_aux[songs_feats_df_aux[Column.IS_REF_PL] == 1].drop(columns=[Column.IS_REF_PL, Column.TRACK_ARTISTS])
    non_ref_songs = songs_feats_df_aux[songs_feats_df_aux[Column.IS_REF_PL] == 0].drop(columns=[Column.IS_REF_PL, Column.TRACK_ARTISTS])
            
    cosine_similarity_result = cosine_similarity(non_ref_songs, ref_songs)

    cosine_similarity_df = pd.DataFrame(cosine_similarity_result)

    cosine_similarity_df.columns = ref_track_ids
    cosine_similarity_df[Column.TRACK_ID] = non_ref_track_ids
    cosine_similarity_df[Column.TRACK_ARTISTS] = non_ref_artists

    return cosine_similarity_df            

def _create_reco_pls(sp, simil_new_df, only_hard_rules_df, config, ml=False):
    # let's know first how many pls we are going to create, and what is their configuration
    pls_to_create = config[Config.RESULT_PLS]

    pls_dfs = []

    # for each pl to create
    for _, pl in pls_to_create.items():
        pl_name = f'{pl[Config.PL_NAME]} ({today})'

        min_simil_range = min(pl[Config.SIMILITUDE_RANGE])
        max_simil_range = max(pl[Config.SIMILITUDE_RANGE])        

        ## keep only those whose highest similitude is above threshold configured
        filtered_df = simil_new_df[(simil_new_df[REF_SIMIL_COL_PREFIX(1)] > min_simil_range) 
                                   & (simil_new_df[REF_SIMIL_COL_PREFIX(1)] < max_simil_range)]
        
        filtered_df = filtered_df.sort_values(by=REF_SIMIL_COL_PREFIX(1), ascending=False).head(pl[Config.N_SONGS])

        if pl[Config.INCLUDE_FAV_ARTISTS]:        
            hr_and_filtered_df = pd.merge(filtered_df, only_hard_rules_df[[Column.TRACK_ID]], on=Column.TRACK_ID, how='outer')
        else:
            hr_and_filtered_df = filtered_df

        hr_and_filtered_df['pl_name'] = pl_name

        hr_and_filtered_df = hr_and_filtered_df.drop_duplicates(subset=Column.TRACK_ID)

        pls_dfs.append(hr_and_filtered_df)

        ## if nothing gets selected, pass
        if len(hr_and_filtered_df) == 0:
            logger.info('No songs to recommend')
        else:
            logger.info(f'Creating "{pl_name}" playlist! {len(hr_and_filtered_df)} songs in this playlist.')

            error = create_playlist(sp, hr_and_filtered_df, config[Config.USER], pl_name)

            if error is not None:
                logger.error(error)

    # create debug_df
    # it must have a track of what song goes to what pl, what was the song that made it become included, the name of the pl in which it was included
    
    debug_df = pd.concat(pls_dfs)
    
    return debug_df

def _hard_rules(sp, simil_df, fav_songs_df, config):
    """
    Simple rules that makes a song to be included in the final playlists.

    - Artist is included in fav pl are included in each pl (if enabled)
    - Songs with similitude = 1 -> song won't be included in final pl.
    - Do not add songs that were added in previous playlists
    """

    simil_df_aux = simil_df.copy()

    simil_df_aux = simil_df_aux[ ~ simil_df_aux[Column.TRACK_ID].isin( fav_songs_df[Column.TRACK_ID].values ) ]

    fav_songs_df['artists_str'] = fav_songs_df[Column.TRACK_ARTISTS].apply(lambda x: ':'.join(sorted(x)))
    simil_df_aux['artists_str'] = simil_df_aux[Column.TRACK_ARTISTS].apply(lambda x: ':'.join(sorted(x)))

    count_artists_songs = (fav_songs_df[[Column.TRACK_ID, 'artists_str']]     
     .groupby('artists_str')
     [Column.TRACK_ID]
     .count()
     .reset_index()
    )

    count_artists_songs = count_artists_songs[count_artists_songs[Column.TRACK_ID] > config[Config.FAVED_ARTISTS_SECTION][Config.MIN_SONGS_IN_FAV_PL]]

    only_hard_rules = simil_df_aux[ simil_df_aux['artists_str'].isin( count_artists_songs['artists_str'].values ) ].drop(columns=['artists_str'])

    simil_df_aux = simil_df_aux.drop(columns=['artists_str'])

    # remove previously added songs
    user_pls = get_user_pls(sp)
    pl_to_create_names = [v[Config.PL_NAME] for _, v in config[Config.RESULT_PLS].items()]

    pls_created = []

    for pl in user_pls:
        if pl['name'].strip() != '' and any([x in pl['name'] for x in pl_to_create_names]):
            pls_created.append(pl['id'])

    prev_pls_songs = []
    for pl_id in pls_created:
        prev_pls_songs.append(get_songs_from_pl(sp, pl_id))

    prev_pls_songs = pd.concat(prev_pls_songs)
    # remove from simil_df_aux and from only_hard_rules all songs that appeared in previous pls

    simil_df_aux = simil_df_aux[ ~ simil_df_aux[Column.TRACK_ID].isin( prev_pls_songs[Column.TRACK_ID].values ) ]
    only_hard_rules = only_hard_rules[ ~ only_hard_rules[Column.TRACK_ID].isin( prev_pls_songs[Column.TRACK_ID].values ) ]

    return simil_df_aux, only_hard_rules

def _create_ml_df(debug_df, songs_feats_df):
    result_df = pd.merge(debug_df, songs_feats_df, on=Column.TRACK_ID, how='inner')

    # calculate embeddings

    # calculate subgenres

    # calculate positive - negative class
    pass

def _train_ml_model(train_df):
    pass