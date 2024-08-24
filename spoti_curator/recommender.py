import os
import logging

from spoti_curator.ml import create_ml_df, train_and_predict, FEATURES_TO_USE

logging.basicConfig(filename='spoti_recommender.log',
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.INFO)

logger = logging.getLogger()

from datetime import datetime

today = datetime.today().strftime('%Y/%m/%d')

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from spoti_curator.constants import DEBUG_DF_PATH, Column, Config, get_config
from spoti_curator.spoti_utils import create_playlist, get_prev_pls_songs, get_songs_feats, get_songs_from_pl, get_user_pls, login
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
                              for pl_url in config[Config.FAVED_ARTISTS_SECTION][Config.FAV_PLAYLISTS_URL]], ignore_index=True)
    
    ## apply hard rules (keep songs from faved artists)
    simil_new_df, only_hard_rules_df, prev_pls_songs = _hard_rules(sp, simil_new_df, fav_songs_df, config)     

    debug_df = create_reco_pls(sp, simil_new_df, only_hard_rules_df, config, songs_feats_df) #TODO: run everything until here, comenting the actual pl creation, screenshot of debug_df shape

    _save_debug_df(debug_df)

def _feature_similarity(songs_feats_df):
    """
    Output shape: reference track ids on column names. non-reference track ids in TRACK_ID column.
    Each value is the cosine similarity between each non-ref song and each ref-song
    """
    ref_df = songs_feats_df[songs_feats_df[Column.IS_REF_PL] == 1]
    ref_track_ids = ref_df[Column.TRACK_ID].values
    non_ref_track_ids = songs_feats_df[songs_feats_df[Column.IS_REF_PL] == 0][Column.TRACK_ID].values

    non_ref_artists = songs_feats_df[songs_feats_df[Column.IS_REF_PL] == 0][Column.TRACK_ARTISTS].values

    songs_feats_df_aux = songs_feats_df.copy().drop(columns=['mode', 'key', Column.TRACK_ID])

    for c in songs_feats_df_aux.columns:
        if c not in [Column.IS_REF_PL, Column.TRACK_ARTISTS]:
            songs_feats_df_aux[c] = ((songs_feats_df_aux[c] - ref_df[c].min()) 
                                    /  
                                    max(ref_df[c].max() - ref_df[c].min(), 0.00001)
                                    )
            
    ref_songs = songs_feats_df_aux[songs_feats_df[Column.IS_REF_PL] == 1].drop(columns=[Column.IS_REF_PL, Column.TRACK_ARTISTS])
    non_ref_songs = songs_feats_df_aux[songs_feats_df[Column.IS_REF_PL] == 0].drop(columns=[Column.IS_REF_PL, Column.TRACK_ARTISTS])
            
    cosine_similarity_result = cosine_similarity(non_ref_songs, ref_songs)

    cosine_similarity_df = pd.DataFrame(cosine_similarity_result)

    cosine_similarity_df.columns = ref_track_ids
    cosine_similarity_df[Column.TRACK_ID] = non_ref_track_ids
    cosine_similarity_df[Column.TRACK_ARTISTS] = non_ref_artists

    return cosine_similarity_df            

def create_reco_pls(sp, simil_new_df, only_hard_rules_df, config, songs_feats_df):
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
        
        filtered_df = filtered_df.sort_values(by=REF_SIMIL_COL_PREFIX(1), ascending=False)

        if not pl[Config.USE_ML]:
            filtered_df = filtered_df.head(pl[Config.N_SONGS])

            if pl[Config.INCLUDE_FAV_ARTISTS]:        
                hr_and_filtered_df = pd.merge(filtered_df, only_hard_rules_df[[Column.TRACK_ID, Column.IS_HARD_RULES]], on=[Column.TRACK_ID, Column.IS_HARD_RULES], how='outer')
            else:
                hr_and_filtered_df = filtered_df

            hr_and_filtered_df[Column.PL_NAME] = pl_name

            hr_and_filtered_df = hr_and_filtered_df.drop_duplicates(subset=Column.TRACK_ID).sort_values(by=REF_SIMIL_COL_PREFIX(1), ascending=False)

        # ml logic (if used, always alongside the simil logic above)
        if pl[Config.USE_ML]:
            # create ml df for training
            ml_df = create_ml_df(sp, config)

            # create df of the songs to be detected
            to_pred_df = songs_feats_df[songs_feats_df[Column.TRACK_ID].isin(simil_new_df[Column.TRACK_ID])]

            preds = train_and_predict(ml_df, to_pred_df).drop(columns=['p0'])

            # concat predictions
            to_pred_with_preds = pd.concat([to_pred_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

            # add simils
            to_pred_with_preds = pd.merge(to_pred_with_preds, simil_new_df.drop(columns=[Column.TRACK_ARTISTS]), on=Column.TRACK_ID, how='inner')

            # add hard_rules
            hr_and_filtered_df_aux = pd.merge(to_pred_with_preds.drop(columns=[Column.IS_HARD_RULES]), 
                                              only_hard_rules_df[[Column.TRACK_ID, Column.IS_HARD_RULES]], on=[Column.TRACK_ID], how='outer')

            hr_and_filtered_df_aux = hr_and_filtered_df_aux.fillna({Column.PREDICTION: -1.0,  Column.PRED_P1: -1.0, Column.IS_HARD_RULES: 0})

            # make sure the final df is built correctly: redoing it again little by little
            hard_rules_songs = hr_and_filtered_df_aux[hr_and_filtered_df_aux[Column.IS_HARD_RULES] == 1]
            ok_pred_songs = hr_and_filtered_df_aux[hr_and_filtered_df_aux[Column.PREDICTION] == 1]
            ok_simil_songs = hr_and_filtered_df_aux[hr_and_filtered_df_aux[Column.TRACK_ID].isin(filtered_df[Column.TRACK_ID])]

            pred_and_simil_ok_songs = pd.concat([ok_pred_songs, ok_simil_songs], ignore_index=True).drop_duplicates(subset=Column.TRACK_ID)
            pred_and_simil_ok_songs = (pred_and_simil_ok_songs
                                       .sort_values(by=[Column.PRED_P1, REF_SIMIL_COL_PREFIX(1)], ascending=[False, False])
                                       .head(pl[Config.N_SONGS])
                                       )
            
            hr_and_filtered_df_aux = pd.concat([pred_and_simil_ok_songs, hard_rules_songs], ignore_index=True)
            
            hr_and_filtered_df = (hr_and_filtered_df_aux
                                  .dropna(subset=[Column.TRACK_ID])
                                  .drop(columns=FEATURES_TO_USE)
                                  .drop_duplicates(subset=Column.TRACK_ID)
                                  )
            
            hr_and_filtered_df[Column.PL_NAME] = pl_name

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
    
    debug_df = pd.concat(pls_dfs, ignore_index=True)

    debug_df = (pd.merge(debug_df, songs_feats_df.drop(columns=[Column.TRACK_ARTISTS, Column.IS_REF_PL]), on=Column.TRACK_ID, how='left')
                .fillna({Column.IS_REF_PL: 0, Column.PREDICTION: -1.0,  Column.PRED_P1: -1.0})
                )
    
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
    prev_pls_songs = get_prev_pls_songs(sp, config)
    
    # remove from simil_df_aux and from only_hard_rules all songs that appeared in previous pls

    simil_df_aux = simil_df_aux[ ~ simil_df_aux[Column.TRACK_ID].isin( prev_pls_songs[Column.TRACK_ID].values ) ]
    only_hard_rules = only_hard_rules[ ~ only_hard_rules[Column.TRACK_ID].isin( prev_pls_songs[Column.TRACK_ID].values ) ]

    only_hard_rules[Column.IS_HARD_RULES] = 1
    simil_df_aux[Column.IS_HARD_RULES] = 0    

    return simil_df_aux, only_hard_rules, prev_pls_songs

def _save_debug_df(debug_df):
    if os.path.isfile(DEBUG_DF_PATH):
        old_debug_df = pd.read_csv(DEBUG_DF_PATH, sep=';')
        old_debug_df[Column.TRACK_ARTISTS] = old_debug_df[Column.TRACK_ARTISTS].apply(lambda x: eval(x) if str(x) != 'nan' else x)

        debug_df = pd.concat([old_debug_df, debug_df], ignore_index=True).drop_duplicates(Column.TRACK_ID)
    
    debug_df.to_csv(DEBUG_DF_PATH, index=False, sep=';') 
