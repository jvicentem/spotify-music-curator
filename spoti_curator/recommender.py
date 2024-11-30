from datetime import datetime
import os
import logging

import numpy as np


from spoti_curator.embeddings import GenreAnalyzer
from spoti_curator.ml import create_ml_df, train_and_predict, FEATURES_TO_USE

logging.basicConfig(filename='spoti_recommender.log',
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.INFO)

logger = logging.getLogger()

today = datetime.today().strftime('%Y/%m/%d')

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch

from spoti_curator.constants import DEBUG_DF_PATH, FIX_GENRE_SIMIL_SUFFIX, Column, Config, get_config
from spoti_curator.spoti_utils import create_playlist, get_prev_pls_songs, get_songs_from_pl, get_user_pls, get_artists_genres, login
from spoti_curator.utils import REF_COL_PREFIX, REF_SIMIL_COL_PREFIX, transform_simil_df




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

    # get artists genres embeddings
    artists_genres_df = get_artists_genres(sp, songs_in_pls_df[Column.TRACK_ARTISTS].to_list(), manipulate_genres=True)
    songs_in_pls_emb_df = _get_songs_genres_embeddings(songs_in_pls_df, artists_genres_df)

    songs_in_pls_no_emb_df = songs_in_pls_emb_df[songs_in_pls_emb_df[Column.GENRE_EMBEDDING].apply(lambda x: len(x) == 0)]

    songs_in_pls_emb_df = songs_in_pls_emb_df[songs_in_pls_emb_df[Column.GENRE_EMBEDDING].apply(lambda x: len(x) > 0)]

    songs_with_emb_simil_df = _get_genres_similarities(songs_in_pls_emb_df)

    simil_genres_new_df = transform_simil_df(songs_with_emb_simil_df, config[Config.MAX_COMPARISONS])
    simil_genres_new_df.columns = [c if '_ref' not in c else c + '_genre' for c in simil_genres_new_df.columns]

    fav_songs_df = pd.concat([get_songs_from_pl(sp, pl_url) 
                              for pl_url in config[Config.FAVED_ARTISTS_SECTION][Config.FAV_PLAYLISTS_URL]], ignore_index=True)
    
    ## apply hard rules (keep songs from faved artists)
    simil_new_df, only_hard_rules_df, prev_pls_songs = _hard_rules(sp, simil_genres_new_df, fav_songs_df, config)    
    
    simil_new_df[Column.IS_GENRE_FIX] = 1

    songs_in_pls_no_emb_df[Column.IS_GENRE_FIX] = 0
    songs_in_pls_no_emb_df[Column.IS_HARD_RULES] = 0

    songs_in_pls_no_emb_df[Column.POPULARITY] = songs_in_pls_no_emb_df[Column.TRACK_ARTISTS].apply(lambda x: artists_genres_df[artists_genres_df['artist'].isin(x)][Column.POPULARITY].max()  
                                                                                                   )

    songs_in_pls_no_emb_df = songs_in_pls_no_emb_df.sort_values(by=Column.POPULARITY, ascending=False)

    simil_new_df = pd.merge(simil_new_df, songs_in_pls_no_emb_df[[Column.TRACK_ID, Column.IS_GENRE_FIX, Column.IS_HARD_RULES, Column.POPULARITY]], 
                            on=[Column.TRACK_ID, Column.IS_GENRE_FIX, Column.IS_HARD_RULES], how='outer')

    debug_df = create_reco_pls(sp, simil_new_df, only_hard_rules_df, config, simil_genres_new_df)

    _save_debug_df(debug_df)

def _get_genres_similarities(songs_in_pls_emb_df: pd.DataFrame):
    ref_df = songs_in_pls_emb_df[songs_in_pls_emb_df[Column.IS_REF_PL] == 1]
    ref_track_ids = ref_df[Column.TRACK_ID].values

    non_ref_songs = songs_in_pls_emb_df[songs_in_pls_emb_df[Column.IS_REF_PL] == 0].reset_index(drop=True)

    ref_songs_embeddings_vals = ref_df[Column.GENRE_EMBEDDING].values
    non_ref_songs_with_embeddings_vals = non_ref_songs[non_ref_songs[Column.GENRE_EMBEDDING].apply(lambda x: len(x) > 0)][Column.GENRE_EMBEDDING].values

    # Convert array of arrays to 2D array
    ref_songs_2d = np.stack(ref_songs_embeddings_vals)
    non_ref_songs_2d = np.stack(non_ref_songs_with_embeddings_vals)

    # Now calculate similarity
    cosine_similarity_result = cosine_similarity(non_ref_songs_2d, ref_songs_2d)

    cosine_similarity_df = pd.DataFrame(cosine_similarity_result)

    cosine_similarity_df.columns = ref_track_ids
    cosine_similarity_df[Column.TRACK_ID] = non_ref_songs[Column.TRACK_ID]
    cosine_similarity_df[Column.TRACK_ARTISTS] = non_ref_songs[Column.TRACK_ARTISTS]

    return cosine_similarity_df       

def _get_songs_genres_embeddings(songs_in_pls_df: pd.DataFrame, artists_genres_df: pd.DataFrame):
    # Initialize the analyzer
    analyzer = GenreAnalyzer(device="cuda" if torch.cuda.is_available() else "cpu",
                             model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Process multiple texts
    artists_with_genres_df = artists_genres_df[artists_genres_df[Column.GENRES] != ''].copy()

    texts = ['Music genres that define a music band: ' + g for g in artists_with_genres_df[Column.GENRES]]
    embeddings = analyzer.get_embedding(texts)

    artists_with_genres_df[Column.GENRE_EMBEDDING] = embeddings.tolist()

    artists_genres_dict = {k: embeddings[i] for i, k in enumerate(artists_with_genres_df['artist'])}

    songs_in_pls_df[Column.GENRE_EMBEDDING] = songs_in_pls_df[Column.TRACK_ARTISTS].apply(lambda x: artists_genres_dict.get(x[0], []))

    return songs_in_pls_df
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

def create_reco_pls(sp, simil_new_df, only_hard_rules_df, config, songs_simil_df):
    # let's know first how many pls we are going to create, and what is their configuration
    result_playlists = [v for _, v in config[Config.RESULT_PLS].items()]

    # Sort list by 'order' field
    pls_to_create = sorted(result_playlists, key=lambda x: x['order'])    

    pls_dfs = []

    # for each pl to create
    for pl in pls_to_create:
        pl_name = f'{pl[Config.PL_NAME]} ({today})'

        min_simil_range = min(pl[Config.SIMILITUDE_RANGE])
        max_simil_range = max(pl[Config.SIMILITUDE_RANGE])   

        ref_simil_col = REF_SIMIL_COL_PREFIX(1) + '_genre'

        simil_new_df[ref_simil_col] = simil_new_df[ref_simil_col].round(2)    

        ## keep only those whose highest similitude is above threshold configured
        filtered_df = simil_new_df[(simil_new_df[ref_simil_col] > min_simil_range) 
                                   & (simil_new_df[ref_simil_col] < max_simil_range)].copy()
        
        if len(pls_dfs) > 0:
            already_added_songs = [item for df in pls_dfs for item in df[Column.TRACK_ID].tolist()]
            filtered_df = filtered_df[ ~filtered_df[Column.TRACK_ID].isin(already_added_songs) ]
        
        filtered_df = filtered_df.sort_values(by=ref_simil_col, ascending=False)

        no_simil_nor_fixed_df = simil_new_df[((simil_new_df[ref_simil_col] <= min_simil_range) 
                                              & 
                                              (simil_new_df[ref_simil_col] >= max_simil_range))
                                              |
                                              (simil_new_df[Column.IS_GENRE_FIX]==0)
                                              ]

        no_simil_nor_fixed_df = no_simil_nor_fixed_df.sort_values(by=[ref_simil_col, Column.POPULARITY], ascending=[False, False])

        filtered_df = pd.concat([filtered_df, no_simil_nor_fixed_df])        

        ml_flag = pl[Config.USE_ML]

        if ml_flag:
            # create ml df for training
            ml_df = create_ml_df(sp, config)

            # create df of the songs to be detected
            to_pred_df = songs_simil_df[songs_simil_df[Column.TRACK_ID].isin(simil_new_df[Column.TRACK_ID])]

            preds, cv_metric = train_and_predict(ml_df, to_pred_df)
            preds = preds.drop(columns=['p0'])        

            preds[Column.PRED_P1] = preds[Column.PRED_P1].round(2)

            if cv_metric < config[Config.ML_METRIC_THRESH]:
                ml_flag = False

                logger.info(f'ML not used! Model not good enough. Metric={cv_metric}')
        else:
            filtered_df = (pd.concat([filtered_df[filtered_df[Column.IS_GENRE_FIX]==1], 
                                        filtered_df[filtered_df[Column.IS_GENRE_FIX]==0]
                                        ])
                            .reset_index(drop=True)
                            )                            

            filtered_df = filtered_df.head(pl[Config.N_SONGS])

            if pl[Config.INCLUDE_FAV_ARTISTS]:        
                hr_and_filtered_df = pd.merge(filtered_df, only_hard_rules_df[[Column.TRACK_ID, Column.IS_HARD_RULES]], 
                                              on=[Column.TRACK_ID, Column.IS_HARD_RULES], how='outer')
            else:
                hr_and_filtered_df = filtered_df

            hr_and_filtered_df[Column.PL_NAME] = pl_name

            hr_and_filtered_df = hr_and_filtered_df.drop_duplicates(subset=Column.TRACK_ID).sort_values(by=ref_simil_col, ascending=False)

        if ml_flag: 
            # ML logic
            # concat predictions column
            to_pred_with_preds = pd.concat([to_pred_df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

            # add simils
            to_pred_with_preds = pd.merge(to_pred_with_preds, simil_new_df.drop(columns=[Column.TRACK_ARTISTS]), on=Column.TRACK_ID, how='inner')

            # add hard_rules        
            hr_and_filtered_df_aux = to_pred_with_preds.drop(columns=[Column.IS_HARD_RULES])

            if pl[Config.INCLUDE_FAV_ARTISTS]:   
                hr_and_filtered_df_aux = pd.merge(hr_and_filtered_df_aux, 
                                                  only_hard_rules_df[[Column.TRACK_ID, Column.IS_HARD_RULES]], on=[Column.TRACK_ID], how='outer')           

            hr_and_filtered_df_aux = hr_and_filtered_df_aux.fillna({Column.PREDICTION: -1.0,  Column.PRED_P1: -1.0, Column.IS_HARD_RULES: 0})

            # make sure the final df is built correctly: redoing it again little by little            
            ok_pred_songs = hr_and_filtered_df_aux[hr_and_filtered_df_aux[Column.PREDICTION] == 1]
            ok_simil_songs = hr_and_filtered_df_aux[hr_and_filtered_df_aux[Column.TRACK_ID].isin(filtered_df[Column.TRACK_ID])]

            pred_and_simil_ok_songs = pd.concat([ok_pred_songs, ok_simil_songs], ignore_index=True).drop_duplicates(subset=Column.TRACK_ID)
            pred_and_simil_ok_songs = (pred_and_simil_ok_songs
                                       .drop_duplicates(subset=Column.TRACK_ID)
                                       .sort_values(by=[Column.PRED_P1, ref_simil_col], ascending=[False, False])
                                       .head(pl[Config.N_SONGS])
                                       )
            
            if pl[Config.INCLUDE_FAV_ARTISTS]:      
                hard_rules_songs = hr_and_filtered_df_aux[hr_and_filtered_df_aux[Column.IS_HARD_RULES] == 1]
                hr_and_filtered_df_aux = pd.concat([pred_and_simil_ok_songs, hard_rules_songs], ignore_index=True)
            else:
                hr_and_filtered_df_aux = pred_and_simil_ok_songs
            
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

    debug_df[Column.IS_REF_PL] = 0

    debug_df = (debug_df
                .fillna({Column.IS_REF_PL: 0, Column.PREDICTION: -1.0,  Column.PRED_P1: -1.0, Column.IS_GENRE_FIX: 0, REF_SIMIL_COL_PREFIX(1) + FIX_GENRE_SIMIL_SUFFIX: -1.0})
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
