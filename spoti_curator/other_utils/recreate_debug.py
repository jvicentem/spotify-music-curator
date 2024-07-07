import pandas as pd
from spoti_curator.constants import DEBUG_DF_PATH, Column, Config, get_config
from spoti_curator.recommender import _feature_similarity, _hard_rules
from spoti_curator.spoti_utils import get_prev_pls_songs, get_songs_feats, get_songs_from_pl, login
from spoti_curator.utils import transform_simil_df


def recreate_debug_pkl():
    # login
    sp = login()

    config = get_config()
    
    # read prev created pls songs
    prev_pls_songs_df = get_prev_pls_songs(sp, config)

    prev_pls_songs_df[Column.IS_REF_PL] = 0

    # get songs from ref playlist
    songs_in_ref_pls_df = None
    for pl in config[Config.ORIGIN_PLS][Config.REF_PL]:
        if songs_in_ref_pls_df is None:
            songs_in_ref_pls_df = get_songs_from_pl(sp, pl)
        else:
            songs_in_ref_pls_df = pd.concat([songs_in_ref_pls_df, get_songs_from_pl(sp, pl)])     

    songs_in_ref_pls_df[Column.IS_REF_PL] = 1

    songs_in_ref_pls_df = songs_in_ref_pls_df.drop_duplicates(subset=Column.TRACK_ID)

    songs_in_pls_df = pd.concat([prev_pls_songs_df, songs_in_ref_pls_df])  

    # calculate features
    songs_feats_df = get_songs_feats(sp, songs_in_pls_df)
    
    # calculate similarities
    simil_df = _feature_similarity(songs_feats_df)
    
    # transform features df
    simil_new_df = transform_simil_df(simil_df, config[Config.MAX_COMPARISONS])   
    
    # calculate is_hard_rule column (groupby counts etc)
    fav_songs_df = pd.concat([get_songs_from_pl(sp, pl_url) 
                              for pl_url in config[Config.FAVED_ARTISTS_SECTION][Config.FAV_PLAYLISTS_URL]])
    
    fav_songs_df['artists_str'] = fav_songs_df[Column.TRACK_ARTISTS].apply(lambda x: ':'.join(sorted(x)))
    simil_new_df['artists_str'] = simil_new_df[Column.TRACK_ARTISTS].apply(lambda x: ':'.join(sorted(x)))

    count_artists_songs = (fav_songs_df[[Column.TRACK_ID, 'artists_str']]     
     .groupby('artists_str')
     [Column.TRACK_ID]
     .count()
     .reset_index()
    )

    count_artists_songs = count_artists_songs[count_artists_songs[Column.TRACK_ID] > config[Config.FAVED_ARTISTS_SECTION][Config.MIN_SONGS_IN_FAV_PL]]

    simil_new_df[Column.IS_HARD_RULES] = simil_new_df['artists_str'].isin(count_artists_songs['artists_str'].values).astype(int)
        
    # remove duplicated songs
    simil_new_df = pd.merge(simil_new_df, songs_in_pls_df[[Column.TRACK_ID, 'pl_name']], on=Column.TRACK_ID, how='inner')

    debug_df = pd.merge(simil_new_df, songs_feats_df.drop(columns=[Column.TRACK_ARTISTS]), on=Column.TRACK_ID, how='inner')

    debug_df = debug_df.drop_duplicates(subset=[Column.TRACK_ID])

    debug_df = debug_df.drop(columns=['artists_str'])
    
    # save df
    debug_df.to_pickle(DEBUG_DF_PATH)

if __name__ == '__main__':
    recreate_debug_pkl()

# ['track_id', 'artists', '1_ref', '2_ref', '3_ref', '4_ref', '5_ref',
#        '6_ref', '7_ref', '8_ref', '9_ref', '10_ref', '11_ref', '12_ref',
#        '13_ref', '14_ref', '15_ref', '16_ref', '17_ref', '18_ref', '19_ref',
#        '20_ref', '21_ref', '22_ref', '23_ref', '24_ref', '25_ref', '26_ref',
#        '27_ref', '28_ref', '29_ref', '30_ref', '31_ref', '32_ref', '33_ref',
#        '34_ref', '35_ref', '36_ref', '37_ref', '38_ref', '39_ref',
#        '1_simil_ref', '2_simil_ref', '3_simil_ref', '4_simil_ref',
#        '5_simil_ref', '6_simil_ref', '7_simil_ref', '8_simil_ref',
#        '9_simil_ref', '10_simil_ref', '11_simil_ref', '12_simil_ref',
#        '13_simil_ref', '14_simil_ref', '15_simil_ref', '16_simil_ref',
#        '17_simil_ref', '18_simil_ref', '19_simil_ref', '20_simil_ref',
#        '21_simil_ref', '22_simil_ref', '23_simil_ref', '24_simil_ref',
#        '25_simil_ref', '26_simil_ref', '27_simil_ref', '28_simil_ref',
#        '29_simil_ref', '30_simil_ref', '31_simil_ref', '32_simil_ref',
#        '33_simil_ref', '34_simil_ref', '35_simil_ref', '36_simil_ref',
#        '37_simil_ref', '38_simil_ref', '39_simil_ref', 'is_hard_rules', 'pl_name', 'danceability', 'energy',
#        'key', 'loudness', 'mode', 'speechiness', 'acousticness',
#        'instrumentalness', 'liveness', 'valence', 'tempo', 'is_ref_pl']