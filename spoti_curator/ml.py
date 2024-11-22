from datetime import datetime
import os

import dotenv
import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd

from spoti_curator.constants import DEBUG_DF_PATH, ML_ASSETS_PATH, Column, Config, get_config
from spoti_curator.spoti_utils import get_songs_from_pl, login
from spoti_curator.utils import REF_SIMIL_COL_PREFIX_CONSTANT

REF_PL_STRING = 'reference playlist'

FEATURES_TO_USE = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
USE_DISTANCES_FEATS = True

REF_SONG_SIMIL_SUFFIX = 'REF_SONG_SIMIL'

def create_ml_df(sp, config, simil_df=None):
    if os.path.isfile(DEBUG_DF_PATH):
        debug_df = pd.read_csv(DEBUG_DF_PATH, sep=';')  

        debug_df[Column.TRACK_ARTISTS] = debug_df[Column.TRACK_ARTISTS].apply(lambda x: eval(x) if str(x) != 'nan' else x)

        debug_df = debug_df.fillna({Column.IS_REF_PL: 0})

        if simil_df is None:
            from spoti_curator.recommender import _feature_similarity

            # simil_df[simil_df['track_id']=='4l9xe2rcwWctjrI43UFkgA']

            aux_df = debug_df.copy()
            aux_is_ref_hack = aux_df[aux_df[Column.IS_REF_PL]==1]
            aux_is_ref_hack[Column.IS_REF_PL] = 0

            aux_df = pd.concat([aux_df, aux_is_ref_hack])

            simil_df = _feature_similarity(aux_df[[Column.TRACK_ID, Column.TRACK_ARTISTS, Column.IS_REF_PL] + FEATURES_TO_USE])

        # check what songs are in possitive class pl

        # calculate positive - negative class
        songs_in_pl_class_pl_df = get_songs_from_pl(sp, config[Config.POSSITIVE_CLASS_PL])

        positive_songs = songs_in_pl_class_pl_df[songs_in_pl_class_pl_df[Column.TRACK_ID].isin(debug_df[Column.TRACK_ID])][Column.TRACK_ID].values  

        # get distances and features (use past debug_df if possible)
        feats_and_dists = debug_df.copy()
        feats_and_dists[Column.LIKED_SONG] = (feats_and_dists[Column.TRACK_ID].isin(positive_songs) | (feats_and_dists[Column.IS_REF_PL] == 1)).astype(int)

        feats_and_dists[Column.PL_NAME] = feats_and_dists[Column.PL_NAME].fillna(REF_PL_STRING)

        feats_and_dists = feats_and_dists[
            (
                feats_and_dists[Column.PL_NAME]
                .apply(lambda x: any(y in x for y in [config[Config.RESULT_PLS]['best_matches'][Config.PL_NAME], REF_PL_STRING]))
            ) 
            | 
            (
                (feats_and_dists[Column.PL_NAME] == config[Config.RESULT_PLS]['worth_listening'][Config.PL_NAME]) & 
                (feats_and_dists[Column.LIKED_SONG] == 1)
            )
        ]

        feats_and_dists = feats_and_dists.drop(columns=['1_ref', '2_ref', '3_ref', '4_ref', '5_ref', '6_ref', '7_ref', '8_ref', '9_ref', '10_ref', 
                                                        '11_ref', '12_ref', '13_ref', '14_ref', '15_ref', '16_ref', '17_ref', '18_ref', '19_ref', '20_ref', 
                                                        '21_ref', '22_ref', '23_ref', '24_ref', '25_ref', '26_ref', '27_ref', '28_ref', '29_ref', '30_ref', 
                                                        '31_ref', '32_ref', '33_ref', '34_ref', '35_ref', '36_ref', '37_ref', '38_ref', '39_ref', 
                                                        '1_simil_ref', '2_simil_ref', '3_simil_ref', '4_simil_ref', '5_simil_ref', '6_simil_ref', 
                                                        '7_simil_ref', '8_simil_ref', '9_simil_ref', '10_simil_ref', '11_simil_ref', '12_simil_ref', 
                                                        '13_simil_ref', '14_simil_ref', '15_simil_ref', '16_simil_ref', '17_simil_ref', '18_simil_ref', 
                                                        '19_simil_ref', '20_simil_ref', '21_simil_ref', '22_simil_ref', '23_simil_ref', '24_simil_ref', 
                                                        '25_simil_ref', '26_simil_ref', '27_simil_ref', '28_simil_ref', '29_simil_ref', '30_simil_ref', 
                                                        '31_simil_ref', '32_simil_ref', '33_simil_ref', '34_simil_ref', '35_simil_ref', '36_simil_ref', 
                                                        '37_simil_ref', '38_simil_ref', '39_simil_ref', 
                                                        '1_ref_genre', '2_ref_genre', '3_ref_genre', '1_simil_ref_genre', '2_simil_ref_genre', '3_simil_ref_genre', 
                                                        '1_ref_genre_fix', '1_simil_ref_genre_fix', Column.IS_GENRE_FIX, Column.TRACK_ARTISTS], errors='ignore')
        
        simil_df.columns = [f'{ix}_{REF_SONG_SIMIL_SUFFIX}' if c not in [Column.TRACK_ID, Column.TRACK_ARTISTS] else c for ix, c in enumerate(simil_df.columns) ]

        feats_and_dists = pd.merge(feats_and_dists, simil_df, on=Column.TRACK_ID, how='left')


        # as a first version, let's use ML on features or distances
        #feats_and_dists['liked_song'].value_counts(normalize=True)

        #feats_and_dists.shape

        # save ml_df concatening it with prev ml_df   

        # for track_id, track_artists in feats_and_dists[[Column.TRACK_ID, Column.TRACK_ARTISTS]].items():
        #     # get song or artist clip
        #     #clip_file_name = get_song_clip(track_id, track_artists)

        #     # calculate embeddings (read clip_file_name, get embeddings)        

        #     # calculate custom subgenre     

        return feats_and_dists

def train_and_predict(train_df, to_pred_df):
    h2o.init()

    base_models, meta_model, cv_metric = _feats_model(train_df)
    predictions = _predict(base_models, meta_model, to_pred_df)

    h2o.shutdown()

    return predictions, cv_metric

def _train_models(df, target_column, features, n_models=100, max_time_per_model=5*60):
    simil_cols = []

    if USE_DISTANCES_FEATS:
        for c in df.columns:
            if REF_SONG_SIMIL_SUFFIX in c:
                if c not in features:
                    simil_cols.append(c)
                    features.append(c)

    # Convert the entire dataframe to an H2OFrame
    print('Train df size: ', df[features + [target_column]].dropna().shape)
    print('Target proportions: ', df[target_column].value_counts())

    more_feats = []
    max_simil = df[simil_cols].max(axis=1)
    for c in simil_cols:
        df[f'{c}_scaled'] = df[c] / max_simil
        more_feats.append(f'{c}_scaled')

    features = features + more_feats

    h2o_df = h2o.H2OFrame(df[features + [target_column]])

    # Identify the predictors and response
    predictors = [col for col in h2o_df.columns if col != target_column]
    response = target_column

    # Ensure the response column is categorical for classification
    h2o_df[response] = h2o_df[response].asfactor()

    # models = []
    # train_predictions = []

    # for i in range(n_models):
    #     # Train AutoML model
    #     aml = H2OAutoML(max_runtime_secs=max_time_per_model,
    #                     seed=i,
    #                     nfolds=0,  # This disables cross-validation
    #                     validation_fraction=0.2,  # 20% of data will be used for validation
    #                     balance_classes=False
    #                     )
    #     aml.train(x=predictors, y=response, training_frame=h2o_df)

    #     models.append(aml.leader)
        
    #     # Get predictions
    #     train_preds = aml.leader.predict(aml.training_frame)[1].as_data_frame().values.ravel()
    #     train_predictions.append(train_preds)

    # # Prepare meta-learning dataset 
    # meta_X = np.column_stack(train_predictions)
    # meta_data = h2o.H2OFrame(pd.DataFrame(meta_X, columns=[f'model_{i}' for i in range(n_models)]))
    # meta_data[response] = h2o_df[response]

    # # Train meta-model
    # meta_aml = H2OAutoML(max_runtime_secs=max_time_per_model, seed=42)
    # meta_aml.train(x=[f'model_{i}' for i in range(n_models)], y=response, training_frame=meta_data)

    # return models, meta_aml.leader

    # For the moment, only one model is trained. Let's make things simple at the beginning...

    aml = H2OAutoML(max_runtime_secs=max_time_per_model,
                    nfolds=3,
                    balance_classes=False,
                    seed=16,
                    stopping_metric='AUCPR',
                    sort_metric='AUCPR'
                    ) 
    
    aml.train(x=predictors, y=response, training_frame=h2o_df)

    bm = aml.get_best_model()

    print('Precision: ', bm.precision())
    print('Recall: ', bm.recall())

    cv_metric = float(aml.leaderboard.as_data_frame().head(1)['aucpr'][0])

    print(aml.get_leaderboard())

    return [], aml, cv_metric

def _save_automl_report(aml, output_path):
    # Get the AutoML leaderboard
    lb = aml.leaderboard
    
    # Convert to pandas DataFrame
    lb_df = lb.as_data_frame()
    
    # Save to CSV
    lb_df.to_csv(output_path, index=False)
    print(f"AutoML leaderboard saved to {output_path}")


def _predict(models, meta_model, X):
    h2o_X = h2o.H2OFrame(X.drop(columns=[Column.TRACK_ARTISTS]))

    if len(models) > 0:
        base_predictions = [model.predict(h2o_X)[1].as_data_frame().values.ravel() for model in models]
        meta_X = h2o.H2OFrame(pd.DataFrame(np.column_stack(base_predictions), 
                                        columns=[f'model_{i}' for i in range(len(models))]))
        
        return meta_model.predict(meta_X)[0].as_data_frame().values.ravel()
    else:
        return meta_model.predict(h2o_X).as_data_frame()

def _feats_model(df):       
    base_models, meta_model, cv_metric = _train_models(df, target_column=Column.LIKED_SONG, n_models=50, features=FEATURES_TO_USE)

    today = datetime.today().strftime('%Y-%m-%d')

    # Save the AutoML report for the meta-model
    _save_automl_report(meta_model, f'{ML_ASSETS_PATH}/meta_model_automl_report_{today}.csv')    

    return base_models, meta_model, cv_metric

if __name__ == '__main__':
    dotenv.load_dotenv(dotenv_path='./spoti_curator/.env')

    sp = login()

    # get yaml config
    config = get_config()    

    ml_df = create_ml_df(sp, config)    

    last_non_ref_pl = list(filter(lambda x: x != REF_PL_STRING, sorted(ml_df[Column.PL_NAME].unique())))[-1]

    train_df = ml_df[ml_df[Column.PL_NAME] != last_non_ref_pl]
    to_pred_df = ml_df[ml_df[Column.PL_NAME] == last_non_ref_pl]

    preds, metric = train_and_predict(train_df, to_pred_df)

    print(1)
