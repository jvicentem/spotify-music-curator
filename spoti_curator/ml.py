from datetime import datetime
import os

import dotenv
import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd

from spoti_curator.constants import DEBUG_DF_PATH, ML_ASSETS_PATH, Column, Config, get_config
from spoti_curator.spoti_utils import get_songs_from_pl, login

REF_PL_STRING = 'reference playlist'

FEATURES_TO_USE = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

def create_ml_df(sp, config):
    if os.path.isfile(DEBUG_DF_PATH):
        debug_df = pd.read_csv(DEBUG_DF_PATH, sep=';')  

        debug_df[Column.TRACK_ARTISTS] = debug_df[Column.TRACK_ARTISTS].apply(lambda x: eval(x) if str(x) != 'nan' else x)

    # check what songs are in possitive class pl

    # calculate positive - negative class
    songs_in_pl_class_pl_df = get_songs_from_pl(sp, config[Config.POSSITIVE_CLASS_PL])

    positive_songs = songs_in_pl_class_pl_df[songs_in_pl_class_pl_df[Column.TRACK_ID].isin(debug_df[Column.TRACK_ID])][Column.TRACK_ID].values  

    # get distances and features (use past debug_df if possible)
    feats_and_dists = debug_df.copy()
    feats_and_dists[Column.LIKED_SONG] = feats_and_dists[Column.TRACK_ID].isin(positive_songs).astype(int) | (feats_and_dists[Column.IS_REF_PL] == 1).astype(int)

    feats_and_dists[Column.PL_NAME] = feats_and_dists[Column.PL_NAME].fillna(REF_PL_STRING)

    feats_and_dists = feats_and_dists[feats_and_dists[Column.PL_NAME]
                                      .apply(lambda x: 
                                             any(y in x for y in [config[Config.RESULT_PLS]['best_matches'][Config.PL_NAME], REF_PL_STRING])
                                             )]

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

    base_models, meta_model = _feats_model(train_df)
    predictions = _predict(base_models, meta_model, to_pred_df[FEATURES_TO_USE])

    h2o.shutdown()

    return predictions

def _train_models(df, target_column, features, n_models=100, max_time_per_model=5*60):
    # Convert the entire dataframe to an H2OFrame
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
                    nfolds=2,
                    balance_classes=False,
                    seed=16
                    )
    aml.train(x=predictors, y=response, training_frame=h2o_df)

    return [], aml

def _save_automl_report(aml, output_path):
    # Get the AutoML leaderboard
    lb = aml.leaderboard
    
    # Convert to pandas DataFrame
    lb_df = lb.as_data_frame()
    
    # Save to CSV
    lb_df.to_csv(output_path, index=False)
    print(f"AutoML leaderboard saved to {output_path}")


def _predict(models, meta_model, X):
    h2o_X = h2o.H2OFrame(X)

    if len(models) > 0:
        base_predictions = [model.predict(h2o_X)[1].as_data_frame().values.ravel() for model in models]
        meta_X = h2o.H2OFrame(pd.DataFrame(np.column_stack(base_predictions), 
                                        columns=[f'model_{i}' for i in range(len(models))]))
        
        return meta_model.predict(meta_X)[0].as_data_frame().values.ravel()
    else:
        return meta_model.predict(h2o_X).as_data_frame()

def _feats_model(df):       
    base_models, meta_model = _train_models(df, target_column=Column.LIKED_SONG, n_models=50, features=FEATURES_TO_USE)

    today = datetime.today().strftime('%Y-%m-%d')

    # Save the AutoML report for the meta-model
    _save_automl_report(meta_model, f'{ML_ASSETS_PATH}/meta_model_automl_report_{today}.csv')    

    return base_models, meta_model

if __name__ == '__main__':
    dotenv.load_dotenv(dotenv_path='./spoti_curator/.env')

    sp = login()

    # get yaml config
    config = get_config()    

    ml_df = create_ml_df(sp, config)    

    last_non_ref_pl = list(filter(lambda x: x != REF_PL_STRING, sorted(ml_df[Column.PL_NAME].unique())))[-1]

    train_df = ml_df[ml_df[Column.PL_NAME] != last_non_ref_pl]
    to_pred_df = ml_df[ml_df[Column.PL_NAME] == last_non_ref_pl]

    preds = train_and_predict(train_df, to_pred_df)

    print(1)
