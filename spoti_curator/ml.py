import os

import dotenv
import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd

from spoti_curator.constants import DEBUG_DF_PATH, ML_DF_PATH, Column, Config, get_config
from spoti_curator.spoti_utils import create_playlist, get_prev_pls_songs, get_songs_feats, get_songs_from_pl, get_user_pls, login

def create_ml_df(sp, config):
    if os.path.isfile(DEBUG_DF_PATH):
        debug_df = pd.read_pickle(DEBUG_DF_PATH)  

    # check what songs are in possitive class pl

    # calculate positive - negative class
    songs_in_pl_class_pl_df = get_songs_from_pl(sp, config[Config.POSSITIVE_CLASS_PL])

    positive_songs = songs_in_pl_class_pl_df[songs_in_pl_class_pl_df[Column.TRACK_ID].isin(debug_df[Column.TRACK_ID])][Column.TRACK_ID].values  

    # get distances and features (use past debug_df if possible)
    feats_and_dists = debug_df.copy()
    feats_and_dists[Column.LIKED_SONG] = feats_and_dists[Column.TRACK_ID].isin(positive_songs).astype(int) | (feats_and_dists[Column.IS_REF_PL] == 1).astype(int)

    # as a first version, let's use ML on features or distances
    #feats_and_dists['liked_song'].value_counts(normalize=True)

    #feats_and_dists.shape

    return feats_and_dists

    # save ml_df concatening it with prev ml_df   

    # for track_id, track_artists in feats_and_dists[[Column.TRACK_ID, Column.TRACK_ARTISTS]].items():
    #     # get song or artist clip
    #     #clip_file_name = get_song_clip(track_id, track_artists)

    #     # calculate embeddings (read clip_file_name, get embeddings)        

    #     # calculate custom subgenre     

def train_and_predict(df, to_pred_df):
    h2o.init()

    base_models, meta_model = _feats_model(df)
    predictions = _predict(base_models, meta_model, to_pred_df)

    h2o.shutdown()
    
    return predictions.as_data_frame()

def _train_models(df, target_column, features, n_models=100, max_time_per_model=300):
    # Convert the entire dataframe to an H2OFrame
    h2o_df = h2o.H2OFrame(df[features])

    # Identify the predictors and response
    predictors = [col for col in h2o_df.columns if col != target_column]
    response = target_column

    # Ensure the response column is categorical for classification
    h2o_df[response] = h2o_df[response].asfactor()

    models = []
    val_predictions = []

    for i in range(n_models):
        # Train AutoML model
        aml = H2OAutoML(max_runtime_secs=max_time_per_model,
                        seed=i,
                        nfolds=0,  # This disables cross-validation
                        validation_fraction=0.25,  # 20% of data will be used for validation
                        balance_classes=False,
                        max_after_balance_size=5.0)
        aml.train(x=predictors, y=response, training_frame=h2o_df)

        models.append(aml.leader)
        
        # Get predictions on the validation set
        valid_frame = aml.validation_frame
        if valid_frame is not None:
            val_preds = aml.leader.predict(valid_frame)[1].as_data_frame().values.ravel()
            val_predictions.append(val_preds)
        else:
            print(f"Warning: No validation frame available for model {i+1}. Using training frame predictions.")
            train_preds = aml.leader.predict(aml.training_frame)[1].as_data_frame().values.ravel()
            val_predictions.append(train_preds)

    # Prepare meta-learning dataset
    meta_X = np.column_stack(val_predictions)
    meta_data = h2o.H2OFrame(pd.DataFrame(meta_X, columns=[f'model_{i}' for i in range(n_models)]))
    meta_data[response] = h2o_df[response]

    # Train meta-model
    meta_aml = H2OAutoML(max_runtime_secs=max_time_per_model, seed=42)
    meta_aml.train(x=[f'model_{i}' for i in range(n_models)], y=response, training_frame=meta_data)

    return models, meta_aml.leader

def _predict(models, meta_model, X):
    h2o_X = h2o.H2OFrame(X)
    base_predictions = [model.predict(h2o_X)[1].as_data_frame().values.ravel() for model in models]
    meta_X = h2o.H2OFrame(pd.DataFrame(np.column_stack(base_predictions), 
                                       columns=[f'model_{i}' for i in range(len(models))]))
    return meta_model.predict(meta_X)[1].as_data_frame().values.ravel()

def _feats_model(df):
    features_to_use = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    
    
    base_models, meta_model = _train_models(df, target_column=Column.LIKED_SONG, n_models=50, features=features_to_use)

    return base_models, meta_model

def _dists_model(df):
    pass

if __name__ == '__main__':
    dotenv.load_dotenv(dotenv_path='./spoti_curator/.env')

    sp = login()

    # get yaml config
    config = get_config()    

    ml_df = create_ml_df(sp, config)    
 
    create_ml_df()    