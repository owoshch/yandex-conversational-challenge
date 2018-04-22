import time
import sys
import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from ast import literal_eval
import tqdm
#from separate_NER_model import NERModel
from separate_variate_batch_model_negative import NERModel
from model.negative_config import Config
from model.data_utils import load_dataset, load_test_set, load_pairwise_dataset, load_pairwise_testset, save_submission



config = Config()
model = NERModel(config)
model.build()

print (config.dir_model)

model.restore_session(config.dir_model)

#train = load_pairwise_dataset(model.config.path_to_preprocessed_train)
#train_preds = model.predict_proba(train)
#np.save("../data/regressor_nn_train_preds.npy", train_preds) 



sub = load_pairwise_testset(model.config.path_to_final_preprocessed_test)
sub_df = pd.read_csv(model.config.path_to_final_preprocessed_test)
sub_preds = model.predict_proba(sub, sub_df)
#np.save("../data/regressor_nn_sub_preds.npy", sub_preds)
path_to_submission = "../data/final_negative_separate_3_epoch.txt"
print ('path to file', path_to_submission)
save_submission(path_to_submission, sub_df, sort_xgb_predictions(sub_df,  3 - np.array(sub_preds)))
