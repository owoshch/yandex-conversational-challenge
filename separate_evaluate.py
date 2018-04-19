import time
import sys
import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from ast import literal_eval
import tqdm
from separate_NER_model import NERModel
from model.separate_config import Config
from model.data_utils import load_pairwise_testset, sort_xgb_predictions, save_submission 


config = Config()
model = NERModel(config)
model.build()

print (config.dir_model)

model.restore_session(config.dir_model)


sub = load_pairwise_testset(model.config.path_to_preprocessed_test)
sub_preds = model.predict_proba(sub)
sub_df = pd.read_csv(model.config.path_to_preprocessed_test)
save_submission("../data/nn_separate_regression_train_1_epoch_test.txt", sub_df, sort_xgb_predictions(sub_df, sub_preds))