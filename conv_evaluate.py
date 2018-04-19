import time
import sys
import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from ast import literal_eval
import tqdm
from conv_model import NERModel
from model.conv_config import Config
from model.data_utils import load_test_set, load_regression_dataset, sort_xgb_predictions, save_submission 


config = Config()

model = NERModel(config)
model.build()
model.restore_session(config.dir_model)

sub = load_test_set(model.config.path_to_preprocessed_test)
sub_preds = model.predict_proba(sub)

sub_df = pd.read_csv(model.config.path_to_preprocessed_test)
save_submission("../data/conv_preds.txt", sub_df, sort_xgb_predictions(sub_df, sub_preds))