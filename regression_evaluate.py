import time
import sys
import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from ast import literal_eval
import tqdm
from regression_dnn_model import NERModel
from model.config import Config
from model.data_utils import load_dataset, load_test_set, load_regression_dataset, save_submission, sort_xgb_predictions


config = Config()

model = NERModel(config)
model.build()

model.restore_session(config.dir_model)
sub = load_test_set(model.config.path_to_preprocessed_test)
sub_preds = model.predict_proba(sub)
sub_df = pd.read_csv(model.config.path_to_preprocessed_test)
save_submission("../data/nn_reg_one_more_layer_server.txt", sub_df, sort_xgb_predictions(sub_df, sub_preds))