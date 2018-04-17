import time
import sys
import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from ast import literal_eval
import tqdm
from regression_model import NERModel
from model.config import Config
from model.data_utils import load_dataset, load_regression_dataset


config = Config()
model = NERModel(config)
model.build()


train = load_regression_dataset(model.config.path_to_train)
test = load_regression_dataset(model.config.path_to_test)
val = load_regression_dataset(model.config.path_to_val)


t = train[:100]

model.train(t, test)


sub = load_test_set(model.config.path_to_preprocessed_test)
sub_preds = model.predict_proba(sub)

sub_df = pd.read_csv(model.config.path_to_preprocessed_test)
save_submission("../data/nn_reg_server", sub_df, sort_xgb_predictions(sub_df, sub_preds))