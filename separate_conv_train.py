import time
import sys
import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from ast import literal_eval
import tqdm
from separate_conv_model import NERModel
from model.conv_config import Config
from model.data_utils import load_dataset, load_test_set, load_regression_dataset, \
                                sort_xgb_predictions, save_submission


config = Config()
model = NERModel(config)
model.build()

train = load_regression_dataset(model.config.path_to_train)
test = load_regression_dataset(model.config.path_to_test)
val = load_regression_dataset(model.config.path_to_val)

model.train(train, test)


# Print evaluating on test set

model.restore_session(config.dir_model)


sub = load_test_set(model.config.path_to_preprocessed_test)
sub_preds = model.predict_proba(sub)


save_submission("../data/conv_preds.txt", sub_df, sort_xgb_predictions(sub_df, sub_preds))
