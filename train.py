import time
import sys
import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from ast import literal_eval
import tqdm
from ner_model import NERModel
from model.config import Config
from model.data_utils import load_dataset


config = Config()
model = NERModel(config)
model.build()

#test = load_dataset(model.config.path_to_test)
val = load_dataset(model.config.path_to_val)
train = load_dataset(model.config.path_to_train)

model.train(train, val)


# Print evaluating on test set

model.restore_session(config.dir_model)
model.create_submission()

