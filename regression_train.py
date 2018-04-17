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