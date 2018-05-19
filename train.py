"""
import time
import sys
import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from ast import literal_eval
import tqdm
"""
from separate_ranking_model import RankingModel
from model.config import Config
from model.data_utils import load_pairwise_dataset


config = Config()

model = RankingModel(config)
model.build()

# uncomment the line below if you want to continue training using the stored weights
# model.restore_session(model.config.dir_model)

train = load_pairwise_dataset(model.config.path_to_train)
test = load_pairwise_dataset(model.config.path_to_test)

model.train(train, test)