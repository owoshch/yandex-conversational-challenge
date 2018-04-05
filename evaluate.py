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

model.restore_session(config.dir_model)


test = load_dataset(model.config.path_to_test)

model.run_evaluate(test)