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
from separate_matrix_model import NERModel
from model.matrix_config import Config
from model.data_utils import load_dataset, load_test_set, load_pairwise_dataset, load_negative_pairwise_dataset


config = Config()

model = NERModel(config)
model.build()


model.restore_session(config.dir_model)

train = load_negative_pairwise_dataset(model.config.path_to_train, True)
test = load_negative_pairwise_dataset(model.config.path_to_test, True)
#val = load_pairwise_dataset(model.config.path_to_val)



print ('negative train target', [x[2] for x in train[0:20]])
print ('negative test target', [x[2] for x in test[0:20]])


model.train(train, test)


