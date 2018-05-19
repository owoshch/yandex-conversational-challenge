import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import pandas as pd
from collections import Counter
from ast import literal_eval
import itertools
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
#from operator import itemgetter 
#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn import cross_validation
from model.config import Config
from model.data_utils import get_dataset_words_distribution, train_test_split

config = Config(load=False)

train = pd.read_csv(config.path_to_train_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None, engine='python')
train.columns = config.train_column_names


public = pd.read_csv(config.path_to_test_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None, engine='python')
public.columns = config.test_column_names


private = pd.read_csv(config.path_to_private_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None, engine='python')
private.columns = config.test_column_names

print ('Excluding public rows from private')

private = private[~private.context_id.isin(public.context_id.values)]

print ('Private shape', private.shape)

if not os.path.exists(config.dir_pics):
        os.makedirs(config.dir_pics)

print ('--- Train distribution ---')

train_three_lines_unique_indices, train_two_lines_unique_indices, train_single_lines_unique_indices, \
train_mean_three_lines, train_mean_two_lines, train_mean_single_lines, train_median_three_lines, \
train_median_two_lines, train_median_single_lines = get_dataset_words_distribution(train, config.train_distribution_image)

print ('--- Test distibution ---')

test_three_lines_unique_indices, test_two_lines_unique_indices, test_single_lines_unique_indices, \
test_mean_three_lines, test_mean_two_lines, test_mean_single_lines, \
test_median_three_lines, test_median_two_lines, test_median_single_lines = get_dataset_words_distribution(private, config.private_distribution_image)



three_X_train, three_X_test, three_y_train, three_y_test = train_test_split(train, train_three_lines_unique_indices)
two_X_train, two_X_test, two_y_train, two_y_test = train_test_split(train, train_two_lines_unique_indices)
single_X_train, single_X_test, single_y_train, single_y_test = train_test_split(train, train_single_lines_unique_indices)
X_train = np.concatenate([three_X_train, two_X_train, single_X_train], axis=0)
X_test = np.concatenate([three_X_test, two_X_test, single_X_test], axis=0)

train_ids = train.loc[X_train, 'context_id']
test_ids = train.loc[X_test, 'context_id']

print ('Train shape', len(train_ids))
print ('Validation shape', len(test_ids))

np.save(config.train_ids, train_ids)
np.save(config.test_ids, test_ids)