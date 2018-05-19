import pandas as pd
import numpy as np
import requests
import tqdm
import pymorphy2
from ast import literal_eval
from model.config_final import Config
from model.data_utils_used import load_vocab, sentence_to_indices, \
                    merge_context_and_reply

config = Config(load=False)


print ('Loading train from', config.train_ids)
train_ids = np.load(config.train_ids)
print ('Loading test from', config.test_ids)
test_ids = np.load(config.test_ids)


data = pd.read_csv(config.path_to_train_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None, engine='python')
data.columns = config.train_column_names

print ('Load public and private datasets from', config.path_to_private_dataframe)

private = pd.read_csv(config.path_to_private_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None, engine='python')
private.columns = config.test_column_names


print ('Loading vocab from', config.filename_words)
vocab = load_vocab(config.filename_words)


print ('Loading unk dictionary from', config.unk_dict)
unk_dict = np.load(config.unk_dict).item()

print ('Mapping words to indices in train dataset')

data['context_2'] = data['context_2'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
data['context_1'] = data['context_1'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
data['context_0'] = data['context_0'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
data['reply'] = data['reply'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
data['one_hot_label'] = data.label.apply(lambda x: config.mapping[x])
data['weighted_label'] = [list(np.multiply(data.loc[i, 'one_hot_label'], 
                                        data.loc[i, 'confidence'])) for i in data.index]

print ('Mapping words to indices in private dataset')

private['context_2'] = private['context_2'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
private['context_1'] = private['context_1'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
private['context_0'] = private['context_0'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
private['reply'] = private['reply'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))


data['merged_contexts'] = [merge_context_and_reply(data, i, ['context_2', 'context_1', 'context_0']) 
                            for i in tqdm.tqdm(data.index)]

data['contexts_and_reply'] = [merge_context_and_reply(data, i) 
                            for i in tqdm.tqdm(data.index)]


private["merged_contexts"] = [merge_context_and_reply(private, i, ['context_2', 'context_1', 'context_0']) 
                            for i in tqdm.tqdm(private.index)]

private['contexts_and_reply'] = [merge_context_and_reply(private, i) 
                            for i in tqdm.tqdm(private.index)]

data.to_csv(config.path_to_preprocessed_train, index=False)
private.to_csv(config.path_to_preprocessed_private, index=False)

train = data.loc[data['context_id'].isin(train_ids)]

print ('Train shape', train.shape)

test = data.loc[data['context_id'].isin(test_ids)]

print ('Test shape', test.shape)

train.to_csv(config.path_to_train, index=False)
test.to_csv(config.path_to_test, index=False)

