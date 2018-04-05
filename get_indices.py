import pandas as pd
import numpy as np
import requests
import tqdm
import pymorphy2
from ast import literal_eval
from model.config import Config
from model.data_utils import UNK, NUM, BEGIN, END, \
    get_glove_vocab, write_vocab, load_vocab, \
    export_trimmed_glove_vectors, get_processing_word, \
    get_vocab, get_unique_column_words, correct_sentence, \
    change_letter, unk_to_normal_form, sentence_to_indices, \
    merge_context_and_reply



config = Config(load=False)

train_ids = np.load(config.train_indices)
test_ids = np.load(config.test_indices)
val_ids = np.load(config.val_indices)



data = pd.read_csv(config.path_to_train_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None)
data.columns = config.train_column_names

public = pd.read_csv(config.path_to_test_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None)
public.columns = config.test_column_names


vocab = load_vocab(config.filename_words)

unk_dict = np.load(config.unk_dict).item()


data['context_2'] = data['context_2'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
data['context_1'] = data['context_1'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
data['context_0'] = data['context_0'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
data['reply'] = data['reply'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
data['one_hot_label'] = data.label.apply(lambda x: config.mapping[x])
data['weighted_label'] = [list(np.multiply(data.loc[i, 'one_hot_label'], 
                                        data.loc[i, 'confidence'])) for i in data.index]


public['context_2'] = public['context_2'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
public['context_1'] = public['context_1'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
public['context_0'] = public['context_0'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))
public['reply'] = public['reply'].apply(lambda x: sentence_to_indices(x, vocab, unk_dict))

data['merged_contexts'] = [merge_context_and_reply(data, i, ['context_2', 'context_1', 'context_0']) 
                            for i in tqdm.tqdm(data.index)]

data['contexts_and_reply'] = [merge_context_and_reply(data, i) 
                            for i in tqdm.tqdm(data.index)]

public["merged_contexts"] = [merge_context_and_reply(public, i, ['context_2', 'context_1', 'context_0']) 
                            for i in tqdm.tqdm(public.index)]

public['contexts_and_reply'] = [merge_context_and_reply(public, i) 
                            for i in tqdm.tqdm(public.index)]

data.to_csv(config.path_to_preprocessed_train, index=False)
public.to_csv(config.path_to_preprocessed_test, index=False)

train = data.loc[data['context_id'].isin(train_ids)]
test = data.loc[data['context_id'].isin(test_ids)]
val = data.loc[data['context_id'].isin(val_ids)]

train.to_csv(config.path_to_train, index=False)
test.to_csv(config.path_to_test, index=False)
val.to_csv(config.path_to_val, index=False)

