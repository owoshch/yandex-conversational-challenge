import pandas as pd
import numpy as np
import requests
import tqdm
import pymorphy2
from model.config import Config
from model.data_utils import UNK, NUM, BEGIN, END, \
    get_glove_vocab, write_vocab, load_vocab, \
    export_trimmed_glove_vectors, get_processing_word, \
    get_vocab, get_unique_column_words, correct_sentence, \
    change_letter, unk_to_normal_form


config = Config(load=False)

train = pd.read_csv(config.path_to_train_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None)
train.columns = config.train_column_names
train_vocab = get_vocab(train, config.train_vocab)

public = pd.read_csv(config.path_to_test_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None)
public.columns = config.test_column_names
test_vocab = get_vocab(public, config.test_vocab)

vocab_fasttext = get_glove_vocab(config.path_to_embedding_vectors)


train_unk_to_normal = unk_to_normal_form(train_vocab, vocab_fasttext, config.train_unk)
test_unk_to_normal = unk_to_normal_form(test_vocab, vocab_fasttext, config.test_unk)
unk_dict = {**train_unk_to_normal, **test_unk_to_normal}

np.save(config.unk_dict, unk_dict)

vocab = (train_vocab | test_vocab  | set(unk_dict.keys())) & vocab_fasttext
vocab.add(UNK)
vocab.add(NUM)
vocab.add(BEGIN)
vocab.add(END)


write_vocab(vocab, config.filename_words)

vocab = load_vocab(config.filename_words)
export_trimmed_glove_vectors(vocab, config.filename_glove,
                                config.filename_trimmed, config.dim_word)


