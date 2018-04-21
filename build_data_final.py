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
    change_letter, unk_to_normal_form, export_trimmed_fasttext_vectors
import os.path

config = Config(load=False)

train = pd.read_csv(config.path_to_train_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None)
train.columns = config.train_column_names
train_vocab = get_vocab(train, config.train_vocab)

private = pd.read_csv(config.path_to_private_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None)
private.columns = config.test_column_names
private_vocab = get_vocab(private, config.private_vocab)

vocab_fasttext = get_glove_vocab(config.path_to_embedding_vectors)

if not os.path.exists(config.train_unk):
    train_unk_to_normal = unk_to_normal_form(train_vocab, vocab_fasttext, config.train_unk)
    np.save(config.train_unk, train_unk_to_normal)
else:
    train_unk_to_normal = np.load(config.train_unk).item()
print ('train unk len', len(train_unk_to_normal))

if not os.path.exists(config.private_unk):
    test_unk_to_normal = unk_to_normal_form(private_vocab, vocab_fasttext, config.private_unk)
    np.save(config.private_unk, test_unk_to_normal)
else:
    test_unk_to_normal = np.load(config.private_unk).item()

print ('test unk len', len(test_unk_to_normal))

final_unk_dict = {**train_unk_to_normal, **test_unk_to_normal}

print ('final unk dict length', len(final_unk_dict))

np.save(config.final_unk_dict, final_unk_dict)

final_vocab = (train_vocab | private_vocab  | set(final_unk_dict.values())) & vocab_fasttext

final_vocab.add(UNK)
final_vocab.add(NUM)
final_vocab.add(BEGIN)
final_vocab.add(END)

write_vocab(final_vocab, config.filename_words_final)


final_vocab = load_vocab(config.filename_words_final)


export_trimmed_fasttext_vectors(final_vocab, config.filename_bin_fasttext,
                                config.filename_trimmed, config.dim_word)