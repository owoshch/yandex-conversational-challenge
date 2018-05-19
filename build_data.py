import pandas as pd
import numpy as np
import requests
import tqdm
import pymorphy2
import os.path
from model.config_final import Config
from model.data_utils_used import UNK, NUM, get_vocab, get_embedding_vocab, unk_to_normal_form, \
                                    export_trimmed_fasttext_vectors, write_vocab, load_vocab


config = Config(load=False)

train = pd.read_csv(config.path_to_train_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None, engine='python')
train.columns = config.train_column_names
train_vocab = get_vocab(train, config.train_vocab)

print ('Train was loaded')

private = pd.read_csv(config.path_to_private_dataframe, error_bad_lines=False, sep = '[  . ? , !]?\t', 
                   header=None, engine='python')
private.columns = config.test_column_names
private_vocab = get_vocab(private, config.private_vocab)


print ('Test was loaded')

vocab_fasttext = get_embedding_vocab(config.filename_fasttext)

print ('Fasttext vocab was loaded')


if not os.path.exists(config.train_unk):
    print ('Start creating dictionary of unknown words from train:')
    train_unk_to_normal = unk_to_normal_form(train_vocab, vocab_fasttext, config.train_unk)
    np.save(config.train_unk, train_unk_to_normal)
else:
    train_unk_to_normal = np.load(config.train_unk).item()


if not os.path.exists(config.private_unk):
    print ('Start creating dictionary of unknown words from test:')
    test_unk_to_normal = unk_to_normal_form(private_vocab, vocab_fasttext, config.private_unk)
    np.save(config.private_unk, test_unk_to_normal)
else:
    test_unk_to_normal = np.load(config.private_unk).item()


final_unk_dict = {**train_unk_to_normal, **test_unk_to_normal}


np.save(config.unk_dict, final_unk_dict)

print ('Finak dictionary for unkknown words was created and stored at', config.unk_dict)

final_vocab = (train_vocab | private_vocab  | set(final_unk_dict.values())) & vocab_fasttext

final_vocab.add(UNK)
final_vocab.add(NUM)

write_vocab(final_vocab, config.filename_words)

vocab = load_vocab(config.filename_words)

export_trimmed_fasttext_vectors(vocab, config.filename_bin_fasttext,
                                config.filename_trimmed, config.dim_word)



