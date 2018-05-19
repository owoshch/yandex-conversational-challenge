import numpy as np
import pandas as pd
import pymorphy2
import requests
import tqdm
import os
from ast import literal_eval
import itertools
from fastText import load_model
import argparse
import errno
from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn.preprocessing import normalize
import random
import copy
from sklearn import cross_validation
from operator import itemgetter 
from collections import Counter
import matplotlib.pyplot as plt

# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

BEGIN = "<s>"
END = "</s>"



# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)




"""
 Reference from https://gist.github.com/bwhite/3726239
"""

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max



def get_mean_NDCG(dataframe, predictions = None, conf=1):
    label_to_num = {"good": 2, "neutral": 1, "bad": 0}
    if predictions is None:
        predictions = dataframe.predicted
        print ('setting predictions to predicted column in test')
    else:
        print ('type preds', type(predictions[0]))
        predictions = np.array(predictions)
        dataframe['predicted'] = predictions
    scores = np.array([])
    for line_context_id in dataframe.context_id.unique():
        partition = dataframe.loc[dataframe['context_id'] == line_context_id]
        partial_preds = np.take(predictions, partition.index, axis=0)
        y_labels= np.array([label_to_num[x] for x in partition.label])
        answers = [y_labels[x] for x in partial_preds]
        if not all(v == 0 for v in answers):
            context_ndcg = ndcg_at_k(answers, len(answers))
        else:
            # if all correct predictions are bad
            context_ndcg = 1.0
        scores = np.append(scores, context_ndcg)
    final_score = np.mean(scores) * 100000
    return final_score


def compute_lengths(dataframe):
    lens = []
    for _id in list(dataframe.context_id.unique()):
        partition = dataframe.loc[dataframe['context_id'] == _id]
        lens.append(len(partition))
    return lens


def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm




def load_pairwise_dataset(path_to_data, conf=0.999, normalize=False):
    label_to_num = {"good": 3, "neutral": 1, "bad": 1 - conf}
    data = pd.read_csv(path_to_data)
    sentences = [literal_eval(sentence) for sentence in data['merged_contexts']]
    replies = [literal_eval(sentence) for sentence in data['reply']]
    y_labels= np.array([label_to_num[x] for x in data.label])
    tags = y_labels * data.confidence
    if normalize:
        tags = normalize([tags])[0]
    return list(zip(sentences, replies, tags))

def load_pairwise_testset(path_to_data):
    data = pd.read_csv(path_to_data)
    sentences = [literal_eval(sentence) for sentence in data['merged_contexts']]
    replies = [literal_eval(sentence) for sentence in data['reply']]
    #y_labels= np.array([label_to_num[x] for x in data.label])
    tags = np.zeros(len(sentences))
    return list(zip(sentences, replies, tags))


def minibatches_w_replies(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, replies, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, replies_batch, y_batch = [], [], []
    for (x, reply, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, replies_batch, y_batch
            x_batch, replies_batch, y_batch = [], [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        
        if type(reply[0]) == tuple:
            reply = zip(*reply)
        
        x_batch += [x]
        replies_batch += [reply]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, replies_batch, y_batch




def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length


def rank_predictions(dataframe, predictitons):
    total_predictions = []
    print ('type', type(predictitons[0]))
    for _id in list(dataframe.context_id.unique()):
        partition = dataframe.loc[dataframe['context_id'] == _id]
        #print (partition.index)
        #print (np.take(l, partition.index, axis=0))
        partial_preds = np.take(predictitons, partition.index, axis=0)
        p = list(enumerate(partial_preds))
        #print (p)
        p = sorted(p, key=lambda x: -x[-1])
        predicted_indices = [x[0] for x in p]
        #print (predicted_indices)
        predicted_probas = [x[1] for x in p]
        #print (predicted_probas)
        total_predictions.extend(predicted_indices)
    return total_predictions




def get_processing_word(vocab_words=None, 
                    lowercase=False, chars=False, allow_unk=True, unk_dict=None):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        if unk_dict is not None:
            if word in unk_dict:
                word = unk_dict[word]

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        return word

    return f







def get_vocab(dataframe, filename, column_names=['context_2', 'context_1', 'context_0', 'reply']):
    words = []
    for name in column_names:
        print ('getting unique words from ' + name)
        cur_words = list(dataframe[name].str.split(' ', expand=True).stack().unique())
        cur_words = [word.strip() for word in cur_words]
        #print (cur_words)
        words += cur_words
    unique_words = set(words)
    np.save(filename, unique_words)
    return unique_words

def get_embedding_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab

def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))



def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_fasttext_vectors(vocab, fasttext_bin_filename, trimmed_filename, dim=300):
    embeddings = np.zeros([len(vocab), dim])
    f = load_model(fasttext_bin_filename)
    for word in tqdm.tqdm(vocab):
        embeddings[vocab[word]] = f.get_sentence_vector(word)
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_embedding_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)




def correct_sentence(string):
    query = 'http://speller.yandex.net/services/spellservice.json/checkText?text=%s' % "+".join(string.split())
    response = requests.get(query)
    try:
        mistakes = {element['word']:element['s'][0] for element in response.json()}
    except IndexError:
        return string
    correct_string = ""
    for word in string.split():
        if word not in mistakes.keys():
            correct_string += word + " "
        else:
            correct_string += mistakes[word] + " "
    return correct_string.strip()


def change_letter(new_word, letter_to_change, change):
    e_index = new_word.index(letter_to_change)
    new_word = new_word[:e_index] + change + new_word[e_index + len(letter_to_change):]
    return new_word



def unk_to_normal_form(dataset_vocab, vocab_fasttext, path_to_dict):
    changes = dict()
    morph = pymorphy2.MorphAnalyzer()
    unfound_words = dataset_vocab - vocab_fasttext
    for word in tqdm.tqdm(unfound_words):
        new_word = word
        if '€' in new_word:
            new_word = change_letter(new_word, '€', 'я')
        
        elif '√' in new_word:
            new_word = change_letter(new_word, '√', 'г')
            
        elif new_word == '<CENSORED>':
            new_word = 'censored'

        elif 't' in new_word:
            new_word = change_letter(new_word, 't', 'т')

        elif 'ƒ' in new_word:
            new_word = change_letter(new_word, 'ƒ', 'ф')

        elif 'ьl' in new_word:
            new_word = change_letter(new_word, 'ьl', 'ы')
        
        else:
            new_word = correct_sentence(word)


        if new_word in vocab_fasttext:
            changes[word] = new_word
        else:
            p = morph.parse(new_word)[0]
            if p.normal_form in vocab_fasttext:
                changes[word] = p.normal_form
    np.save(path_to_dict, changes)
    
    return changes



def sentence_to_indices(sentence, vocab, unk_dict, preprocess_word=None): 
#                       preprocess_word=get_processing_word(vocab_words=vocab, 
#                                                            lowercase=True, unk_dict=unk_dict)):
    if preprocess_word is None:
        preprocess_word=get_processing_word(vocab_words=vocab, 
                                                            lowercase=True, unk_dict=unk_dict)

    try:
        sentence = [preprocess_word(word) for word in sentence.split()]
    except AttributeError:
        sentence = [preprocess_word(str(sentence))]
    return sentence

def get_embedded_sentence(dataframe, row_number, column_name):
    #try:
    #    sentence = literal_eval(dataframe.loc[row_number, [column_name]].values[0])
    #except ValueError:
    sentence = dataframe.loc[row_number, [column_name]].values[0]
    return sentence


def merge_context_and_reply(dataframe, row_number,
                           column_names=['context_2', 'context_1', 'context_0', 'reply']):
    ids = [get_embedded_sentence(dataframe, row_number, name) for name in column_names]
    return list(itertools.chain.from_iterable(ids))



def get_filtered_words(dataset, indices, 
                       columns=['context_2', 'context_1', 'context_0']):
    lenghts = []
    dataset.head()
    for line in dataset.loc[indices, columns].values:
        filtered_words = ("".join(line[~pd.isnull(line)]).split())
        lenghts.append(len(filtered_words))
    return lenghts

def get_column_unique_lines(dataframe, column_name, indices):
    mask = np.roll(dataframe.loc[indices, column_name],1)!=dataframe.loc[indices,column_name]
    return list(mask[mask==True].index)



def get_dataframe_unique_lines(dataframe, indices):
    unique_lines = []
    context_2_unique_lines = get_column_unique_lines(dataframe, 'context_2', indices)
    context_1_unique_lines = get_column_unique_lines(dataframe, 'context_1', indices)
    context_0_unique_lines = get_column_unique_lines(dataframe, 'context_0', indices)
    for x in context_2_unique_lines:
        if x in context_1_unique_lines and x in context_0_unique_lines:
            unique_lines.append(x)
    return unique_lines

def get_dataset_words_distribution(dataframe, filename):
    
    line_stats_rangeindex = dataframe.context_2.isnull() * 1 + dataframe.context_1.isnull() * 1 + dataframe.context_0.isnull() * 1
    
    print (Counter(line_stats_rangeindex))
    
    single_lines_indices = line_stats_rangeindex[line_stats_rangeindex == 2].index
    two_lines_indices = line_stats_rangeindex[line_stats_rangeindex == 1].index
    three_lines_indices = line_stats_rangeindex[line_stats_rangeindex == 0].index
    
    print ('lines are counted')
    
    print ('Distribution:')
    print ('single lines', len(single_lines_indices) / len(line_stats_rangeindex))
    print ('two lines', len(two_lines_indices) / len(line_stats_rangeindex))
    print ('three lines', len(three_lines_indices) / len(line_stats_rangeindex))
    
    print ('----------------------------')
    
    
    three_lines_unique_indices = get_dataframe_unique_lines(dataframe, three_lines_indices)
    two_lines_unique_indices = get_dataframe_unique_lines(dataframe, two_lines_indices)
    single_lines_unique_indices = get_dataframe_unique_lines(dataframe, single_lines_indices)
    
    print ('unique lines are counted')
    
    num_unique_lines = len(three_lines_unique_indices) + len(two_lines_unique_indices) + len(single_lines_unique_indices)
    
    
    print ('Unique lines distribution:')
    print ('three lines', len(three_lines_unique_indices) / num_unique_lines)
    print ('two lines', len(two_lines_unique_indices) / num_unique_lines)
    print ('single lines', len(single_lines_unique_indices) / num_unique_lines)
    
    print ('-------------------------------------')

    x = get_filtered_words(dataframe, three_lines_unique_indices)
    print ('three lines filtered')
    y = get_filtered_words(dataframe, two_lines_unique_indices)
    print ('two lines filtered')
    z = get_filtered_words(dataframe, single_lines_unique_indices)
    print ('single lines filtered')
    
    print ('words are filtered')
    
    print ("Sizes:")
    print ('three lines', len(x))
    print ('two lines', len(y))
    print ('single line', len(z))
    
    print (len(x) + len(y) + len(z))
    
    print ("Average lenght:")
    print ('three lines', np.mean(x))
    print ('two lines', np.mean(y))
    print ('single lines', np.mean(z))
    
    print ("Median lenghts:")
    
    print ('three lines', np.percentile(x, 50))
    print ('two lines', np.percentile(y, 50))
    print ('single lines', np.percentile(z, 50))
    
    
    fig, axs = plt.subplots(1, 3, sharey=True)
    # We can set the number of bins with the `bins` kwarg
    axs[0].hist(x)
    axs[1].hist(y)
    axs[2].hist(z)


    axs[0].title.set_text('Three lines')
    axs[1].title.set_text('Two lines')
    axs[2].title.set_text('Single lines')

    plt.savefig(filename, dpi=300, transparent=True)
    #plt.show()
    
    return three_lines_unique_indices, two_lines_unique_indices, single_lines_unique_indices,   \
                    np.mean(x), np.mean(y), np.mean(z), \
                        np.percentile(x, 50), np.percentile(y, 50), np.percentile(z, 50)


def train_test_split(dataframe, indices, test_size=0.1):
    y = get_filtered_words(dataframe, indices)
    values2drop = []
    for key, value in Counter(y).items():   
        if value == 1:
            values2drop.append(key)
            

    valid_indices = [j for j, i in enumerate(y) if i not in values2drop]
    
    
    X = itemgetter(*valid_indices)(indices)    
    y = get_filtered_words(dataframe, X)

    
    values2drop = []
    for key, value in Counter(y).items():    # for name, age in list.items():  (for Python 3.x)
        if value == 1:
            values2drop.append(key)
    
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=test_size)

    return X_train, X_test, y_train, y_test


def save_submission(path_to_submission, dataframe, predictions):
    with open(path_to_submission,"w+") as f:
        for k, v in (zip(dataframe.context_id.values, predictions)):
            f.write("%s %s" % (k, v))
            f.write("\n")