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
from scipy.spatial.distance import euclidean

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




def load_dataset(path_to_data):
    data = pd.read_csv(path_to_data)
    sentences = [literal_eval(sentence) for sentence in data['contexts_and_reply']]
    tags = [literal_eval(tag) for tag in data['one_hot_label']]
    return list(zip(sentences, tags))


def load_regression_dataset(path_to_data, conf=0.999):
    label_to_num = {"good": 2, "neutral": 1, "bad": 1 - conf}
    data = pd.read_csv(path_to_data)
    sentences = [literal_eval(sentence) for sentence in data['contexts_and_reply']]
    y_labels= np.array([label_to_num[x] for x in data.label])
    tags = y_labels * data.confidence
    return list(zip(sentences, tags))


def load_test_set(path_to_data, conf=0.999):
    label_to_num = {"good": 2, "neutral": 1, "bad": 1 - conf}
    data = pd.read_csv(path_to_data)
    sentences = [literal_eval(sentence) for sentence in data['contexts_and_reply']]
    tags = np.zeros(len(sentences))
    #y_labels= np.array([label_to_num[x] for x in data.label])
    #tags = y_labels * data.confidence
    return list(zip(sentences, tags))


def get_processing_word(vocab_words=None, vocab_chars=None,
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
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

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
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f

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



def get_glove_vocab(filename):
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

def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)



def export_trimmed_fasttext_vectors(vocab, fasttext_bin_filename, trimmed_filename):
    embeddings = np.zeros([len(vocab), dim])
    f = load_model(fasttext_bin_filename)
    for word in tqdm.tqdm(vocab):
        embeddings[vocab[word]] = f.get_sentence_word(word)
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
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

def get_unique_column_words(dataframe, column_name):
    return list(dataframe[column_name].str.strip().split(' ', expand=True).stack().unique())


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
    new_word = new_word[:e_index] + change + new_word[e_index + 1:]
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




""" Reference from https://gist.github.com/bwhite/3726239
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


def ndcg_at_k(r, correct_rankings, k=None, method=0):
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
    if k is None:
        k = len(correct_rankings)
    
    dcg_max = dcg_at_k(correct_rankings, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max



def get_predictions(weighted_labels):
    weighted_labels = np.array(weighted_labels)
    max_elements = weighted_labels.max(axis=1)[:, np.newaxis]
    one_hot_labels = (weighted_labels == max_elements).astype(int)
    labels = one_hot_labels * max_elements
    return one_hot_labels, labels


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    D-dimensional vector (treat the vector as a single row) and
    for N x D matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        #raise NotImplementedError
        max_elements = np.max(x, axis=1)
        max_elements = max_elements[:, np.newaxis]
        e_x = np.exp(x - max_elements)
        rows_sums = np.sum(e_x, axis=1)
        rows_sums = rows_sums[:, np.newaxis]
        x = e_x / rows_sums
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        e_x = np.exp(x - np.max(x))
        x = e_x / np.sum(e_x)
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x



def sort_predictions(dataframe, predictitons):
    total_predictions = []
    for _id in list(dataframe.context_id.unique()):
        partition = dataframe.loc[dataframe['context_id'] == _id]
        #print (partition.index)
        #print (np.take(l, partition.index, axis=0))
        partial_preds = np.take(predictitons, partition.index, axis=0)
        p = list(enumerate(partial_preds))
        p = sorted(p, key=lambda x: -x[-1][-1])
        predicted_indices = [x[0] for x in p]
        predicted_probas = [x[1] for x in p]
        total_predictions.extend(predicted_indices)
    return total_predictions





def get_scores_and_ids(dataframe, context_id):
    label_to_num = {"good": 2, "neutral": 1, "bad": 0}
    partition = dataframe.loc[dataframe['context_id'] == context_id]
    try: 
        weighted_labels = [literal_eval(l) for l in partition.weighted_label]
        #predicted_labels = [literal_eval(l) for l in partition.predicted]
    except ValueError:
        weighted_labels = partition.weighted_label
        #predicted_labels = partition.predicted
    
    try:
        predicted_labels = [literal_eval(l) for l in partition.predicted]
    except ValueError:
        predicted_labels = partition.predicted

    correct_labels = partition.label.apply(lambda x: label_to_num[x])
    id_to_label = dict(zip(partition.reply_id, correct_labels))
    
        
    correct_ids_and_labels = list(zip(partition.reply_id, correct_labels))
    correct_ids_and_labels = sorted(correct_ids_and_labels, key=lambda x: -x[-1])
    
    
    ids_and_labels = list(zip(partition.reply_id, predicted_labels))
    ids_and_labels = sorted(ids_and_labels, key=lambda x: -x[-1])
    
    
    predicted_ids = [x[0] for x in ids_and_labels]
    predicted_labels = [id_to_label[_id] for _id in predicted_ids]
    correct_labels = [x[1] for x in correct_ids_and_labels]
    
    return predicted_ids, predicted_labels, correct_labels



def get_mean_NDCG(dataframe, predictions = None):
    if predictions is None:
        predictions = dataframe.predicted
        print ('setting predictions to predicted column in test')
    else:
        print ('type preds', type(predictions[0]))
        predictions = np.array(predictions)
        dataframe['predicted'] = predictions
    
    scores = np.array([])
    
    for line_context_id in dataframe.context_id.unique():
        _, predicted_scores, correct_scores = get_scores_and_ids(dataframe, line_context_id)
        
        if not all(v == 0 for v in correct_scores):
            context_ndcg = ndcg_at_k(predicted_scores, correct_scores, len(correct_scores))
        else:
            # if all correct predictions are bad
            context_ndcg = 1.0
            
        scores = np.append(scores, context_ndcg)
    
    final_score = np.mean(scores) * 100000
    
    return final_score



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


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def one_hot_to_num(value):
    if value == [1, 0, 0]:
        return 0
    elif value == [0, 1, 0]:
        return 1
    elif value == [0, 0, 1]:
        return 2


def get_embedding(indices, vocab):
    # indices must be a list of int indices
    try:
        indices = literal_eval(indices)
    except ValueError:
        indices = indices
    
    embedded_sentence = np.take(vocab, indices, axis=0)
    return embedded_sentence


def get_distances(dataframe, vocab, f=euclidean):
    total_distances = []
    for _id in list(dataframe.context_id.unique()):
        partition = dataframe.loc[dataframe['context_id'] == _id]
        distances = []
        context = literal_eval(partition.merged_contexts.iloc[0])
        context_vector = get_embedding(context, vocab)
        mean_context = np.mean(context_vector, axis=0)
        replies = [literal_eval(x) for x in partition.reply]
        for reply, reply_id in zip(replies, partition.reply_id):
            reply_vector = get_embedding(reply, vocab)
            mean_reply = np.mean(reply_vector, axis=0)
            #distance = (mean_context - mean_reply) ** 2
            distance = f(mean_context, mean_reply)
            #distances = np.append(distances, distance)
            distances.append(distance)
        total_distances.extend(distances)
    return total_distances 

def get_pointwise_distances(dataframe, vocab, f=euclidean):
    total_distances = []
    for _id in list(dataframe.context_id.unique()):
        partition = dataframe.loc[dataframe['context_id'] == _id]
        distances = []
        context = literal_eval(partition.merged_contexts.iloc[0])
        context_vector = get_embedding(context, vocab)
        mean_context = np.mean(context_vector, axis=0)
        replies = [literal_eval(x) for x in partition.reply]
        for reply, reply_id in zip(replies, partition.reply_id):
            reply_vector = get_embedding(reply, vocab)
            mean_reply = np.mean(reply_vector, axis=0)
            distance = (mean_context - mean_reply) ** 2
            #distance = f(mean_context, mean_reply)
            #distances = np.append(distances, distance)
            distances.append(distance)
        total_distances.extend(distances)
    return total_distances

def compute_lengths(dataframe):
    lens = []
    for _id in list(dataframe.context_id.unique()):
        partition = dataframe.loc[dataframe['context_id'] == _id]
        lens.append(len(partition))
    return lens

def sort_xgb_predictions(dataframe, predictitons):
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


def save_submission(path_to_submission, dataframe, predictions):
    with open(path_to_submission,"w+") as f:
        for k, v in (zip(dataframe.context_id.values, predictions)):
            f.write("%s %s" % (k, v))
            f.write("\n")


def indices_to_sentence(sentence, id2word):
    sentence = [id2word[x] for x in literal_eval(sentence)]
    return " ".join(sentence)


def ids_to_sentences(dataframe, config, columns = ['context_2', 'context_1', 'context_0', 
                                           'reply', 'merged_contexts', 'contexts_and_reply']):
    id2word = {value:key for key,value in config.vocab_words.items()}
    for column_name in tqdm.tqdm(columns):
        dataframe[column_name] = dataframe[column_name].apply(lambda line: indices_to_sentence(line, id2word))