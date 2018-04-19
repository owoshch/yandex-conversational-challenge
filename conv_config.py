import os
import numpy as np

from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors,  load_vocab, \
        get_processing_word



class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()
            
    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        #self.vocab_tags  = load_vocab(self.filename_tags)
        #self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        #self.nchars     = len(self.vocab_chars)
        self.ntags      = 3

        # 2. get processing functions that map str -> id
        self.unk_dictionary = np.load(self.unk_dict).item()
        self.processing_word = get_processing_word(self.vocab_words,
                lowercase=True, chars=self.use_chars)
        #self.processing_tag  = get_processing_word(self.vocab_tags,
        #        lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                if self.use_pretrained else None)
        
        
    dir_output = "results/conv_regression/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"
        
    #filename_words = "../data/vocab.txt"
    filename_words = "./vocab.txt"

    path_to_embedding_vectors = "../data/wiki.ru.vec" #fasttext embedding
    filename_glove = "../data/wiki.ru.vec"
    filename_bin_fasttext = "../data/wiki.ru.bin"
    #filename_trimmed = "../data/embedding_vectors.npy.npz"
    filename_trimmed = "../data/fasttext_embedding_vectors.npy.npz"

    # train and test imports

    train_column_names = ['context_id', 'context_2', 'context_1', 
                           'context_0', 'reply_id', 'reply', 'label', 'confidence']

    path_to_train_dataframe = "../data/train.tsv"
    train_vocab = "../data/train_vocab.npy"

    train_unk = "../data/train_unk_dict.npy"
    
    test_column_names = ['context_id', 'context_2', 'context_1', 
                           'context_0', 'reply_id', 'reply']

    path_to_test_dataframe =  "../data/public.tsv"
    test_vocab = "../data/test_vocab.npy"

    test_unk = "../data/test_unk_dict.npy"

    unk_dict = "../data/unk_dict.npy"

    dim_word = 300

    train_indices = "../data/train_context_ids.npy"
    test_indices = "../data/test_context_ids.npy"
    val_indices = "../data/val_context_ids.npy"

    path_to_train = "./datasets/train_splitted.csv"
    path_to_test = "./datasets/test_splitted.csv"
    path_to_val = "./datasets/val_splitted.csv"

    path_to_not_preprocessed_train = "./datasets/train_words_dataframe.csv"
    path_to_not_preprocessed_val = "./datasets/val_words_dataframe.csv"
    path_to_not_preprocessed_test = "./datasets/test_words_dataframe.csv"

    path_to_filtered_train = "./datasets/filtered_train.csv"
    path_to_filtered_test = "./datasets/filtered_test.csv"


    path_to_preprocessed_train = "../data/train_preprocessed.csv"
    path_to_preprocessed_test = "../data/test_preprocessed.csv"
    path_to_submission = "../data/submission.txt"
    path_to_predicted_labels = "../data/predicted_labels.npy"

    path_to_xgb_models = "../data/xgb_models/"
    path_to_xgb_log = "../data/xgb_models/xgb_log.txt"
    path_to_xgb_submissions="../data/xgb_sumbissions/"

    path_to_matrix_xgb_models = "../data/matrix_xgb_models/"
    path_to_matrix_xgb_log = "../data/matrix_xgb_models/matrix_xgb_log.txt"

    mapping = {"good": [0, 0, 1], "neutral": [0, 1, 0], "bad": [1, 0, 0]}
    label_to_num = {"good": 2, "neutral": 1, "bad": 0}

    use_pretrained = True


    # training
    train_embeddings = False
    nepochs          = 3
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 512 # lstm on chars
    hidden_size_lstm = 1024  # lstm on word embeddings
    hidden_dense_dim = 300

    use_chars = False
    use_crf = False 
