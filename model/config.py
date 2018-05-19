import os
import numpy as np

from .general_utils import get_logger
from .data_utils_used import get_trimmed_embedding_vectors, load_vocab, \
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
        self.nwords     = len(self.vocab_words)

        # 2. get processing functions that map str -> id
        self.unk_dictionary = np.load(self.unk_dict).item()
        self.processing_word = get_processing_word(self.vocab_words,
                lowercase=True, chars=self.use_chars)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_embedding_vectors(self.filename_trimmed)
                if self.use_pretrained else None)

    dir_output = "results/bilstm_regression_model_test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    path_to_submission = dir_output + "submission.csv"

    dir_pics = "pics/"

    train_distribution_image = dir_pics + "train_distribution.png"
    private_distribution_image = dir_pics + "private_distribution.png"


    filename_words = "../data/vocab.txt"

    filename_fasttext = "../data/wiki.ru.vec"
    filename_bin_fasttext = "../data/wiki.ru.bin"
    filename_trimmed = "../data/fasttext_embedding_vectors.npy.npz"


    path_to_train_dataframe = "../data/train.tsv"
    path_to_preprocessed_train = "../data/train_preprocessed.csv"
    train_column_names = ['context_id', 'context_2', 'context_1', 
                           'context_0', 'reply_id', 'reply', 'label', 'confidence']

    train_vocab = "../data/train_vocab.npy"
    train_unk = "../data/train_unk_dict.npy"
    
    path_to_test_dataframe =  "../data/public.tsv" 
    test_column_names = ['context_id', 'context_2', 'context_1', 
                           'context_0', 'reply_id', 'reply']
    test_vocab = "../data/test_vocab.npy"
    test_unk = "../data/test_unk_dict.npy"


    path_to_private_dataframe = "../data/final.tsv"
    path_to_preprocessed_private = "../data/private_preprocessed.csv"
    private_vocab = "../data/private_vocab.npy"
    private_unk = "../data/private_unk_dict.npy"

    unk_dict = "../data/final_unk_dict.npy"

    train_ids = "../data/train_ids.npy"
    test_ids = "../data/test_ids.npy"

    path_to_train = "../data/train.csv"
    path_to_test = "../data/test.csv"


    mapping = {"good": [0, 0, 1], "neutral": [0, 1, 0], "bad": [1, 0, 0]}
    
    dim_word = 300
    use_pretrained = True


    # training
    train_embeddings = False
    nepochs          = 10
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





    




