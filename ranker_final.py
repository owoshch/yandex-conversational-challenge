from xgboostextension import XGBRanker, XGBFeature
from model.config import Config
import pandas as pd
import numpy as np
from model.data_utils import get_trimmed_glove_vectors, \
    get_embedding, get_distances, get_pointwise_distances, \
    compute_lengths, sort_xgb_predictions, get_mean_NDCG, save_submission
import pickle
import os

def train_ranker_save_preds(train, test, vocab, config, n_estimators=250, max_depth=15, learning_rate=0.1, \
                    subsample=0.99, conf=0.9):
    # train, test and val are the dataframes
    # vocab is a dictionary
    label_to_num = {"good": 2, "neutral": 1, "bad": 1 - conf}
    print ('label to num', label_to_num)
    ranker = XGBRanker(n_estimators=n_estimators, learning_rate=learning_rate, \
                       subsample=subsample, max_depth=max_depth)
    X_train = np.array(get_pointwise_distances(train, vocab))
    print ('Train was preprocessed')
    lengths_train = compute_lengths(train)
    y_train = np.array([label_to_num[x] for x in train.label])
    print ('Start training')
    ranker.fit(X_train, y_train, lengths_train, eval_metric=['ndcg', 'map@5-'])
    train_preds = ranker.predict(X_train, lengths_train)
    sorted_train_preds = sort_xgb_predictions(train, train_preds)
    train_ndcg = get_mean_NDCG(train, sorted_train_preds)
    print ('Train NDCG:', train_ndcg)
    
    print ('Start test')
    X_test = np.array(get_pointwise_distances(test, vocab))
    print ('Test dataframe was preprocessed')
    lengths_test = compute_lengths(test)
    test_preds = ranker.predict(X_test, lengths_test)
    print ('Test predictions were computed')
    sorted_test_preds = sort_xgb_predictions(test, test_preds)
    
    path_to_xgb_model = "xgb_n_estimators_%s_depth_%s_lr_%s_subsample_%s_conf_%s_train_%s" % \
    (n_estimators, max_depth, learning_rate, subsample, conf, train_ndcg)
    
    if not os.path.exists(config.path_to_xgb_submissions):
        os.makedirs(config.path_to_xgb_submissions)
    
    save_submission(config.path_to_xgb_submissions + path_to_xgb_model + ".txt", test, sorted_test_preds)
    
    path_to_xgb_model = config.path_to_xgb_models + path_to_xgb_model
    path_to_xgb_model += ".pickle.dat"
    
    print ('Saving model')
    pickle.dump(ranker, open(path_to_xgb_model, "wb"))
    
    return train_ndcg

if __name__ == "__main__":
    config = Config()
    train = pd.read_csv(config.path_to_preprocessed_train)
    test = pd.read_csv(config.path_to_preprocessed_test)
    vocab = get_trimmed_glove_vectors(config.filename_trimmed)
    print ('Data is loaded')
    n_estimators = 2
    max_depth = 3
    learning_rate = 0.1
    subsample = 0.9
    conf = 0.99
    for max_depth in [5, 15, 30, 50, 100]:
        for n_estimators in [150, 250, 300, 350, 400, 450, 500]:    
        train_ndcg = train_ranker_save_preds(train, test, vocab, config, \
                        conf=conf, max_depth=max_depth, n_estimators=n_estimators)