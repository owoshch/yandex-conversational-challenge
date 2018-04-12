from xgboostextension import XGBRanker, XGBFeature
from model.config import Config
import pandas as pd
import numpy as np
from model.data_utils import get_trimmed_glove_vectors, \
    get_embedding, get_distances, get_pointwise_distances, \
    compute_lengths, sort_xgb_predictions, get_mean_NDCG
import pickle
import os


def train_xgb_ranker(train, test, val, vocab, config, n_estimators=150, max_depth=3, learning_rate=0.1, \
                    subsample=0.9, conf=0.9):
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
    
    print ('Start validation')
    X_val = np.array(get_pointwise_distances(val, vocab))
    lengths_val = compute_lengths(val)
    val_preds = ranker.predict(X_val, lengths_val)
    sorted_val_preds = sort_xgb_predictions(val, val_preds)
    val_ndcg = get_mean_NDCG(val, sorted_val_preds)
    print ('val NDCG:', val_ndcg)

    print ('Start test')
    X_test = np.array(get_pointwise_distances(test, vocab))
    print ('Test dataframe was preprocessed')
    lengths_test = compute_lengths(test)
    test_preds = ranker.predict(X_test, lengths_test)
    print ('Test predictions were computed')
    sorted_test_preds = sort_xgb_predictions(test, test_preds)
    test_ndcg = get_mean_NDCG(test, sorted_test_preds)
    print ('Test NDCG:', test_ndcg)
    
    path_to_xgb_model = config.path_to_xgb_models + "xgb_n_estimators_%s_depth_%s_lr_%s_subsample_%s_conf_%s_val_%s_test_%s.pickle.dat" % \
    (n_estimators, max_depth, learning_rate, subsample, conf, val_ndcg, test_ndcg)
    
    print ('Saving model')
    pickle.dump(ranker, open(path_to_xgb_model, "wb"))
    
    return train_ndcg, val_ndcg, test_ndcg, path_to_xgb_model


if __name__ == "__main__":
    config = Config()
    train = pd.read_csv(config.path_to_train)
    val = pd.read_csv(config.path_to_val)
    test = pd.read_csv(config.path_to_test)
    vocab = get_trimmed_glove_vectors(config.filename_trimmed)
    print ('Data is loaded')
    n_estimators = 150
    max_depth = 3
    learning_rate = 0.1
    subsample = 0.9
    conf = 0.9
    if not os.path.exists(config.path_to_xgb_models):
        os.makedirs(config.path_to_xgb_models)
    for n_estimators in [150, 250, 300, 350, 400, 450, 500]:
        for max_depth in [3, 5, 10, 15, 30, 50, 100]:
            for conf in [0.3, 0.5, 0.7, 0.9, 0.99]:
                train_ndcg, val_ndcg, test_ndcg, path_to_xgb_model = train_xgb_ranker(train, test, val, vocab, config, \
                                                                                     conf=conf, max_depth=max_depth, n_estimators=n_estimators)
                with open(config.path_to_xgb_log, 'a+') as log_file:
                    log_file.write('train NDCG:' + str(train_ndcg) + '\n')
                    log_file.write('val NDCG: ' + str(val_ndcg) + '\n')
                    log_file.write('test NDCG: ' + str(test_ndcg) + '\n')
                    log_file.write('_______________________\n')
                    log_file.write('n estimators: ' + str(n_estimators) + '\n')
                    log_file.write('max depth: ' + str(max_depth) + '\n')
                    log_file.write('learning rate: ' + str(learning_rate) + '\n')
                    log_file.write('subsample: ' + str(subsample) + '\n')
                    log_file.write('conf: ' + str(conf) + '\n')
                    log_file.write('path to model: ' + path_to_xgb_model + ' \n')
                    log_file.write('=======================\n')