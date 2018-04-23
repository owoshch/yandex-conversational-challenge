from model.config import Config
import pandas as pd
import numpy as np
from model.data_utils import get_trimmed_glove_vectors, \
    get_embedding, get_distances, get_pointwise_distances, \
    compute_lengths, sort_xgb_predictions, get_mean_NDCG, save_submission, \
    indices_to_sentence, ids_to_sentences
import pickle
import os
from ast import literal_eval
import tqdm
from collections import Counter



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import SelectPercentile, chi2
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from hyperopt import hp
from hyperopt import fmin, tpe, Trials
from hyperopt import space_eval


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import nltk
#nltk.download("stopwords")

#from nltk.corpus import stopwords
#cachedStopWords = stopwords.words('russian')



def cleanText(s):
    s = s.lower()                         # Convert to lowercase
    s = s.replace(r'<.*?>', ' ')          # Remove HTML characters
    s = s.replace('\'', '')               # Remove single quotes ' 
    s = s.replace('-', '')                # Remove dashes -
    s = s.replace(r'[^a-zA-Z]', ' ')      # Remove non alpha characters
    s = s.strip()                         # Remove whitespace at start and end
    return s

def merge_train_and_test(dataframe):
    sentences = []
    for i in tqdm.tqdm(range(len(dataframe))):
        merged_sentences = ""
        for column_name in ['context_2', 'context_1', 'context_0', 'reply']:
            cur_sentence = dataframe.loc[i, column_name]
            if isinstance(cur_sentence, str):
                if cur_sentence.lower() != "$unk$":
                    merged_sentences += " " + str(cur_sentence)
        merged_sentences = cleanText(merged_sentences)
        sentences.append(merged_sentences)
    return sentences


def objective(params):
    pipe.set_params(**params)
    shuffle = KFold(n_splits=10, shuffle=True)
    score = cross_val_score(pipe, X_train, y_train, cv=shuffle, scoring='neg_mean_squared_error', n_jobs=1)
    return 1-score.mean()

if __name__ == "__main__":
    from nltk.corpus import stopwords
    cachedStopWords = stopwords.words('russian')
    conf = 0.999
    label_to_num = {"bad": 1, "neutral": 0.3, "good": 1 - conf}


    config = Config()

    vocab = get_trimmed_glove_vectors(config.filename_trimmed)
    print ('Word vectors were loaded')

    train = pd.read_csv(config.path_to_splitted_train)
    test = pd.read_csv(config.path_to_splitted_test)
    #train_encoded = pd.read_csv(config.path_to_train)
    #test_encoded = pd.read_csv(config.path_to_test)
    private = pd.read_csv(config.path_to_filtered_test)

    print ('Datasets are loaded')

    X_train = merge_train_and_test(train)
    X_test = merge_train_and_test(test)

    print ('Datasets are preprocessed')


    y_test_labels = np.array([label_to_num[x] for x in test.label])
    y_test = np.array(y_test_labels * test.confidence)

    y_train_labels= np.array([label_to_num[x] for x in train.label])
    y_train = np.array(y_train_labels * train.confidence)


    print ('Target values are preprocessed')

    vectorizer = TfidfVectorizer()
    reg = Ridge()
    pipe = Pipeline([('vec', vectorizer),
                     ('reg', reg)])

    print ('Pipeline is initialized')

    # Parameter search space
    space = {}


    # One of (1,1), (1,2), or (1,3)
    space['vec__ngram_range'] = hp.choice('vec__ngram_range', [(1,1), (1,2), (1,3)])

    # Random integer in [1,3]
    space['vec__min_df'] = 1+hp.randint('vec__min_df', 3)

    # Uniform between 0.7 and 1
    space['vec__max_df'] = hp.uniform('vec__max_df', 0.7, 1.0)
        
    # One of True or False
    space['vec__sublinear_tf'] = hp.choice('vec__sublinear_tf', [True, False])
        
    # Log-uniform between 1e-9 and 1e-4
    space['reg__alpha'] = hp.loguniform('reg__alpha', -9*np.log(10), -4*np.log(10))

    # Random integer in 20:5:80
    space['reg__max_iter'] = 20 + 5*hp.randint('reg__max_iter', 12)

    print ('hyperopt is initialized')

    # The Trials object will store details of each iteration
    trials = Trials()

    print ('Start running')

    # Run the hyperparameter search using the tpe algorithm
    best = fmin(objective,
                space,
                algo=tpe.suggest,
                max_evals=10,
                trials=trials)

    print ('Saving best params')

    # Save the hyperparameter at each iteration to a csv file
    param_values = [x['misc']['vals'] for x in trials.trials]
    param_values = [{key:value for key in x for value in x[key]} for x in param_values]
    param_values = [space_eval(space, x) for x in param_values]

    param_df = pd.DataFrame(param_values)
    param_df['neg_mean_squared_error'] = [1 - x for x in trials.losses()]
    param_df.index.name = 'Iteration'
    param_df.to_csv("../parameter_values2.csv")


    # Get the values of the optimal parameters
    best_params = space_eval(space, best)


    pipe.set_params(**best_params)

    print ('Fitting with the best params')

    pipe.fit(X_train, y_train)

    X_private = merge_train_and_test(private)


    print ('Predicting for private')

    preds = pipe.predict(X_private)

    sorted_test_preds = sort_xgb_predictions(public, 3 - preds)

    print ('Saving submission')

    save_submission("../final_submissions/tfidf_logreg_scaled_private_ho_demo_test.txt", private, sorted_test_preds)

