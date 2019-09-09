#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation

import os
import re
import pickle
import functools
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from scipy.sparse import hstack
from nltk.corpus import stopwords
from gensim.models import KeyedVectors


# redefine print function for unbuffered outputs
print = functools.partial(print, flush=True)

import gensim; import sklearn
print('pandas version:  ', pd.__version__, '\t;suggested: 0.23.4')
print('gensim version:  ', gensim.__version__, '\t;suggested: 3.4.0')
print('sklearn version: ', sklearn.__version__, '\t;suggested: 0.20.0')
print(' ')
del gensim; del sklearn 

# all training and testing data should be in directory 'semeval-tweets'
data_dir = 'semeval-tweets'
train_file = 'twitter-training-data.txt' 
# pre-trained vectors should be in 'word-vectors
bin_path = 'D:/DATA/word-vectors/GoogleNews-vectors-negative300.bin.gz'
# bin_path = 'word-vectors/GoogleNews-vectors-negative300.bin.gz'
# sentiment lexicons should be in 'lexicon-dicts/opinion-lexicon-English'
lexicon_path = 'lexicon-dicts/opinion-lexicon-English'

# save and load trained model 
def save_model(clf, model_path): 
    with open(model_path, 'wb') as file:
        pickle.dump(clf, file)

def load_model(model_path): 
    with open(model_path, 'rb') as file:
        clf = pickle.load(file)
    return clf

#replace URLs with "URLLINK"
def replaceURLs(tweet):
	return re.sub(r'https?://[^\s]+', "URLLINK", tweet)

#replace user mentions with "USERMENTION"
def replaceUserMentions(tweet):
	return re.sub(r'(@[A-Za-z0-9_]+)', "USERMENTION", tweet)

#replace all non-alphanumeric
def replaceRest(tweet):
	result = re.sub(r'[^a-zA-Z0-9]', " ", tweet)
	return re.sub(' +',' ', result)

#replace faces/emojis with related words, repeat 3 times for higher modelling effects
#reference on some top common emojis http://emojitracker.com/
def replace_happyf(tweet):
	return re.sub(r'❤️|❤|🙌|😍|😊|💕|😘|☺️|😄|👍|😀|😎|🎉|(:[-]?\))', \
                  'happy happy happy', tweet)

def replace_sadf(tweet):
	return re.sub(r'🤔|🙄|😡|😒|😩|😔|😢|💔|💩|(:\()', \
                  'hate hate hate', tweet)

# replace words with repeated letters, to shorter; e.g. loooong => loong
def replace_long(tweet): 
    return re.compile(r"(.)\1{1,}", re.DOTALL).sub(r"\1\1", tweet)

# load sentiment lexicons
def load_sentiment_lexicons(lexicon_path):
    with open(os.path.join(lexicon_path, 'positive-words.txt')) as infile:
        words_pos = infile.read().split('\n')
        words_pos.pop(-1)
        words_pos = words_pos[30:]

    with open(os.path.join(lexicon_path, 'negative-words.txt')) as infile:
        words_neg = infile.read().split('\n')
        words_neg.pop(-1)
        words_neg = words_neg[31:]
    return words_pos, words_neg

# build features with sentiment lexicons
def count_opinon_lexicons(tweet, words_pos, words_neg):
    tmp = [token.lower() for token in tweet.split(' ')]
    count_pos = len(set(tmp).intersection(set(words_pos)))
    count_neg = len(set(tmp).intersection(set(words_neg)))
    return (np.float64(count_pos), np.float64(count_neg))

def count_opinon_lexicons2(tweet, words_pos, words_neg, stopWords):
    tmp = [token.lower() for token in tweet.split(' ')]
    count_pos = len(set(tmp).intersection(words_pos))
    count_neg = len(set(tmp).intersection(words_neg))
    count_neu = len(set(tmp).difference(stopWords).difference(words_pos).difference(words_neg))
    return (np.float64(count_pos), np.float64(count_neg), np.float64(count_neu))

# build features with word vectors
def tweet_vectors(sent, dim=300, method='sum'): 
    """
    compute different statistics on word vectors, including sum, mean, max, min. 
    """
    tokens = list(filter(None, sent.split(' ')))     # filter out ' ' token
    tmp = np.empty((len(tokens), dim))  # default dimension is 400
    for idx, token in enumerate(tokens): 
        try: 
            tmp[idx, :] = wv_model.get_vector(token)
        except KeyError:
            tmp[idx, :] = 0

    if method == 'sum': return np.sum(tmp, axis=0)
    if method == 'max': return np.max(tmp, axis=0)
    if method == 'min': return np.min(tmp, axis=0)
    if method == 'mean': return np.mean(tmp, axis=0)

col_names = ['id', 'label', 'tweet']; data_type = {'id': 'object', 'label': 'category'}
# load training data
train_path = os.path.join(data_dir, train_file)
data = pd.read_csv(train_path, names=col_names, dtype=data_type, sep='\t')

# replace label values with target as integers
target_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
data['target'] = data.label.replace(to_replace=target_dict)
y_train = data.target

# You may rename the names of the classifiers to something more descriptive
for classifier in ['baseline_tfidf', 'baseline_word_vectors', 'combined_model']: 
    if classifier == 'baseline_tfidf':
        print('Training ' + classifier)
        # data preprocessing: 
        process_funcs = [replaceURLs, replaceUserMentions, replaceRest] 
        for func in process_funcs: data.tweet = data.tweet.apply(func)

        # features extraction: 
        tfidf_vect = TfidfVectorizer(ngram_range=(1, 2), use_idf=True)
        X_train = tfidf_vect.fit_transform(data.tweet)

        # load model if exists, train new if not found
        try: 
            sentiment_clf = load_model('model1_1896548.pickle')
        except IOError: 
            print('...trained model not found, fitting new model...')
            # training sentiment classifier, with linear SVM
            sentiment_clf = LinearSVC()
            sentiment_clf.fit(X=X_train, y=y_train)
            save_model(sentiment_clf, 'model1_1896548.pickle')
        else: 
            print('...trained model found, loading...')

    elif classifier == 'baseline_word_vectors':
        print('Training ' + classifier)
        # data preprocessing: 
        process_funcs = [replaceURLs, replaceUserMentions, replaceRest] 
        for func in process_funcs: data.tweet = data.tweet.apply(func)

        # features extraction: 
        wp, wn = load_sentiment_lexicons(lexicon_path)
        # create features of word counts from sentiment lexicons
        data['count_pos'], \
            data['count_neg'] = zip(*data.tweet.apply(count_opinon_lexicons, args=(wp, wn)))
        # load pre-trained word vectors
        print('...loading pre-trained word vector, this may take some time...')
        wv_model = KeyedVectors.load_word2vec_format(bin_path, binary=True, unicode_errors='ignore')
        print('...loading successful...')
        # create features of word vectors by sum and mean on each dimension
        wv_sum = data.tweet.apply(tweet_vectors, args=(300, 'sum')).apply(pd.Series) 
        wv_mean = data.tweet.apply(tweet_vectors, args=(300, 'mean')).apply(pd.Series) 
        data = pd.concat([data, wv_sum, wv_mean], axis=1)
        # normalize the dataset before training
        scaler = StandardScaler()
        X_train = scaler.fit_transform(data.drop(['id', 'tweet', 'target', 'label'], axis=1))

        # load model if exists, train new if not found
        try: 
            sentiment_clf = load_model('model1_1896548.pickle')
        except IOError: 
            print('...trained model not found, fitting new model...')
            # training sentiment classifier, with Logistic R
            sentiment_clf = LogisticRegression(solver='liblinear', multi_class='ovr')
            sentiment_clf.fit(X=X_train, y=y_train)
            save_model(sentiment_clf, 'model2_1896548.pickle')
        else: 
            print('...trained model found, loading...')

    elif classifier == 'combined_model':
        print('Training ' + classifier)
        stopWords = set(stopwords.words('english'))
        # data preprocessing: 
        process_funcs = [replaceURLs, replaceUserMentions, replace_happyf, \
                        replace_sadf, replace_long, replaceRest]
        for func in process_funcs: data.tweet = data.tweet.apply(func)

        # fetures extraction
        tfidf_vect = TfidfVectorizer(ngram_range=(1, 2), use_idf=True)
        X_train = tfidf_vect.fit_transform(data.tweet)

        # create features of word counts from sentiment lexicons
        wp, wn, = load_sentiment_lexicons(lexicon_path)
        data['count_pos'], data['count_neg'], \
            data['count_neu']= zip(*data.tweet.apply(count_opinon_lexicons2, args=(wp, wn, stopWords)))

        try:
            wv_model
        except NameError: 
                print('...loading pre-trained word vector, this may take some time...')
                wv_model = KeyedVectors.load_word2vec_format(bin_path, binary=True, unicode_errors='ignore')
                print('...loading successful...')
        wv_sum = data.tweet.apply(tweet_vectors, args=(300, 'sum')).apply(pd.Series) 
        wv_mean = data.tweet.apply(tweet_vectors, args=(300, 'mean')).apply(pd.Series) 
        data = pd.concat([data, wv_sum, wv_mean], axis=1)
        # normalize the columns of counts
        scaler = StandardScaler()
        counts = scaler.fit_transform(data[['count_pos', 'count_neg', 'count_neu']])
        # combine all features together
        X_train = hstack([X_train, wv_sum.values, wv_mean.values, counts])

        # load model if exists, train new if not found
        try: 
            sentiment_clf = load_model('model3_1896548.pickle')
        except IOError: 
            print('...trained model not found, fitting new model...')
            # training sentiment classifier, with Logistic R
            sentiment_clf = LogisticRegression(solver='liblinear', multi_class='ovr')
            sentiment_clf.fit(X=X_train, y=y_train)
            save_model(sentiment_clf, 'model3_1896548.pickle')
        else: 
            print('...trained model found, loading...')

    for testset in testsets.testsets:
        # load testset
        test_path = os.path.join(data_dir, testset)
        test = pd.read_csv(test_path, names=col_names, dtype=data_type, sep='\t')
        test['target'] = test.label.replace(to_replace=target_dict)
        y_test = test.target

        # preprocessing and feature extraction on testset
        if classifier == 'baseline_tfidf':
            for func in process_funcs: test.tweet = test.tweet.apply(func)
            X_test = tfidf_vect.transform(test.tweet)

        elif classifier == 'baseline_word_vectors':
            for func in process_funcs: test.tweet = test.tweet.apply(func)
            test['count_pos'], \
                test['count_neg'] = zip(*test.tweet.apply(count_opinon_lexicons, args=(wp, wn)))
            wv_sum = test.tweet.apply(tweet_vectors, args=(300, 'sum')).apply(pd.Series) 
            wv_mean = test.tweet.apply(tweet_vectors, args=(300, 'mean')).apply(pd.Series) 
            test = pd.concat([test, wv_sum, wv_mean], axis=1)
            X_test = scaler.transform(test.drop(['id', 'tweet', 'target', 'label'], axis=1))

        elif classifier == 'combined_model':
            for func in process_funcs: data.tweet = data.tweet.apply(func)
            X_test = tfidf_vect.transform(test.tweet)
            test['count_pos'], test['count_neg'], \
                test['count_neu']= zip(*test.tweet.apply(count_opinon_lexicons2, args=(wp, wn, stopWords)))
            wv_sum_test = test.tweet.apply(tweet_vectors, args=(300, 'sum')).apply(pd.Series) 
            wv_mean_test = test.tweet.apply(tweet_vectors, args=(300, 'mean')).apply(pd.Series) 
            counts_test = scaler.transform(test[['count_pos', 'count_neg', 'count_neu']])
            X_test = hstack([X_test, wv_sum_test.values, wv_mean_test.values, counts_test])

        # creating predictions dictionary
        test_pred = sentiment_clf.predict(X=X_test)
        target_names = list(target_dict.keys())
        predictions = dict(zip(test['id'].values, map(lambda x: target_names[x], test_pred)))
        
        evaluation.evaluate(predictions, test_path, classifier)
        evaluation.confusion(predictions, test_path, classifier)
