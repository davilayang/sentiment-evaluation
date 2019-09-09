# Sentiment Classification w/ Tweets

Sentiment Evaluation on Tweets

## 1. Dataset Explorations

## 2. Build baseline model with tf-idf features

+ Pre-processing (with baseline, this is kept as simple as possible):
  + Replace URL with tag "URLLINK"
  + Replace @someuser with tag "USERMENTION"
  + Remove other non-alphaneumeric characters
+ Features:
  + Using lowercasing on each token
  + Using unigram and bigrams
  + Using term-frequency and inverse-document-frequency
+ Classifier:
  + Support Vector Machine with Linear kernel
    + _Accuracy: **0.6575**_
    + _Macro F1 Average Score: **0.63**_

## 3. Build baseline model with word-embedding and sentiment lexicons

+ Pre-processing: same as in baseline model with tf-idf features
+ Features:
  + Using sentiment lexicons for positive/negative word counts
  + Using pre-trained word-vectors
    + sum on each dimension of each tweet
    + average on each dimension of each tweet
+ Classifier:
  + Logistic Regression,
    + _Accuracy: **0.644**_
    + _Macro F1 Average Score: **0.62**_

## 4. Combine baselines models and optimizations

+ 1st. Stage
  + Pre-processing: as done in baseline models
  + Features:
    + baseline tf-idf + baseline world-embedding + sentiment lexicons
  + Classifier:
    + Logistic Regression,
      + _Accuracy: **0.687**_
      + _Macro F1 Average Score: **0.67**_
+ 2st. Stage, improvement based on errors
  + Pre-proecssing:
    + Add function to handle emojis
      + Add function to handle long words, e.g. loooooong
  + Features:
    + Add neutral word counts, excluding stop words
  + Classifier:
    + Logistic Regression,
      + _Accuracy: **0.689**_
      + _Macro F1 Average Score: **0.67**_

