# """
# This module is based on :
# youtube videos- "NLTK with Python 3 for Natural Language Processing" by sentdex.

# sentiment - is the function you should use for this module.
# """

import random
import pickle

import nltk
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk import word_tokenize


from statistics import mode

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

class VoteClassifier (ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    # Actual classification:
    # features - is the input for classification.
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return self.most_frequent(votes)

    # How many classifiers agree with the voted classification:
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(self.most_frequent(votes))
        conf = choice_votes / len(votes)
        return conf

    def most_frequent(self, List):
        counter = 0
        num = List[0]

        for i in List:
            curr_frequency = List.count(i)
            if (curr_frequency > counter):
                counter = curr_frequency
                num = i

        return num

short_pos = open("positive_reviews.txt", "r", encoding="utf-8").read()
short_neg = open("negative_reviews.txt", "r", encoding="utf-8").read()

documents = []
all_words = []
allowed_word_types = ["j"]

for p in short_pos.split('\n'):
    documents.append((p,"pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p,"neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

# save_documents:
# save_documents = open("documents.pickle", "wb")
# pickle.dump(documents, save_documents)
# save_documents.close()

# load documents:
save_documents = open("documents.pickle", "rb")
documents = pickle.load(save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000] # word_features-  list of 5000 words appear the most in all movie_reviews.

# save word_features:
# save_word_features = open("word_features.pickle", "wb")
# pickle.dump(word_features, save_word_features)
# save_word_features.close()

# load word_features:
save_word_features = open("word_features.pickle", "rb")
word_features = pickle.load(save_word_features)
save_word_features.close()

def find_features(document):
# for each word in word_features checks if appears in document
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents] # labaled data set

random.shuffle(featuresets)

"""
* Note - you should increase limits of featuresets. I got memory error so needed to take less data, 
  this gave poor results and mistakes in classification.
* You should do the following:
    training_set = featuresets[:10000]
    testing_set = featuresets[10000:]
"""
training_set = featuresets[:1500]
testing_set = featuresets[5000:6000]

# Note - commented classifiers gave poor results when doing final tests.

# NaiveBayesClassifier:
# This algorithm finds the most common words appear in neg/pos reviews.
# classifier = nltk.NaiveBayesClassifier.train(training_set)

# # Save classifier :
# save_classifier = open("naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

# load classifier:
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# print("Original Naive Bayes Algo accurancy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
# classifier.show_most_informative_features(15)


# MNB_classifier:
# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)

# Save classifier :
# save_classifier = open("MNB_classifier.pickle", "wb")
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()

# load classifier:
classifier_f = open("MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()

# print("MNB_classifier accurancy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)


# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier accurancy percent:", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)


# BernoulliNB_classifier:
# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)

# Save classifier :
# save_classifier = open("BernoulliNB_classifier.pickle", "wb")
# pickle.dump(BernoulliNB_classifier, save_classifier)
# save_classifier.close()

# load classifier:
classifier_f = open("BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(classifier_f)
classifier_f.close()

# print("BernoulliNB_classifier accurancy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

# LogisticRegression_classifier:
# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)

# Save classifier :
# save_classifier = open("LogisticRegression_classifier.pickle", "wb")
# pickle.dump(LogisticRegression_classifier, save_classifier)
# save_classifier.close()

# load classifier:
classifier_f = open("LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(classifier_f)
classifier_f.close()

# print("LogisticRegression_classifier accurancy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)


# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
# print("SGDClassifier_classifier accurancy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)


# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier accurancy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

# LinearSVC_classifier
# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)

# Save classifier :
# save_classifier = open("LinearSVC_classifier.pickle", "wb")
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()

# load classifier:
classifier_f = open("LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(classifier_f)
classifier_f.close()

# print("LinearSVC_classifier accurancy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


# NuSVC_classifier:
# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)

# Save classifier :
# save_classifier = open("NuSVC_classifier.pickle", "wb")
# pickle.dump(NuSVC_classifier, save_classifier)
# save_classifier.close()

# load classifier:
classifier_f = open("NuSVC_classifier.pickle", "rb")
NuSVC_classifier = pickle.load(classifier_f)
classifier_f.close()

# print("NuSVC_classifier accurancy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


# Note - classifiers chosen after doing tests and tossing the ones who gave poor results.
voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

# print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

"""
This is the main function:
input - text as string
output - classification of text: is it a positive or negative text.
       - confidence - which percentage of classifiers agree with the voted classification.
"""
def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats), voted_classifier.confidence(feats)
