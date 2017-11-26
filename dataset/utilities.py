"""Loading and operating on the dataset
data from: https://data.world/crowdflower/sentiment-analysis-in-text
"""
import pickle
import os
import numpy as np
import torch
import csv

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)


def where_am_i():
    return os.path.dirname(__file__)


def unpickle(file):
    with open(file, 'rb') as fo:
        object_pickle = pickle.load(fo)
    return object_pickle


def pickle_object(object, filename):
    with open(filename, "wb") as fp:
        pickle.dump(object, fp)


def load_cvs(filename):
    # filename = where_am_i() + "/text_emotion_full.csv"
    with open(filename, "r") as fp:
        data_iter = csv.reader(fp)
        data = [data_row for data_row in data_iter]

    return data


def preprocess_cvs(data):
    final_data = []
    final_labels = []

    original2labels = dict(sadness="sad",
                           boredom="sad",
                           anger="anger",
                           hate="anger",
                           surprise="surprise",
                           happiness="happy",
                           worry="fear",
                           neutral="neutral",
                           relief="happy",
                           love="happy",
                           fun="happy",
                           enthusiasm="happy",
                           empty="empty")
    name2labels = dict(neutral=0,
                       happy=1,
                       sad=2,
                       surprise=3,
                       anger=4,
                       fear=5)
    labels2names = [item for item in name2labels]

    for datarow in data[1:]:
        name = original2labels[datarow[1]]
        if name == "empty":
            continue

        label = name2labels[name]
        final_labels.append(label)
        # text = datarow[3].split()
        text = tknzr.tokenize(datarow[3])

        final_data.append(text)

    # for i in range(20):
    #     print(labels2names[final_labels[i]], final_labels[i], final_data[i])
    return final_data, final_labels, labels2names


def load_csv2lists(filename):
    data = load_cvs(filename)
    final_data, final_labels, labels2names = preprocess_cvs(data)

    return final_data, final_labels, labels2names


if (__name__ == "__main__"):
    final_data, final_labels, labels2names = load_csv2lists("text_emotion.csv")
    pickle_object((final_data, final_labels, labels2names), "dataset_pickled.pkl")
